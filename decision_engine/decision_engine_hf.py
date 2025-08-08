import torch
import os
import platform
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
try:
    # Available in transformers >= 4.30
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Login to Hugging Face Hub if token is provided via env
if HF_API_TOKEN:
    try:
        login(token=HF_API_TOKEN)
    except Exception:
        # Proceed without login; public models may still be accessible
        pass

MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
USE_4BIT = os.getenv("HF_LOAD_IN_4BIT", "0") == "1"  # disabled by default on Windows
USE_8BIT = os.getenv("HF_LOAD_IN_8BIT", "0") == "1"
ATTN_IMPL = os.getenv("HF_ATTN_IMPL", "auto")  # options: auto|sdpa|flash_attention_2

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

# Model (auto device map, dtype adjusts for CPU/GPU)
has_cuda = torch.cuda.is_available()
is_windows = platform.system().lower().startswith("win")
torch_dtype = torch.float16 if has_cuda else torch.float32

quantization_config: Optional[BitsAndBytesConfig] = None  # type: ignore
can_use_bnb = False
if has_cuda and not is_windows and BitsAndBytesConfig is not None:
    try:
        import bitsandbytes as bnb  # noqa: F401
        can_use_bnb = True
    except Exception:
        can_use_bnb = False

if can_use_bnb and (USE_4BIT or USE_8BIT):
    try:
        if USE_4BIT:
            quantization_config = BitsAndBytesConfig(  # type: ignore
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif USE_8BIT:
            quantization_config = BitsAndBytesConfig(  # type: ignore
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False,
            )
    except Exception:
        quantization_config = None

model_load_kwargs = {
    "device_map": "auto",
    "torch_dtype": torch_dtype,
}
if has_cuda:
    # Prefer faster attention when available
    model_load_kwargs["attn_implementation"] = ATTN_IMPL if ATTN_IMPL != "auto" else "sdpa"
if quantization_config is not None:
    model_load_kwargs.pop("torch_dtype", None)
    model_load_kwargs["quantization_config"] = quantization_config

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_load_kwargs,
    )
except Exception:
    # Safe fallback: no quantization, CPU or CUDA fp16
    fallback_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **fallback_kwargs)
model.eval()
try:
    model.config.use_cache = True
except Exception:
    pass

if has_cuda:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# Optional: live decoding streamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def format_chat_prompt(user_text: str, system_text: str = "You are a helpful assistant.") -> Dict[str, torch.Tensor]:
    """
    Formats input for chat-tuned models. Prefer the tokenizer's chat template.
    Fallback to Mistral Instruct-style [INST] prompt if template unavailable.
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        # Build a template string first, then tokenize to get attention_mask
        template_str = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = tokenizer(
            template_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {k: v.to(model.device) for k, v in encoded.items()}
    except Exception:
        # Fallback for tokenizers without chat templates
        prompt = f"[INST] <<SYS>>{system_text}<</SYS>> {user_text} [/INST]"
        encoded = tokenizer(prompt, return_tensors="pt")
        return {k: v.to(model.device) for k, v in encoded.items()}

def generate_answer(prompt: str, temperature: float = 0.1, max_tokens: int = 256) -> str:
    # Build chat-formatted input for Instruct model
    encoded_inputs = format_chat_prompt(prompt)

    with torch.inference_mode():
        outputs = model.generate(
            **encoded_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            streamer=None,  # set to `streamer` for live token printing
        )

    # Since we used a chat template, slice only the newly generated tail
    prompt_length = encoded_inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print(generate_answer("Explain RAG in one paragraph.")) 
