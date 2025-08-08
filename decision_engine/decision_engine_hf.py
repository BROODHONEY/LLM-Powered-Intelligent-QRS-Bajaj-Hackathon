import torch
import os
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
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

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

# Model (auto device map, dtype adjusts for CPU/GPU)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch_dtype,
)

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

def generate_answer(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    # Build chat-formatted input for Instruct model
    encoded_inputs = format_chat_prompt(prompt)

    with torch.no_grad():
        outputs = model.generate(
            **encoded_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=None,  # set to `streamer` for live token printing
        )

    # Since we used a chat template, slice only the newly generated tail
    prompt_length = encoded_inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print(generate_answer("Explain RAG in one paragraph.")) 
