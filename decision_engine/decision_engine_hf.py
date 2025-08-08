import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Login to Hugging Face Hub
login(os.getenv("HF_API_TOKEN"))

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Model (quantized 4-bit, auto device map)
model = AutoModelForCausalLM.from_pretrained( 
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16    # Or float32 if float16 gives error
)

# Optional: live decoding streamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def format_chat_prompt(user_text: str, system_text: str = "You are a helpful assistant.") -> torch.Tensor:
    # Llama 3 Instruct expects chat messages; use the built-in chat template[7][12]
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}, 
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    return input_ids

def generate_answer(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    # Build chat-formatted input for Instruct model
    input_ids = format_chat_prompt(prompt)

    # Define safe terminators for Llama 3 chat models[7][13]
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=terminators,  # stop at end-of-turn
            streamer=None,  # set to `streamer` for live token printing
        )

    # Since we used a chat template, slice only the newly generated tail
    generated = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print(generate_answer("Explain RAG in one paragraph."))
