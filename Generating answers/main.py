import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from utils import read_prompts, generate, save_response

# Konfiguracja
temperature = 0.7
max_tokens = 256
top_k = 100
top_p = 1

input_file = "prompts.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = 'speakleash/Bielik-7B-Instruct-v0.1'
load_4_bit = True

# Tokenizer i streamer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Konfiguracja kwantyzacji
quantization_config = None
if load_4_bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# Wczytanie modelu
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)

# Wczytanie promptów i generowanie odpowiedzi
prompts = read_prompts(input_file)

for index, prompt in enumerate(prompts):
    print(f"--- Generating response for prompt {index+1}: {prompt} ---")
    answer = generate(
        prompt, tokenizer, model, device, streamer,
        max_tokens, temperature, top_k, top_p
    )
    print(f"Response: {answer}")
    save_response(answer, model_name)
    print("Zapisano \n")