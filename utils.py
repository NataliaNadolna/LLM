import torch

def read_prompts(file_path):
    """Wczytuje prompty z pliku tekstowego."""
    prompts = []
    with open(file_path, "r") as file:
        for line in file:
            prompts.append(line.strip())
    return prompts

def generate(prompt, tokenizer, model, device, streamer, max_tokens, temperature, top_k, top_p, system=None):
    """Generuje odpowiedź modelu na podstawie promptu."""
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = input_ids

    if torch.cuda.is_available():
        model_inputs = input_ids.to(device)

    outputs = model.generate(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True if temperature else False,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_response = full_response.split("[/INST]")[-1].strip()

    return cleaned_response

def save_response(response, model_name):
    """Zapisuje odpowiedź modelu do pliku."""
    safe_model_name = model_name.replace('/', '_')
    file_path = f"answers-{safe_model_name}.txt"
    with open(file_path, "a+", encoding="utf-8") as file:
        file.write(f"{response}\n")