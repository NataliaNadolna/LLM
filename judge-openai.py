import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import json
from pathlib import Path


def prepare_batch_file(input_path, model_name, output_dir="batch_requests"):
    """Tworzy pliki JSON do batch API OpenAI."""

    # Wczytuje dane z csv do pandas dataframe
    df = pd.read_csv(input_path)

    # Tworzy folder, jeśli nie istnieje
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Zapisuje do pliku
    filename = os.path.join(output_dir, f"{model_name}.jsonl")
    with open(filename, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            data = {
                "custom_id": f"{model_name}_{index}",  # Indeks wiersza jako DF_ID
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": {"role": "user", "content": f"""ROLA: /"/"/"Jesteś rzetelnym asystentem oceniającym. 
ZAWSZE ściśle trzymasz się instrukcji. Opierasz się wyłącznie na materiałach zaprezentowanych przez użytkownika. 
Wykonujesz TYLKO swoje ZADANIE, pomijasz dodatkowe komentarze./"/"/"

ZADANIE: /"/"/"Twoje zadanie polega wyłącznie na dostarczeniu obiektywnej OCENY odpowiedzi zawartej w treści #OCENIANA_ODPOWIEDZ#. 
Porównaj ją z odpowiedzią zawartą w #PRAWIDLOWA_ODPOWIEDZ#. Są to odpowiedzi na pytanie zawarte w punkcie #PYTANIE#. 
Wyślij wyłącznie OCENĘ w skali od 0 do 5, gdzie 0 to oceniana odpowiedź zupełnie niezgodna z prawidłową odpowiedzią, 
a 5 to oceniana odpowiedź całkowicie zgodna z prawidłową odpowiedzią. 
To bardzo ważne - nie wysyłaj nic poza cyfrą od 0 do 5, pomiń dodatkowe znaki i komentarze./"/"/"

Uwaga! Oceniana odpowiedź może zawierać rozszerzenie tematu lub dodatkowe informacje - nie powinieneś obniżać za to OCENY. 
Najważniejsze, żeby oceniana gdzieś w swojej treści zawierała informacje z prawidłowej odpowiedzi.

#PYTANIE#
{row['Pytanie']}

#OCENIANA_ODPOWIEDZ#
{row['Odpowiedź: speakleash/Bielik-7B-Instruct-v0.1']}

#PRAWIDLOWA_ODPOWIEDZ#
{row['Odpowiedź']}"""},
                    "max_tokens": 50
                }
            }

            # Zapisujemy wiersz JSONL do pliku
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    return filename
    

def main():

    file = prepare_batch_file("responses.csv", "gpt-4o", output_dir="batch_requests")

    # konfiguracja OpenAI API
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

    # przesyłanie pliku wejściowego
    client = OpenAI()

    batch_input_file = client.files.create(
        file=open(f"{file}", "rb"),
        purpose="batch"
    )

    print(batch_input_file)

    # tworzenie batcha
    batch_input_file_id = batch_input_file.id
    
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    batch_id = batch.id
    batch_status = client.batches.retrieve(batch_id)
    print(batch_status)



if __name__ == "__main__":
    main()