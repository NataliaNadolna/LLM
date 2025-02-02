import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import json
from pathlib import Path


def main(input_path, model_judge, model_rated, output_dir):
    """Tworzy pliki JSON do batch API OpenAI."""

    # Wczytuje dane z csv do pandas dataframe
    df = pd.read_csv(input_path)

    # Tworzy folder, jeśli nie istnieje
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Zapisuje do pliku
    filename = os.path.join(output_dir, f"{model_judge}.jsonl")
    with open(filename, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            data = {
                "custom_id": f"{model_rated}_{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_judge,
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
{row['question']}

#OCENIANA_ODPOWIEDZ#
{row[f'generated_answer_{model_rated}']}

#PRAWIDLOWA_ODPOWIEDZ#
{row['answer']}"""},
                    "max_tokens": 5
                }
            }

            # Zapisujemy wiersz JSONL do pliku
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')



if __name__ == "__main__":
    main(input_path="responses.csv", model_judge="gpt-4o", model_rated="Bielik-7B-Instruct-v0.1", output_dir="batch_requests")