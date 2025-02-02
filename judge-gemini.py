import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import time


def main(input_path, model_rated, output_path):

    # konfiguracja google api
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    # wczytanie modelu
    version = 'models/gemini-1.5-flash'
    model = genai.GenerativeModel(version)

    # wczytanie danych
    df = pd.read_csv(input_path)

    # ocenianie odpowiedzi
    responses = []
    for index, row in df.iterrows():

        if (index + 1) % 15 == 0:
            print("Waiting")
            time.sleep(60)

        prompt = f"""ROLA: /"/"/"Jesteś rzetelnym asystentem oceniającym. 
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
{row['answer']}"""
        
        print(f'Answering for promot {index + 1}')
        response = model.generate_content(prompt)
        responses.append(response.text.strip())


    # dodanie ocen
    df['Score'] = responses
    print(df)

    # zapis do .csv
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main('responses.csv', 'Bielik-7B-Instruct-v0.1', 'results.csv')