import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd

def main():

    # konfiguracja google api
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    # wczytanie modelu
    version = 'models/gemini-1.5-flash'
    model = genai.GenerativeModel(version)

    # wczytanie danych
    file_path = "data.xlsx"
    df = pd.read_excel(file_path)

    # ocenianie odpowiedzi
    responses = []
    for index, row in df.iterrows():
        prompt = f"""ROLA: /"/"/"Jesteś rzetelnym asystentem oceniającym. 
ZAWSZE ściśle trzymasz się instrukcji. Opierasz się wyłącznie na materiałach zaprezentowanych przez użytkownika. 
Wykonujesz TYLKO swoje ZADANIE, pomijasz dodatkowe komentarze./"/"/"

ZADANIE: /"/"/"Twoje zadanie polega wyłącznie na dostarczeniu obiektywnej OCENY odpowiedzi zawartej w treści #OCENIANA_ODPOWIEDZ#. 
Porównaj ją z odpowiedzią zawartą w #PRAWIDLOWA_ODPOWIEDZ#. Są to odpowiedzi na pytanie zawarte w punkcie #PYTANIE#. 
Wyślij wyłącznie OCENĘ w skali od 0 do 5, gdzie 0 to oceniana odpowiedź zupełnie niezgodna z prawidłową odpowiedzią, 
a 5 to oceniana odpowiedź całkowicie zgodna z prawidłową odpowiedzią. 
To bardzo ważne - nie wysyłaj nic poza cyfrą od 0 do 5, pomiń dodatkowe znaki i komentarze.

Uwaga! Oceniana odpowiedź może zawierać rozszerzenie tematu lub dodatkowe informacje - nie powinieneś obniżać za to OCENY. 
Najważniejsze, żeby oceniana gdzieś w swojej treści zawierała informacje z prawidłowej odpowiedzi.

#PYTANIE#
{row['Prompt']}

#OCENIANA_ODPOWIEDZ#
{row['Model_answer']}

#PRAWIDLOWA_ODPOWIEDZ#
{row['True_answer']}"""
        
        response = model.generate_content(prompt)
        responses.append(response.text.strip())

    # dodanie ocen
    df['Model_Score'] = responses
    print(df)

    # zapis do .csv
    df.to_csv('results.csv', index=False)

    # zapis do .xlsx
    df2 = df.copy()
    with pd.ExcelWriter('results.xlsx') as writer:
        df.to_excel(writer, sheet_name='Sheet_name_1')

if __name__ == "__main__":
    main()