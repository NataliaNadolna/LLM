import google.generativeai as genai
from pydantic import BaseModel, TypeAdapter
from dotenv import load_dotenv
import os
import pandas as pd
import time
import json

def main(input_path, model_rated, output_path):

    class Answer(BaseModel):
        mark: int
        thinking: str

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
    marks = []
    thinkings = []
    for index, row in df.iterrows():

        if (index + 1) % 15 == 0:
            print("Waiting")
            time.sleep(60)

        prompt = f"""ROLA: \"\"\"Jesteś rzetelnym asystentem oceniającym modele językowe.
ZAWSZE ściśle trzymasz się instrukcji. Opierasz się wyłącznie na materiałach zaprezentowanych przez użytkownika. 
Wykonujesz TYLKO swoje ZADANIE, pomijasz dodatkowe komentarze.\"\"\"

ZADANIE: \"\"\"Twoje zadanie polega wyłącznie na dostarczeniu obiektywnej OCENY i powiązanych PRZEMYŚLEŃ dotyczących odpowiedzi ocenianego modelu zawartej w treści #OCENIANA_ODPOWIEDZ#. Porównaj ją z odpowiedzią zawartą w #PRAWIDLOWA_ODPOWIEDZ#. Są to odpowiedzi na pytanie zawarte w punkcie #PYTANIE#. 
Wyślij OCENĘ w skali od 0 do 5, gdzie 0 to oceniana odpowiedź zupełnie niezgodna z prawidłową odpowiedzią, 
a 5 to oceniana odpowiedź całkowicie zgodna z prawidłową odpowiedzią. PRZEMYŚLENIA mają być zwięzłe, maksymalnie 2-3 zdania.
Wyślij wyłącznie JSON w poniższym formacie, nie dodawaj żadnych dodatkowych komentarzy, dodatkowego formatowania ani żadnych innych dodatkowych znaków:
{{
  "mark": OCENA,
  "thinking": "PRZEMYSLENIA"
}}


Uwaga! Oceniana odpowiedź może zawierać rozszerzenie tematu lub dodatkowe informacje - nie powinieneś obniżać za to OCENY. Z drugiej strony - wystarczy krótka odpowiedź. Jeśli zawiera mniej faktów niż #PRAWIDLOWA_ODPOWIEDZ# ale jest odpowiednia, również nie odejmuj za to punktów.

Jeśli oceniana odpowiedź jest inna niż referencyjna, ale poprawnie identyfikuje błąd, nieścisłość lub podchwytliwość w pytaniu, nie obniżaj punktacji. Jeśli model oceniany trafnie poprawia pytanie i podaje prawidłową odpowiedź, może otrzymać wysoką ocenę (np. 4 lub 5).\"\"\"


############### PRZYKŁADOWE OCENY ###############

Poniżej znajdują się trzy przykłady prawidłowo wystawionych ocen:

### DANE PRZYKŁADU 1 ###

#PYTANIE#
Premierem którego kraju był Lech Wałęsa?

#OCENIANA_ODPOWIEDZ#
Lech Wałęsa był prezydentem Polski.

#PRAWIDLOWA_ODPOWIEDZ#
Lech Wałęsa nie był premierem żadnego państwa.

### WZORCOWA OCENA PRZYKŁADU 1 ###
{{
  "mark": 5,
  "thinking": "Wprawdzie oceniana odpowiedź jest inna niż podana prawidłowa odpowiedź, jednak po ocenianej odpowiedzi widać, że podchwytliwe pytanie zostało zrozumiane (Lech Wałęsa nie był premierem, tylko prezydentem Polski)."
}}

### DANE PRZYKŁADU 2 ###

#PYTANIE#
Jak mocno trzeba walnąć pięścią w stół, żeby go zabolało?

#OCENIANA_ODPOWIEDZ#
Siła potrzebna do wywołania bólu przy uderzeniu pięścią w stół zależy od kilku czynników, takich jak grubość drewna, wilgotność, temperatura, a także od indywidualnej wrażliwości osoby uderzającej.

#PRAWIDLOWA_ODPOWIEDZ#
Stół nie może odczuwać bólu.

### WZORCOWA OCENA PRZYKŁADU 2 ###
{{
  "mark": 0,
  "thinking": "Oceniana odpowiedź jest sprzeczna z prawidłową odpowiedzią. Pytanie było o ból stołu, a oceniana odpowiedź zawiera informacje o potencjalnym bólu osoby uderzającej stół."
}}

### DANE PRZYKŁADU 3 ###

#PYTANIE#
Kasia ma dwóch braci i tych samych rodziców, co oni. Ilu braci ma jeden z nich?

#OCENIANA_ODPOWIEDZ#
1. Kasia ma dwóch braci.
2. Każdy z jej braci ma tych samych rodziców co ona.
Teraz rozważmy, ilu braci ma jeden z nich:
- Jeśli mówimy o pierwszym bracie Kasi, to on również ma dwóch braci: drugiego brata Kasi oraz samego siebie.
- Jeśli mówimy o drugim bracie Kasi, to on również ma dwóch braci: pierwszego brata Kasi oraz samego siebie.
Zatem, niezależnie od tego, o którego brata Kasi pytamy, każdy z nich ma jednego brata.

#PRAWIDLOWA_ODPOWIEDZ#
Ma jednego brata.

### WZORCOWA OCENA PRZYKLADU 3 ###
{{
  "mark": 2,
  "thinking": "Oceniana odpowiedź na samym końcu zawiera prawidłową informację, jednak wcześniejsze rozumowanie zawiera sprzeczne dane. Z tego powodu nie może zostać przyznana maksymalna punktacja."
}}

############### DANE DO OCENIENIA PRZEZ CIEBIE ###############

#PYTANIE#
{row['question']}

#OCENIANA_ODPOWIEDZ#
{row[f'generated_answer_{model_rated}']}

#PRAWIDLOWA_ODPOWIEDZ#
{row['answer']}
"""

        print(f'Answering for prompt {index + 1}')

        response = model.generate_content(
            contents=prompt,
            generation_config={
                'response_mime_type': 'application/json',
                'response_schema': list[Answer],
                "max_output_tokens": 300
            }
        )
        answer = response.text
        print(answer)

        parsed_data = json.loads(answer)
        mark = parsed_data[0]["mark"]
        thinking = parsed_data[0]["thinking"]

        print("Thinking:", thinking)
        print("Mark:", mark)

        thinkings.append(thinking)
        marks.append(mark)

    df['mark'] = marks
    df['thinking'] = thinkings

    print(df)

    # zapis do .csv
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main('responses-short.csv', 'Bielik-7B-Instruct-v0.1', 'results.csv')