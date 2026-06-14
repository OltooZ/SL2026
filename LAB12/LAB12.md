# Politechnika Bydgoska im. Jana i Jędrzeja Śniadeckich
Wydział Telekomunikacji,  
Informatyki i Elektrotechniki  
Nowoczesne Technologie Przetwarzania Danych  
Laboratorium 12  
Temat: Business Intelligence - wizualizacja i analiza danych w narzędziu Metabase


## Cel ćwiczenia

Celem ćwiczenia jest praktyczne poznanie procesu Business Intelligence (BI); załadowanie wcześniej przetworzonych danych do bazy analitycznej; podłączenie narzędzia BI (Metabase) do bazy danych; tworzenie zapytań, wykresów i interaktywnego dashboardu; definiowanie wskaźników (KPI) oraz przygotowanie wyników w formie raportu. Laboratorium pokazuje, jak dane przetworzone we wcześniejszych ćwiczeniach (np. w Apache Spark) zamienić w czytelną informację dla odbiorcy biznesowego.

## Materiały

https://www.metabase.com/docs/latest/  
https://www.metabase.com/learn/  
https://www.postgresql.org/docs/  
https://docs.docker.com/  
https://superset.apache.org/docs/intro

## Zadanie 1: Przygotowanie środowiska

- Upewnij się, że masz zainstalowany Docker oraz Docker Compose.

- Uruchom dwa kontenery: bazę danych PostgreSQL (hurtownia danych analitycznych) oraz narzędzie BI Metabase.

- Sprawdź, czy interfejs Metabase jest dostępny w przeglądarce pod adresem `http://localhost:3000` i utwórz konto administratora.

- W sprawozdaniu podaj wersje użytych obrazów oraz sposób uruchomienia środowiska.

Przykład startowy (`docker-compose.yml`):

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: bi
      POSTGRES_PASSWORD: bi
      POSTGRES_DB: ntpd
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  metabase:
    image: metabase/metabase:latest
    ports:
      - "3000:3000"
    depends_on:
      - postgres

volumes:
  pgdata:
```

Uruchomienie:

```bash
docker compose up -d
```

## Zadanie 2: Załadowanie danych do bazy analitycznej

- Przygotuj dane wejściowe. Najlepiej wykorzystaj dane przetworzone w poprzednich laboratoriach (np. wynik agregacji ze Spark z LAB09-LAB11) albo zbiór transakcji o polach: czas zdarzenia, identyfikator użytkownika, kategoria, wartość, status.

- Załaduj dane do bazy PostgreSQL do tabeli (np. `transactions`).

- W Metabase dodaj połączenie do bazy danych (typ: PostgreSQL, host: `postgres` lub `localhost`, port: `5432`, baza: `ntpd`).

- Sprawdź w Metabase, że tabela jest widoczna i można przejrzeć jej zawartość.

Przykład załadowania danych (Python):

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://bi:bi@localhost:5432/ntpd")

df = pd.read_csv("data/transactions.csv")
df.to_sql("transactions", engine, if_exists="replace", index=False)

print("Załadowano wierszy:", len(df))
```

Przykładowy format rekordu:

```csv
event_time,user_id,category,amount,status
2026-05-01 10:00:00,u001,books,39.99,paid
```

## Zadanie 3: Pierwsze pytania (questions) i wykresy

- Utwórz w Metabase co najmniej trzy pytania (`questions`):

  - jedno zbudowane wizualnym kreatorem zapytań (bez pisania SQL);
  - jedno z agregacją według kategorii (np. liczba zdarzeń i suma wartości);
  - jedno zapisane jako zapytanie SQL.

- Dla każdego pytania wybierz odpowiedni typ wizualizacji: tabela, wykres słupkowy, liniowy albo kołowy.

- W sprawozdaniu wyjaśnij, dlaczego dany typ wykresu pasuje do danego pytania.

Przykład pytania SQL:

```sql
SELECT category,
       COUNT(*)   AS events,
       SUM(amount) AS revenue
FROM transactions
WHERE status = 'paid'
GROUP BY category
ORDER BY revenue DESC;
```

## Zadanie 4: Budowa dashboardu

- Utwórz dashboard i umieść na nim minimum cztery karty (wcześniej przygotowane pytania).

- Dodaj co najmniej jeden filtr dashboardu (parametr), np. zakres dat, kategorię albo status, i połącz go z kartami.

- Sprawdź interaktywność: zmiana filtra powinna aktualizować wszystkie powiązane karty.

- W sprawozdaniu dołącz zrzut ekranu gotowego dashboardu.

Wskazówka: dobry dashboard zaczyna się od najważniejszych wskaźników na górze (np. łączna sprzedaż, liczba zamówień), a dopiero niżej znajdują się wykresy szczegółowe.

Na maksymalną ocenę 5 dodaj parametr czasu i pokaż analizę trendu sprzedaży w czasie (wykres liniowy z grupowaniem po dniu lub tygodniu).

## Zadanie 5: Wskaźniki, analiza i udostępnianie wyników

- Zdefiniuj minimum dwa wskaźniki biznesowe (KPI), np. łączny przychód, średnia wartość transakcji, udział transakcji opłaconych.

- Przygotuj analizę odpowiadającą na konkretne pytanie biznesowe, np. "które kategorie generują największy przychód?" albo "jak zmienia się sprzedaż w czasie?".

- Pokaż sposób udostępnienia wyników: zapisanie pytania w kolekcji, eksport danych do CSV albo udostępnienie dashboardu (link publiczny lub subskrypcja).

- W sprawozdaniu opisz różnice między:

  - przetwarzaniem danych a warstwą Business Intelligence;
  - dashboardem a raportem statycznym;
  - zapytaniem ad-hoc a zdefiniowanym wskaźnikiem.

Na maksymalną ocenę 5 porównaj Metabase z innym narzędziem BI (np. Apache Superset, Power BI albo Grafana) i opisz, kiedy które rozwiązanie jest wygodniejsze.

UWAGA: Rozwiązanie zadania należy przesłać w aplikacji Teams. Rozwiązaniem może być link do repozytorium GitHub/GitLab zawierającego kod źródłowy (pliki `docker-compose.yml`, skrypt ładujący dane, ewentualne dane przykładowe) oraz plik `README.md`. Plik `README.md` będzie traktowany jako sprawozdanie: należy w nim opisać sposób uruchomienia projektu, odpowiedzieć na pytania z zadań, a także dodać zrzuty ekranu z wykonania ćwiczeń (połączenie z bazą, pytania, dashboard).
