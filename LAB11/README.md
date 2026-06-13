# LAB11 - Apache Spark Structured Streaming

## Temat

Przetwarzanie strumieniowe danych w Apache Spark Structured Streaming.

## Sposob uruchomienia

Projekt nalezy uruchomic w PyCharmie z pliku:

```text
lab11_structured_streaming.py
```

Interpreter powinien miec zainstalowany pakiet `pyspark`. Program tworzy lokalna sesje Spark, przygotowuje folder wejsciowy, uruchamia generator plikow CSV i przetwarza dane strumieniowo bez restartu aplikacji.

## Wersja Spark/PySpark

Wersja Spark/PySpark jest wypisywana w konsoli po uruchomieniu programu. Na ekranie nalezy pokazac linie:

```text
Spark/PySpark version: ...
```

## Zadanie 1 - przygotowanie srodowiska

Utworzono aplikacje PySpark uruchamiajaca lokalna sesje Spark:

```python
SparkSession.builder.appName("LAB11_StructuredStreaming").master("local[1]").getOrCreate()
```

Program wypisuje wersje Pythona, Spark/PySpark oraz folder projektu.

## Zadanie 2 - strumieniowe wczytywanie danych

Dane sa wczytywane z folderu:

```text
data/input_stream
```

Zrodlem strumienia sa pliki CSV dodawane w trakcie pracy programu. Zdefiniowano schemat z kolumnami:

- `event_time`
- `user_id`
- `category`
- `amount`
- `status`

Program sprawdza, czy DataFrame jest strumieniowy przez `df.isStreaming`, a nastepnie wypisuje schemat przez `printSchema()`.

## Zadanie 3 - transformacje i agregacje

Wykonano transformacje:

- konwersja `event_time` na typ timestamp;
- odfiltrowanie rekordow z niepoprawnym czasem lub kwota;
- filtrowanie tylko statusu `paid`;
- wyliczenie dodatkowej kolumny `amount_with_tax`.

Nastepnie wykonano agregacje wedlug kategorii:

- liczba zdarzen;
- suma wartosci `amount`.

Wynik jest wypisywany do konsoli w trybie `complete`.

## Zadanie 4 - okna czasowe i watermarking

Utworzono agregacje w oknach czasowych 10 minut:

```python
withWatermark("event_time", "10 minutes")
groupBy(window(col("event_time"), "10 minutes"), col("category"))
```

Program zawiera tez porownanie:

- okien stalych 10-minutowych;
- okien przesuwajacych 10-minutowych z przesunieciem co 5 minut.

Czas zdarzenia to czas zapisany w danych, np. moment dokonania transakcji. Czas przetwarzania to moment, w ktorym Spark fizycznie odczytuje rekord. Watermarking pozwala Sparkowi okreslic, jak dlugo czekac na spoznione dane i kiedy mozna domknac stan agregacji.

## Zadanie 5 - zapis wynikow i checkpointing

Wyniki okien czasowych sa zapisywane do folderu:

```text
data/output_stream
```

Checkpointy sa zapisywane w:

```text
checkpoints/category_summary
checkpoints/window_summary
```

Po pierwszej fazie program zatrzymuje zapytania strumieniowe, dodaje nowy plik i uruchamia przetwarzanie ponownie z tym samym checkpointem. Dzieki temu Spark nie przetwarza ponownie tych samych plikow wejsciowych, tylko kontynuuje od nowego pliku.

## Batch a streaming

Przetwarzanie batch dziala na skonczonym zestawie danych i konczy sie po wykonaniu zapytania. Streaming dziala na danych naplywajacych w czasie i aktualizuje wyniki po pojawieniu sie nowych rekordow.

## Tryby output mode

`append` dopisuje tylko nowe gotowe wyniki. `update` aktualizuje tylko te wiersze, ktore zmienily sie od ostatniego mikro-batcha. `complete` wypisuje cala tabele wynikowa po kazdej aktualizacji.

## Checkpointing a zwykly zapis

Zwykly zapis przechowuje tylko wynik. Checkpointing przechowuje stan zapytania strumieniowego, informacje o przetworzonych plikach i metadane potrzebne do kontynuacji po restarcie.

## Screeny do sprawozdania

1. `SCREEN 1` - wersja Spark/PySpark i uruchomienie aplikacji.
2. `SCREEN 2` - `isStreaming=True`, schemat strumienia i wyniki agregacji w konsoli.
3. `SCREEN 3` - komunikaty generatora pokazujace dodawanie kolejnych plikow CSV bez restartu.
4. `SCREEN 4` - restart z tym samym checkpointem i przetworzenie nowego pliku.
5. `SCREEN 5` - zapis wynikow do CSV oraz ponowny odczyt jako zwykly batch DataFrame.
6. `SCREEN 6` - porownanie okien stalych i przesuwajacych.
7. `SCREEN 7` - poprawne zakonczenie programu bez bledu.
