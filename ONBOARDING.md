# 🚀 Onboarding — Start tutaj

Jesteś członkiem drużyny w **Hackology II**. Ten dokument przeprowadzi Cię przez pierwsze kroki.

## Przygotowanie (15 minut)

### 1. Zainstaluj narzędzia

```bash
# uv — package manager (jeśli nie masz)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Zweryfikuj
uv --version
```

### 2. Sklonuj to repo

```bash
git clone https://github.com/HACKOLOGY-ORG/TEAM-XXX-detector
cd TEAM-XXX-detector
```

### 3. Zainstaluj zależności

```bash
uv sync
```

### 4. Pobierz dane

```bash
bash download_data.sh
```

**Chwila — sprawdź:**
```bash
ls -la data/
# Powinieneś widzieć train/ i public_test/
```

## Eksploracja danych (30 minut)

Otwórz notebook:

```bash
uv run jupyter notebook notebooks/01_exploration.ipynb
```

Zapoznaj się z:
- Strukturą obrazów
- Format anotacji (COCO)
- Kategorie produktów w `taxonomy.json`
- Przykładowe bounding boxy

### Uwaga o źródłach danych

W `data/train/annotations.json` każdy obraz ma pole `source_dataset`.
To identyfikator źródła technicznego danych, przydatny do filtrowania i analiz.

Ważne:
- obrazy z różnych źródeł są zmieszane w `data/train/images/`,
- nie ma osobnego katalogu `data/synthetic/`,
- obrazy ze źródeł `SIDG_TRAIN` i `SIDG_SYNTH_TRAIN` są syntetyczne.

Jeśli chcesz porównywać warianty treningu między źródłami, filtruj dane po
`images[*].source_dataset` z pliku COCO.

## Trening w Google Colab (opcjonalnie)

Nie masz GPU? Otwórz `notebooks/02_train_colab.ipynb` w [Google Colab](https://colab.research.google.com/) — darmowe T4, identyczne jak środowisko ewaluacji. Notebook przeprowadzi Cię przez cały proces: od setup po submission.

## Uruchomienie baseline (15 minut)

Spróbuj domyślnego modelu:

```bash
uv run predict --input data/public_test/images/ --output test_predictions.json
```

Sprawdź format:

```bash
python3 -c "import json; d = json.load(open('test_predictions.json')); print(f'Predykcji: {len(d)}')"
```

## Twoje pierwsze zgłoszenie (5 minut)

1. Skopiuj predykcje do formalnego katalogu:
   ```bash
   cp test_predictions.json submissions/predictions.json
   ```

2. Commituj i pushuj:
   ```bash
   git add submissions/predictions.json
   git commit -m "submission: baseline yolov8n"
   git push
   ```

3. Czekaj na wynik w **commit status** (GitHub Actions):
   ```bash
   # Zobaczy się tutaj:
   # https://github.com/HACKOLOGY-ORG/TEAM-XXX-detector/commits/main
   ```

## Limity i deadline

| Parametr | Wartość |
|----------|---------|
| Maksymalnie zgłoszeń/godz | 5 |
| Maksymalnie zgłoszeń łącznie | 30 |
| Max rozmiar predictions.json | 50 MB |
| Deadline | **[UZUPEŁNIĆ: data i godzina]** |
| Leaderboard | https://hackology-xxx.github.io/eval-runner/ |

**Ważne:** Zgłoszenia po deadline'ie będą automatycznie odrzucone.

## Format submissions

Twój plik `submissions/predictions.json` musi być listą JSON:

```json
[
  {
    "image_id": 1,
    "category_id": 1,
    "bbox": [100, 50, 200, 150],
    "score": 0.95
  },
  ...
]
```

**Wymogi:**
- `image_id` — musi istnieć w danych
- `category_id` — musi istnieć w `taxonomy.json`
- `bbox` — format `[x, y, width, height]` (lewy-górny róg + wymiary)
- `score` — wartość od 0 do 1

Plik z nieprawidłowym formatem będzie odrzucony + zobaczysz błąd w commit status.

## Ewaluacja finalna

Na koniec hackathonu oznacz swoje **najlepsze rozwiązanie** tagiem:

```bash
git tag final
git push origin final
```

**Jeśli zapomnisz:** Użyjemy Twojego ostatniego zgłoszenia z najlepszym wynikiem.

Ewaluacja finalna będzie na innym zbiorze danych (private test) w następującym środowisku:

| Parametr | Spec |
|----------|------|
| Python | 3.11 |
| CUDA | 12.x |
| GPU | NVIDIA T4 (16 GB VRAM) |
| Timeout | 30 minut |
| Instalacja | `uv sync` z Twojego `uv.lock` |

**Zasady:**
- Twój `uv.lock` MUSI być w repo
- Jeśli pobierasz wagi modelu z sieci, czas wlicza się w 30-minutowy timeout
- Interfejs: `uv run predict --input <DIR> --output predictions.json`

## Gdzie szukać pomocy?

### Problemy techniczne

**UZUPEŁNIĆ: Support contacts**
- Discord channel: `#[UZUPEŁNIĆ]`
- Email: `[UZUPEŁNIĆ]`
- GitHub Issues: [link do repo]

### Pytania ogólne

- FAQ: [UZUPEŁNIĆ]
- Regulamin: patrz `README.md`

## Checklist przed finalnym submisjem

- [ ] Testowałem `uv run predict` lokalnie
- [ ] `uv.lock` jest w repo i aktualny
- [ ] `predictions.json` ma prawidłowy format (sprawdzony JSON + liczba predykcji > 0)
- [ ] Wszystkie `image_id` i `category_id` są w zbiorze
- [ ] Rozmiar pliku < 50 MB
- [ ] Commitnąłem tag `final` na najlepszym komicie
- [ ] `git push origin final` wyszło bez błędów

## Szybkie komendy

```bash
# Sync zależności
uv sync

# Uruchom baseline
uv run predict --input data/public_test/images/ --output submissions/predictions.json

# Walidacja formatu JSON
python3 -c "import json; json.load(open('submissions/predictions.json'))"

# Push do GitHub
git add submissions/predictions.json
git commit -m "submission: [OPIS]"
git push

# Oznaczenie finalnego rozwiązania
git tag final && git push origin final
```

---

**Powodzenia! 🚀**
