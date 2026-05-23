# Hackology II — Object Detection Challenge

## Harmonogram

- **Start hackathonu:** sobota **2026-05-23 12:00** (Europe/Warsaw)
- **Deadline submissions:** niedziela **2026-05-24 12:00** (Europe/Warsaw)
- **Format:** 24h on-site na uczelni
- **Komunikacja:** Discord — `TODO: invite link` (uzupełnione przed startem)

## Zadanie

Wykrywanie obiektów (object detection) w formacie COCO. Metryka: **mAP@0.5**.

## Szybki start

```bash
# 1. Zainstaluj zależności
uv sync

# 2. Pobierz dane
./download_data.sh

# 3. Uruchom baseline
uv run predict --input data/ --output submissions/predictions.json

# 4. Zgłoś wynik
git add submissions/predictions.json
git commit -m "submission: baseline yolov8n"
git push
```

## Struktura repo

```
├── predict.py              # Twój skrypt predykcji (interfejs CLI)
├── taxonomy.json           # Mapowanie kategorii
├── download_data.sh        # Pobranie datasetu
├── checksums.sha256        # Weryfikacja integralności danych
├── pyproject.toml          # Zależności (uv)
├── submissions/            # Tutaj wrzucasz predictions.json
│   └── predictions.json
├── notebooks/
│   └── 01_exploration.ipynb  # EDA notebook
│   └── 01_train_colab.ipynb  # EDA notebook
└── .github/workflows/
    └── submit.yml          # Auto-submit po pushu do submissions/
```

## Interfejs predict.py

**Wymagany interfejs CLI** — ewaluacja prywatna uruchomi Twój skrypt dokładnie tak:

```bash
uv run predict --input <DIR_Z_OBRAZAMI> --annotations test_images.json --output predictions.json
```

**WAŻNE:** Predykcje generuj na zdjęciach z `test_images.json` — to jest zbiór testowy
na którym odbywa się ewaluacja. Plik zawiera listę zdjęć i ich ID (bez anotacji).
Predykcje na innych zdjęciach zostaną odrzucone (`unknown image_id`).

Plik `predictions.json` musi być listą w formacie COCO:

```json
[
  {
    "image_id": 1,
    "category_id": 1,
    "bbox": [x, y, width, height],
    "score": 0.95
  }
]
```

- `bbox` w formacie `[x, y, w, h]` (lewy-górny róg + rozmiar)
- `score` w zakresie `(0, 1]`
- `image_id` i `category_id` zgodne z `taxonomy.json`

## Zgłaszanie wyników

1. Umieść `predictions.json` w katalogu `submissions/`
2. Commituj i pushuj — workflow automatycznie wyśle zgłoszenie
3. Wynik (mAP@0.5) pojawi się jako **commit status** na GitHubie

### Limity

- **5 zgłoszeń na godzinę**
- **30 zgłoszeń łącznie**
- Maksymalny rozmiar pliku: **50 MB**

## Ewaluacja finalna

Na koniec hackathonu oznacz swój najlepszy commit tagiem `final`:

```bash
git tag final
git push origin final
```

Jeśli zapomnisz — użyjemy Twojego ostatniego zgłoszenia z najlepszym wynikiem.

Ewaluacja finalna odbywa się na **zbiorze prywatnym** (innym niż publiczny leaderboard).

## Środowisko ewaluacji prywatnej

Twój kod zostanie uruchomiony w następującym środowisku:

| Parametr | Wartość |
|----------|---------|
| Python | 3.11 |
| CUDA | 12.x |
| GPU | NVIDIA T4 (16 GB VRAM) |
| Timeout | 30 minut (łącznie z pobieraniem wag) |
| Package manager | uv (`uv sync` z Twojego `uv.lock`) |

**WAŻNE:**
- `uv.lock` MUSI być w repo — bez niego ewaluacja się nie powiedzie
- Jeśli model pobiera wagi z sieci, czas pobierania wlicza się w 30-minutowy timeout
- Testuj lokalnie: `uv run predict --input data/ --output test.json`

## Trening w Google Colab

Nie masz GPU? Możesz trenować w Google Colab (darmowe T4 — identyczne jak środowisko ewaluacji).

1. Otwórz `notebooks/02_train_colab.ipynb` w Colab:
   - Pobierz plik z repo → [colab.research.google.com](https://colab.research.google.com/) → File → Upload notebook
2. Ustaw runtime na GPU: Runtime → Change runtime type → T4 GPU
3. Postępuj zgodnie z instrukcjami w notebooku

Notebook przeprowadzi Cię przez: setup → pobranie danych → fine-tuning YOLOv8 → submission.

## Dane

Po uruchomieniu `./download_data.sh` w katalogu `data/` znajdziesz:
- Obrazy do detekcji
- Anotacje treningowe (format COCO)
- `taxonomy.json` — mapowanie kategorii

### Jak rozpoznać dane syntetyczne

Zbiór treningowy w `data/train/` zawiera obrazy pochodzące z różnych źródeł.
Nie są one rozdzielone do osobnego katalogu `data/synthetic/`.

Rozróżnienie jest zapisane w metadanych COCO w pliku `data/train/annotations.json`,
w polu `source_dataset` dla każdego obrazu:

- `SIDG_TRAIN` — obrazy syntetyczne
- `SIDG_SYNTH_TRAIN` — obrazy syntetyczne

Przykład rekordu w `images`:

```json
{
  "id": 3427,
  "file_name": "image.9d843c17-7a14-4516-8699-f97e07629142.rgb_b97c2f79ae52a723322dbbd41012a442.jpg",
  "width": 1080,
  "height": 1920,
  "source_dataset": "SIDG_SYNTH_TRAIN"
}
```

Jeśli chcesz trenować tylko na części danych albo porównać wyniki dla różnych źródeł,
filtruj obrazy po `images[*].source_dataset` i jawnie dokumentuj przyjęty podział w README.

Sprawdź notebook `notebooks/01_exploration.ipynb` żeby zapoznać się z danymi.

## Leaderboard

Aktualny ranking: https://hackology-2026.github.io/eval-runner/

Auto-refresh co 60 sekund. Żeby zobaczyć wynik swoje ostatniego zgłoszenia, poczekaj ~10 sekund na refresh'a i sprawdź swoje miejsce na liście.

## Regulamin

1. Każda drużyna pracuje na swoim prywatnym repo
2. Zabronione jest kopiowanie rozwiązań innych drużyn
3. Dozwolone są pretrenowane modele (np. z ultralytics, torchvision)
4. `uv.lock` musi być w repo
5. `predict.py` musi implementować wymagany interfejs CLI
6. Deadline jest egzekwowany automatycznie — zgłoszenia po terminie będą odrzucone
