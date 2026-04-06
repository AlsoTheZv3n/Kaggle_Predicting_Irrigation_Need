# Kaggle Playground S6E4 — Predicting Irrigation Need

Lösung für die [Kaggle Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4/overview).

- **Task:** Vorhersage des Bewässerungsbedarfs (`Irrigation_Need`) als 3-Klassen-Problem
- **Klassen:** `Low` / `Medium` / `High`
- **Metric:** Balanced Accuracy
- **Deadline:** 30. April 2026

## Pipeline

LightGBM + CatBoost Ensemble mit:

1. **Feature Engineering** — Interaktions-Features (Multiplikation, Division), Polynome, Row-Statistiken
2. **Label Encoding** — Target und kategoriale Features
3. **LightGBM** — 5-Fold Stratified CV, `class_weight="balanced"`, Multi-Seed (`[42, 2024, 7]`)
4. **CatBoost** — 5-Fold Stratified CV, `auto_class_weights="Balanced"`, `eval_metric="TotalF1:average=Macro"`, Multi-Seed
5. **Ensemble** — Optimales Blending-Gewicht via Nelder-Mead auf OOF Balanced Accuracy
6. **Threshold-Tuning** — Per-Klassen-Probability-Scaling (kompensiert Class-Bias)
7. **Submission** — `submission.csv` im erwarteten Kaggle-Format

## Dateien

| Datei | Zweck |
|---|---|
| [kaggle_s6e4_irrigation.ipynb](kaggle_s6e4_irrigation.ipynb) | **Haupt-Notebook** — auf Kaggle hochladen |
| [kaggle_s6e4_irrigation.py](kaggle_s6e4_irrigation.py) | Skript-Variante (gleiche Logik wie Notebook) |
| [Task.md](Task.md) | Original-Aufgabenbeschreibung |
| [sample_submission.csv](sample_submission.csv) | Kaggle-Sample (270 000 Test-IDs) |
| [test_pipeline.py](test_pipeline.py) | Lokaler Smoke-Test (FE + Encoding + Submission-Format) |
| [run_notebook_test.py](run_notebook_test.py) | End-to-End-Test des Notebooks mit synthetischen Daten |

## Lokale Tests ausführen

```bash
# Schneller Logik-Test (kein lightgbm/catboost nötig)
python test_pipeline.py

# Echter End-to-End-Test mit synthetischen Daten
pip install lightgbm catboost nbclient nbformat ipykernel seaborn
python -m ipykernel install --user --name python3
python run_notebook_test.py
```

## Workflow auf Kaggle

1. Kaggle → Code → New Notebook → File → Import Notebook → [kaggle_s6e4_irrigation.ipynb](kaggle_s6e4_irrigation.ipynb) hochladen
2. Rechts unter **Data** → "Add Data" → `playground-series-s6e4` Dataset hinzufügen
3. **Run All**
4. Nach erstem Run die echten `train.columns` checken — falls die Spaltennamen nicht
   `Temperature`/`Humidity`/`Soil_Moisture`/... heißen, in Cell 12 die `interaction_pairs`
   anpassen
5. `submission.csv` → Submit to Competition

## Optionales Hyperparameter-Tuning

In Cell 26 `RUN_OPTUNA = True` setzen — startet 50 Optuna-Trials für LightGBM
(Laufzeit ~30–60 min auf Kaggle).

## Hinweise

- **Multi-Seed verdreifacht die Trainingszeit.** Falls Kaggle-Zeit knapp wird:
  `SEEDS = [42, 2024]` oder `n_estimators` reduzieren.
- **CatBoost < 1.0** kennt evtl. `TotalF1:average=Macro` nicht — Fallback: `"TotalF1"`.
- **`is_unbalance` wurde durch `class_weight="balanced"` ersetzt** — `is_unbalance` ist
  primär ein Binary-Parameter und greift bei Multiclass kaum.
