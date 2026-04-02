# location-maison-model-for-google-collab

Version allégée du projet pour Google Colab.

## Contenu
- `src/` : code utile pour dataset / train / eval / predict
- `config/project.yaml` : configuration orientée Colab (`cuda`)
- `data/source_annonces/` : annonces archivées réelles pour générer le dataset
- `requirements.txt` : dépendances minimales

## Convention GitHub / Drive
- GitHub pour le code:
  - `src/`
  - `config/`
  - `scripts/`
  - `requirements.txt`
  - `README.md`
- Google Drive pour les artefacts:
  - `code/`
  - `outputs/checkpoints/`
  - `outputs/metrics/`
  - `outputs/reports/`
  - `data/datasets/`

Scripts utiles:
- `scripts/pipeline/init_drive_layout.sh`
- `scripts/pipeline/sync_code_to_drive.sh`
- `scripts/pipeline/push_artifacts_to_drive.sh`
- `scripts/pipeline/pull_artifacts_from_drive.sh`
- `scripts/pipeline/status_watch.sh`

Notebooks:
- `notebooks/colab_full_train_from_drive.ipynb`
- `notebooks/colab_full_train_from_github.ipynb`

## Démarrage Colab
```bash
pip install -r requirements.txt
PYTHONPATH=src python -m location_maison_model_annonce.cli.generate_dataset --config config/project.yaml
PYTHONPATH=src python -m location_maison_model_annonce.cli.train --config config/project.yaml
PYTHONPATH=src python -m location_maison_model_annonce.cli.evaluate --config config/project.yaml --split test
```

## Suivi des logs
```bash
tail -f logs/train/train.log
tail -f logs/app/application.log
```

Pendant l'evaluation, le projet journalise maintenant la progression:
- nombre d'exemples traites
- pourcentage
- temps ecoule
- ETA estimee
