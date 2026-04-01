Pipeline dataset du projet Colab.

Usage local:

```bash
cd /Users/glenneriss/Documents/projets/location-maison-model-for-google-collab
python3 scripts/dataset/prepare_sources.py
PYTHONPATH=src python -m location_maison_model_annonce.cli.generate_dataset --config config/project.yaml
```

Sources attendues:
- `data/source_annonces/*.annonce.json` : annonces deja normalisees
- `data/source_properties/*.json` ou `*.jsonl` : export Firestore `properties`
- `data/post-for-facebook/*.json` : posts Facebook bruts pour filtrage de pertinence

Sorties utiles:
- `data/processed/relevance/offers.jsonl`
- `data/processed/relevance/requests.jsonl`
- `data/processed/relevance/uncertain.jsonl`
- `data/processed/relevance/irrelevant.jsonl`
- `data/datasets/train.jsonl`
- `data/datasets/validation.jsonl`
- `data/datasets/test.jsonl`
