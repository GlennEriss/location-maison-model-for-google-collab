Pipeline de synchronisation entre le code local, GitHub et Google Drive.

Convention recommandee:

- GitHub:
  - `src/`
  - `config/`
  - `scripts/`
  - `requirements.txt`
  - `README.md`
  - `.gitignore`
- Google Drive:
  - `code/`
  - checkpoints
  - metrics
  - reports
  - datasets generes

Scripts:

- `init_drive_layout.sh <drive_project_dir>`
  - cree l'arborescence Drive recommandee
- `sync_code_to_drive.sh <drive_project_dir>`
  - pousse uniquement le code et la configuration vers `code/`
- `push_artifacts_to_drive.sh <drive_project_dir>`
  - pousse les artefacts locaux vers `outputs/` et `data/`
- `pull_artifacts_from_drive.sh <drive_project_dir>`
  - recupere checkpoints, metrics, reports et datasets depuis Drive
- `status_watch.sh <config_path> [interval_seconds]`
  - affiche en boucle le contenu interprete de `RUN_STATE.json`

Exemple:

```bash
cd /Users/glenneriss/Documents/projets/location-maison-model-for-google-collab
./scripts/pipeline/init_drive_layout.sh "/Users/glenneriss/Library/CloudStorage/GoogleDrive-xxx/My Drive/IA/location-maison-model-for-google-collab"
./scripts/pipeline/sync_code_to_drive.sh "/Users/glenneriss/Library/CloudStorage/GoogleDrive-xxx/My Drive/IA/location-maison-model-for-google-collab"
./scripts/pipeline/pull_artifacts_from_drive.sh "/Users/glenneriss/Library/CloudStorage/GoogleDrive-xxx/My Drive/IA/location-maison-model-for-google-collab"
./scripts/pipeline/status_watch.sh config/project.yaml 15
```
