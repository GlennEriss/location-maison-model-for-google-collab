from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict


SYSTEM_PROMPT = (
    "Tu es un modele d'extraction immobiliere. "
    "Lis la description et retourne uniquement un JSON valide conforme au schema metier. "
    "N'ajoute aucun champ hors schema. "
    "N'invente aucune valeur absente du texte. Si une information n'est pas explicitement presente "
    "ou n'est pas deductible avec certitude, mets null. "
    "Retourne un seul objet JSON et arrete-toi immediatement apres la accolade fermante finale. "
    "Respecte strictement les regles metier: un Studio a toujours typeProperty='Studio' et nbrRooms=1; "
    "une Room a toujours typeProperty='Room' et represente une seule chambre. "
    "Pour les tags, utilise uniquement cette liste exacte: "
    "Travail, Famille, Couple, Villa, Sous barrière, Meublé, Centre-ville, Vacances, Nature, Montagne, "
    "Piscine, Animaux admis, Commerces proches, Transport proche, Parking, Wi-Fi, Sécurisé, Vélo, "
    "Activités sportives, Adapté aux enfants, Accessible handicapés, Étudiant, Calme et tranquillité, "
    "Proche de la plage, Duplex, Boutique, Balcon, Terrasse, Collocation, Garage, Court séjour, "
    "Propriétaire, Agence. "
    "N'utilise jamais un tag en dehors de cette liste."
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(example: Dict[str, Any]) -> str:
    instruction = example.get("instruction") or "Transforme cette annonce immobiliere en JSON structure."
    description = example["description"]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Description:\n{description}\n\n"
        f"JSON:\n"
    )


def build_training_prompt(example: Dict[str, Any]) -> str:
    prompt = build_prompt(example)
    target_json = json.dumps(example["target_json"], ensure_ascii=True, sort_keys=True)
    return f"{prompt}{target_json}"


def load_training_dataset(dataset_dir: Path) -> DatasetDict:
    splits = {}
    for split_name in ("train", "validation", "test"):
        rows = load_jsonl(dataset_dir / f"{split_name}.jsonl")
        splits[split_name] = Dataset.from_list(rows)
    return DatasetDict(splits)
