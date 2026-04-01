from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any

from location_maison_model_annonce.data.dataset_spec import DatasetComposition, DatasetManifest, ExampleRecord


ALLOWED_TAGS = [
    "Travail",
    "Famille",
    "Couple",
    "Villa",
    "Sous barrière",
    "Meublé",
    "Centre-ville",
    "Vacances",
    "Nature",
    "Montagne",
    "Piscine",
    "Animaux admis",
    "Commerces proches",
    "Transport proche",
    "Parking",
    "Wi-Fi",
    "Sécurisé",
    "Vélo",
    "Activités sportives",
    "Adapté aux enfants",
    "Accessible handicapés",
    "Étudiant",
    "Calme et tranquillité",
    "Proche de la plage",
    "Duplex",
    "Boutique",
    "Balcon",
    "Terrasse",
    "Collocation",
    "Garage",
    "Court séjour",
    "Propriétaire",
    "Agence",
]
ALLOWED_TYPES = {
    "Home",
    "Studio",
    "Apartment",
    "Desk",
    "Building",
    "Shop",
    "Kiosk",
    "Room",
    "Property",
    "Logement",
    "Villa",
    "Land",
}
TYPE_ALIASES = {
    "Appartement": "Apartment",
    "appartement": "Apartment",
    "Maison": "Home",
    "maison": "Home",
    "villa": "Villa",
    "VILLA": "Villa",
    "studio": "Studio",
    "STUDIO": "Studio",
    "room": "Room",
    "ROOM": "Room",
    "boutique": "Shop",
    "BOUTIQUE": "Shop",
    "bureau": "Desk",
    "BUREAU": "Desk",
    "immeuble": "Building",
    "IMMEUBLE": "Building",
    "terrain": "Land",
    "Terrain": "Land",
    "kiosque": "Kiosk",
    "Kiosque": "Kiosk",
}
STANDARD_INSTRUCTIONS = [
    "Transforme cette annonce immobiliere en JSON structure.",
    "Analyse cette annonce et retourne uniquement le JSON metier.",
    "Lis cette annonce et extrais les champs immobiliers attendus en JSON.",
]
TEXT_VARIANTS = (
    "verbatim",
    "title_plus_source",
    "description_plus_source",
    "compact",
    "noisy",
    "social_broker",
    "social_compact",
    "bullet_sheet",
    "caps_broker",
    "mixed_sentences",
)
EXTERNAL_POST_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\ba louer\b",
        r"\bà louer\b",
        r"\ba vendre\b",
        r"\bà vendre\b",
        r"\bstudio\b",
        r"\bappartement\b",
        r"\bvilla\b",
        r"\bmaison\b",
        r"\bterrain\b",
        r"\bimmeuble\b",
        r"\bbureau\b",
        r"\bboutique\b",
        r"\bkiosque\b",
        r"\bchambre\b",
    )
]


@dataclass
class SourceAnnonce:
    source_id: str
    type_property: str
    source_text: str
    title: str
    description: str
    target_json: dict[str, Any]
    metadata: dict[str, Any]


def generate_bootstrap_dataset(config: dict[str, Any], dataset_dir: Path) -> DatasetManifest:
    generator = random.Random(int(config["project"]["seed"]))
    dataset_cfg = config["dataset"]
    total_examples = int(dataset_cfg["initial_target_size"])
    composition = DatasetComposition(
        total_examples=total_examples,
        standard_cases_ratio=float(dataset_cfg["composition"]["standard_cases_ratio"]),
        hard_cases_ratio=float(dataset_cfg["composition"]["hard_cases_ratio"]),
    )

    source_records = load_supervised_sources(config)
    if not source_records:
        raise ValueError("No supervised sources could be loaded to build the dataset.")

    records = build_records(generator, source_records, dataset_cfg["split"], composition)
    records = records[:total_examples]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "all": str(dataset_dir / dataset_cfg["output_filename"]),
        "train": str(dataset_dir / "train.jsonl"),
        "validation": str(dataset_dir / "validation.jsonl"),
        "test": str(dataset_dir / "test.jsonl"),
        "manifest": str(dataset_dir / "dataset_manifest.json"),
    }
    write_jsonl(Path(output_files["all"]), records)
    for split in ("train", "validation", "test"):
        write_jsonl(Path(output_files[split]), [record for record in records if record.split == split])

    manifest = build_manifest(records, output_files, total_examples)
    Path(output_files["manifest"]).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_supervised_sources(config: dict[str, Any]) -> list[SourceAnnonce]:
    records: list[SourceAnnonce] = []
    records.extend(load_archived_annonces(config))
    records.extend(load_property_exports(config))
    return deduplicate_sources(records)


def load_archived_annonces(config: dict[str, Any]) -> list[SourceAnnonce]:
    dataset_cfg = config["dataset"]
    source_dir = Path(dataset_cfg.get("source_annonce_dir", "data/source_annonces"))
    files = sorted(source_dir.glob("*.annonce.json"))
    records: list[SourceAnnonce] = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        annonce = payload.get("annonce") or {}
        normalized = normalize_annonce(annonce)
        if normalized is None:
            continue
        source_text = clean_text(annonce.get("sourceText")) or compose_fallback_source_text(annonce)
        if not source_text:
            continue
        records.append(
            SourceAnnonce(
                source_id=str(annonce.get("sourceId") or file_path.stem),
                type_property=normalized["typeProperty"],
                source_text=source_text,
                title=clean_text(annonce.get("title")),
                description=clean_text(annonce.get("description")),
                target_json=normalized,
                metadata={
                    "file_name": file_path.name,
                    "city": clean_text(annonce.get("city")),
                    "street": clean_text(annonce.get("street")),
                    "source_kind": "archived-real-annonce",
                },
            )
        )
    return records


def load_property_exports(config: dict[str, Any]) -> list[SourceAnnonce]:
    dataset_cfg = config["dataset"]
    source_dir = Path(dataset_cfg.get("source_properties_dir", "data/source_properties"))
    if not source_dir.exists():
        return []

    records: list[SourceAnnonce] = []
    for file_path in sorted(source_dir.glob("*")):
        if file_path.suffix not in {".json", ".jsonl"}:
            continue
        payloads = read_property_payloads(file_path)
        for index, payload in enumerate(payloads):
            normalized = normalize_property_document(payload)
            if normalized is None:
                continue
            records.append(
                SourceAnnonce(
                    source_id=str(payload.get("id") or payload.get("propertyId") or payload.get("sourceId") or f"{file_path.stem}-{index}"),
                    type_property=normalized["typeProperty"],
                    source_text=extract_property_source_text(payload),
                    title=clean_text(payload.get("title")),
                    description=clean_text(payload.get("description")),
                    target_json=normalized,
                    metadata={
                        "file_name": file_path.name,
                        "city": clean_text(payload.get("city")),
                        "street": clean_text(payload.get("street")),
                        "source_kind": "property-export",
                    },
                )
            )
    return [record for record in records if record.source_text]


def read_property_payloads(file_path: Path) -> list[dict[str, Any]]:
    if file_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(unwrap_payload(payload))
        return rows

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [unwrap_payload(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("documents"), list):
            return [unwrap_payload(item) for item in payload["documents"] if isinstance(item, dict)]
        if isinstance(payload.get("items"), list):
            return [unwrap_payload(item) for item in payload["items"] if isinstance(item, dict)]
        return [unwrap_payload(payload)]
    return []


def unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("annonce"), dict):
        base = dict(payload["annonce"])
        for key in ("id", "propertyId", "sourceId"):
            if key in payload and key not in base:
                base[key] = payload[key]
        return base
    return payload


def normalize_property_document(payload: dict[str, Any]) -> dict[str, Any] | None:
    normalized = normalize_annonce(payload)
    if normalized is None:
        return None
    return normalized


def deduplicate_sources(records: list[SourceAnnonce]) -> list[SourceAnnonce]:
    deduped: list[SourceAnnonce] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for record in records:
        key = (
            record.type_property,
            normalize_string(record.target_json.get("status")) or "",
            str(record.target_json.get("price") or ""),
            compact_text(record.source_text)[:240],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(record)
    return deduped


def normalize_annonce(annonce: dict[str, Any]) -> dict[str, Any] | None:
    type_property = normalize_type(annonce.get("typeProperty"))
    status = annonce.get("status")
    if type_property not in ALLOWED_TYPES:
        return None
    if status not in {"FOR_RENT", "FOR_SALE"}:
        return None

    target = {
        "typeProperty": type_property,
        "area": normalize_number(annonce.get("area"), allow_zero=False),
        "price": normalize_number(annonce.get("price"), allow_zero=False),
        "tags": normalize_tags(annonce.get("tags")),
        "status": status,
        "contact": extract_contact(annonce.get("sourceText")) or extract_contact(annonce.get("description")) or extract_contact(annonce.get("contact")),
        "nbrRooms": normalize_number(annonce.get("nbrRooms"), allow_zero=False),
        "nbrKitchens": normalize_number(annonce.get("nbrKitchens"), allow_zero=False),
        "nbrBathrooms": normalize_number(annonce.get("nbrBathrooms"), allow_zero=False),
        "nbrToilets": normalize_number(annonce.get("nbrToilets"), allow_zero=False),
        "nbrFloorApartment": normalize_number(annonce.get("nbrFloorApartment"), allow_zero=False),
        "numeroApartment": normalize_string(annonce.get("numeroApartment")),
        "nbrApartments": normalize_number(annonce.get("nbrApartments"), allow_zero=False),
        "nbrFloors": normalize_number(annonce.get("nbrFloors"), allow_zero=False),
        "hasParking": normalize_bool(annonce.get("hasParking")),
        "nbrGarages": normalize_number(annonce.get("nbrGarages"), allow_zero=False),
        "nbrLivingRoom": normalize_number(annonce.get("nbrLivingRoom") or annonce.get("nbrSalons"), allow_zero=False),
        "nbrFloorStudio": normalize_number(annonce.get("nbrFloorStudio"), allow_zero=False),
        "numeroStudio": normalize_string(annonce.get("numeroStudio")),
        "nbrPiscine": normalize_number(annonce.get("nbrPiscine"), allow_zero=False),
        "nbrToilet": normalize_number(annonce.get("nbrToilet"), allow_zero=False),
        "kioskType": normalize_string(annonce.get("kioskType")),
        "roomType": normalize_string(annonce.get("roomType")),
    }

    apply_type_rules(target)
    return target


def apply_type_rules(target: dict[str, Any]) -> None:
    type_property = target["typeProperty"]

    if type_property == "Studio":
        target["nbrRooms"] = 1
    if type_property == "Room":
        target["nbrRooms"] = 1
        target["nbrKitchens"] = None
        target["nbrBathrooms"] = None
        target["nbrToilets"] = None
        target["nbrFloorStudio"] = None
        target["numeroStudio"] = None
    if type_property not in {"Apartment"}:
        target["nbrFloorApartment"] = None
        target["numeroApartment"] = None
    if type_property not in {"Studio"}:
        target["nbrFloorStudio"] = None
        target["numeroStudio"] = None
    if type_property not in {"Building"}:
        target["nbrApartments"] = None
        target["hasParking"] = None
    if type_property not in {"Building", "Home", "Villa"}:
        target["nbrFloors"] = None
    if type_property not in {"Home", "Villa"}:
        target["nbrGarages"] = None
        target["nbrLivingRoom"] = None
    if type_property != "Villa":
        target["nbrPiscine"] = None
    if type_property != "Shop":
        target["nbrToilet"] = None
    if type_property != "Kiosk":
        target["kioskType"] = None
    if type_property != "Room":
        target["roomType"] = None


def build_records(
    generator: random.Random,
    source_records: list[SourceAnnonce],
    split_cfg: dict[str, float],
    composition: DatasetComposition,
) -> list[ExampleRecord]:
    generator.shuffle(source_records)
    split_by_source = assign_splits(source_records, split_cfg)
    difficulty_sequence = build_difficulty_sequence(len(source_records), composition)
    variant_counts = build_variant_counts(len(source_records), composition.total_examples)

    records: list[ExampleRecord] = []
    example_index = 1
    for source_record, variant_count, default_difficulty in zip(source_records, variant_counts, difficulty_sequence):
        for variant_index in range(variant_count):
            difficulty = choose_variant_difficulty(default_difficulty, variant_index)
            description = render_variant(generator, source_record, difficulty, variant_index)
            records.append(
                ExampleRecord(
                    example_id=f"annonce-{example_index:05d}",
                    split=split_by_source[source_record.source_id],
                    difficulty=difficulty,
                    instruction=generator.choice(STANDARD_INSTRUCTIONS),
                    description=description,
                    target_json=dict(source_record.target_json),
                    metadata={
                        "source": source_record.metadata["source_kind"],
                        "source_id": source_record.source_id,
                        "seed_file": source_record.metadata["file_name"],
                        "property_type": source_record.type_property,
                        "city": source_record.metadata.get("city"),
                        "street": source_record.metadata.get("street"),
                        "contains_noise": difficulty == "hard",
                        "variant_index": variant_index,
                    },
                )
            )
            example_index += 1
    return records


def assign_splits(source_records: list[SourceAnnonce], split_cfg: dict[str, float]) -> dict[str, str]:
    total_sources = len(source_records)
    train_count = int(total_sources * float(split_cfg["train"]))
    validation_count = int(total_sources * float(split_cfg["validation"]))
    mapping: dict[str, str] = {}
    for index, record in enumerate(source_records):
        if index < train_count:
            mapping[record.source_id] = "train"
        elif index < train_count + validation_count:
            mapping[record.source_id] = "validation"
        else:
            mapping[record.source_id] = "test"
    return mapping


def build_difficulty_sequence(total_sources: int, composition: DatasetComposition) -> list[str]:
    hard_sources = int(total_sources * composition.hard_cases_ratio)
    return ["hard"] * hard_sources + ["standard"] * (total_sources - hard_sources)


def build_variant_counts(source_count: int, target_size: int) -> list[int]:
    base = max(1, target_size // source_count)
    remainder = target_size % source_count
    return [base + (1 if index < remainder else 0) for index in range(source_count)]


def choose_variant_difficulty(default_difficulty: str, variant_index: int) -> str:
    if variant_index == 0:
        return "standard"
    if default_difficulty == "hard" and variant_index % 2 == 1:
        return "hard"
    return "standard"


def render_variant(
    generator: random.Random,
    source_record: SourceAnnonce,
    difficulty: str,
    variant_index: int,
) -> str:
    variant_kind = TEXT_VARIANTS[variant_index % len(TEXT_VARIANTS)]
    if variant_kind == "verbatim":
        text = source_record.source_text
    elif variant_kind == "title_plus_source":
        prefix = source_record.title or type_label(source_record.type_property)
        text = f"{prefix}\n{source_record.source_text}".strip()
    elif variant_kind == "description_plus_source":
        suffix = source_record.description or source_record.title
        text = f"{source_record.source_text}\n{suffix}".strip()
    elif variant_kind == "compact":
        text = compact_text(source_record.source_text)
    elif variant_kind == "social_broker":
        text = social_broker_text(generator, source_record)
    elif variant_kind == "social_compact":
        text = social_compact_text(generator, source_record)
    elif variant_kind == "bullet_sheet":
        text = bullet_sheet_text(generator, source_record)
    elif variant_kind == "caps_broker":
        text = caps_broker_text(generator, source_record)
    elif variant_kind == "mixed_sentences":
        text = mixed_sentences_text(generator, source_record)
    else:
        text = noisy_text(generator, source_record.source_text)

    if difficulty == "hard" and variant_kind not in {"noisy", "social_compact"}:
        text = noisy_text(generator, text)
    return text


def social_broker_text(generator: random.Random, source_record: SourceAnnonce) -> str:
    target = source_record.target_json
    chunks = [f"{offer_label(target['status'])} {type_label(source_record.type_property).upper()}"]
    if target.get("nbrRooms"):
        chunks.append(f"{target['nbrRooms']} chambres")
    if target.get("nbrBathrooms"):
        chunks.append(f"{target['nbrBathrooms']} douches")
    if target.get("nbrKitchens"):
        chunks.append(f"{target['nbrKitchens']} cuisine")
    if target.get("area"):
        chunks.append(f"{target['area']} m2")
    if target.get("price"):
        chunks.append(f"prix {target['price']} fcfa")
    if target.get("contact"):
        chunks.append(f"contact {target['contact']}")
    suffix = generator.choice(["visite 5000", "agence", "dispo immediat", "serieux seulement"])
    return " | ".join(chunks + [suffix])


def social_compact_text(generator: random.Random, source_record: SourceAnnonce) -> str:
    target = source_record.target_json
    base = compact_text(source_record.source_text)
    snippets = []
    if target.get("price"):
        snippets.append(f"{target['price']}f")
    if target.get("contact"):
        snippets.append(target["contact"])
    if target.get("tags"):
        snippets.append(generator.choice(target["tags"]).lower())
    generator.shuffle(snippets)
    if snippets:
        base = f"{base} {' '.join(snippets)}"
    return noisy_text(generator, base)


def bullet_sheet_text(generator: random.Random, source_record: SourceAnnonce) -> str:
    target = source_record.target_json
    lines = [f"{offer_label(target['status'])} - {type_label(source_record.type_property)}"]
    if target.get("nbrRooms") is not None:
        lines.append(f"- chambres: {target['nbrRooms']}")
    if target.get("nbrBathrooms") is not None:
        lines.append(f"- douches: {target['nbrBathrooms']}")
    if target.get("nbrToilets") is not None:
        lines.append(f"- toilettes: {target['nbrToilets']}")
    if target.get("nbrKitchens") is not None:
        lines.append(f"- cuisines: {target['nbrKitchens']}")
    if target.get("nbrFloors") is not None:
        lines.append(f"- niveaux: {target['nbrFloors']}")
    if target.get("nbrApartments") is not None:
        lines.append(f"- appartements: {target['nbrApartments']}")
    if target.get("area") is not None:
        lines.append(f"- superficie: {target['area']} m2")
    if target.get("price") is not None:
        lines.append(f"- prix: {target['price']} fcfa")
    if target.get("contact"):
        lines.append(f"- contact: {target['contact']}")
    if target.get("tags"):
        lines.append(f"- atouts: {', '.join(target['tags'][:3])}")
    if source_record.description:
        lines.append(source_record.description)
    return "\n".join(lines).strip()


def caps_broker_text(generator: random.Random, source_record: SourceAnnonce) -> str:
    target = source_record.target_json
    parts = [
        f"{offer_label(target['status'])} {type_label(source_record.type_property).upper()}",
        generator.choice(["DISPONIBLE", "URGENT", "BONNE AFFAIRE", "OPPORTUNITE"]),
    ]
    if target.get("nbrRooms"):
        parts.append(f"{target['nbrRooms']} CHAMBRES")
    if target.get("price"):
        parts.append(f"{target['price']}F")
    if target.get("area"):
        parts.append(f"{target['area']}M2")
    if target.get("contact"):
        parts.append(f"CONTACT {target['contact']}")
    if target.get("tags"):
        parts.append(generator.choice(target["tags"]).upper())
    return " | ".join(parts)


def mixed_sentences_text(generator: random.Random, source_record: SourceAnnonce) -> str:
    text = source_record.source_text
    pieces = [chunk.strip() for chunk in re.split(r"[.;!\n]+", text) if chunk.strip()]
    if len(pieces) <= 1:
        pieces = text.split(", ")
    pieces = [piece for piece in pieces if piece]
    if len(pieces) > 2:
        generator.shuffle(pieces)
    intro = generator.choice(
        [
            "Annonce immobiliere:",
            "Details du bien:",
            "Publication:",
            "Informations:",
        ]
    )
    merged = " ".join(pieces[: min(len(pieces), 6)])
    return f"{intro} {merged}".strip()


def compact_text(text: str) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"[\n\r]+", text) if chunk.strip()]
    compact = " ".join(chunks)
    compact = re.sub(r"\s+", " ", compact)
    return compact.strip()


def noisy_text(generator: random.Random, text: str) -> str:
    replacements = {
        "appartement": "appart",
        "chambre": "chambre",
        "chambres": "ch",
        "prix": "prx",
        "quartier": "qrt",
        "a vendre": "avendre",
        "à vendre": "avendre",
        "a louer": "alouer",
        "à louer": "alouer",
        "toilettes": "wc",
        "sécurisé": "securise",
        "sécurisée": "securise",
        "barrière": "barriere",
        "étudiant": "etudiant",
        "propriétaire": "proprietaire",
    }
    noisy = compact_text(text).lower()
    for source, target in replacements.items():
        if source in noisy and generator.random() < 0.8:
            noisy = noisy.replace(source, target)

    tokens = noisy.split()
    if len(tokens) > 12:
        head = tokens[:4]
        tail = tokens[4:]
        generator.shuffle(tail)
        tokens = head + tail

    drop_count = min(5, max(1, len(tokens) // 18))
    for _ in range(drop_count):
        if tokens and generator.random() < 0.6:
            tokens.pop(generator.randrange(len(tokens)))

    suffix = generator.choice(["dispo vite", "urgent", "negociable", "visite payante"])
    return f"{' '.join(tokens)} {suffix}".strip()


def normalize_tags(raw_tags: Any) -> list[str]:
    tags: list[str] = []
    if isinstance(raw_tags, str):
        try:
            parsed = json.loads(raw_tags.replace("'", '"'))
            raw_tags = parsed
        except Exception:
            raw_tags = [chunk.strip() for chunk in raw_tags.split(",")]
    if not isinstance(raw_tags, list):
        return tags
    for tag in raw_tags:
        if isinstance(tag, str) and tag in ALLOWED_TAGS and tag not in tags:
            tags.append(tag)
    return tags


def normalize_type(raw_type: Any) -> str | None:
    if not isinstance(raw_type, str):
        return None
    normalized = TYPE_ALIASES.get(raw_type, raw_type)
    return normalized if normalized in ALLOWED_TYPES else None


def normalize_number(value: Any, allow_zero: bool = False) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        digits = re.sub(r"[^\d-]", "", value)
        if not digits:
            return None
        value = int(digits)
    elif not isinstance(value, (int, float)):
        return None
    number = int(value)
    if number == 0 and not allow_zero:
        return None
    return number


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def normalize_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = clean_text(value)
    return normalized or None


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.replace("\r", " ").strip())


def compose_fallback_source_text(annonce: dict[str, Any]) -> str:
    chunks = [
        clean_text(annonce.get("title")),
        clean_text(annonce.get("description")),
    ]
    return " ".join(chunk for chunk in chunks if chunk).strip()


def extract_property_source_text(payload: dict[str, Any]) -> str:
    chunks = [
        clean_text(payload.get("sourceText")),
        clean_text(payload.get("description")),
        clean_text(payload.get("title")),
    ]
    text = " ".join(chunk for chunk in chunks if chunk).strip()
    return text


def extract_contact(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    normalized = text.replace("/", " ").replace("-", " ").replace(".", " ")
    match = re.search(r"(0\d{8})", normalized)
    if match:
        return match.group(1)
    return None


def offer_label(status: str) -> str:
    return "A LOUER" if status == "FOR_RENT" else "A VENDRE"


def type_label(type_property: str) -> str:
    labels = {
        "Home": "Maison",
        "Studio": "Studio",
        "Apartment": "Appartement",
        "Desk": "Bureau",
        "Building": "Immeuble",
        "Shop": "Boutique",
        "Kiosk": "Kiosque",
        "Room": "Chambre",
        "Property": "Propriété",
        "Logement": "Logement",
        "Villa": "Villa",
        "Land": "Terrain",
    }
    return labels.get(type_property, type_property)


def build_manifest(
    records: list[ExampleRecord],
    output_files: dict[str, str],
    total_examples: int,
) -> DatasetManifest:
    split_counts = Counter(record.split for record in records)
    difficulty_counts = Counter(record.difficulty for record in records)
    property_type_counts = Counter(record.target_json["typeProperty"] for record in records)
    source_counts = Counter(record.metadata["source"] for record in records)
    return DatasetManifest(
        target_size=total_examples,
        generated_examples=len(records),
        split_counts=dict(split_counts),
        difficulty_counts=dict(difficulty_counts),
        property_type_counts=dict(property_type_counts),
        source_counts=dict(source_counts),
        output_files=output_files,
    )


def write_jsonl(path: Path, records: list[ExampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def classify_external_post(text: str) -> str:
    normalized = compact_text(text).lower()
    if not normalized:
        return "irrelevant"
    if any(keyword in normalized for keyword in ("bonjour besoin", "je cherche", "besoin d'une", "recherche")):
        return "request"
    if any(pattern.search(normalized) for pattern in EXTERNAL_POST_PATTERNS):
        return "offer"
    return "uncertain"
