import json
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError as exc:
    Draft7Validator = None
    _jsonschema_import_error = exc

#Carica il JSON di configurazione e normalizza la mappa classi in chiavi/valori interi
def _validate_config(config, schema_path):
    if Draft7Validator is None:
        raise SystemExit(
            "jsonschema non e' installato. Installa con: pip install jsonschema"
        ) from _jsonschema_import_error
    with open(schema_path, "r") as f:
        schema = json.load(f)

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
    if errors:
        parts = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) or "<root>"
            parts.append(f"{path}: {err.message}")
        msg = "Config non valida:\n" + "\n".join(parts)
        raise SystemExit(msg)


#Carica il JSON di configurazione e normalizza la mappa classi in chiavi/valori interi
def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)

    root_dir = Path(__file__).resolve().parent.parent
    schema_path = root_dir / "config" / "config_schema.json"
    if not schema_path.exists():
        raise SystemExit(f"Schema non trovato: {schema_path}")
    _validate_config(config, schema_path)

    class_map = config["data"].get("class_map", {})
    config["data"]["class_map"] = {int(k): int(v) for k, v in class_map.items()}
    return config