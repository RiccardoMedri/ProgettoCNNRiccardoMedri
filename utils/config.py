import json

#Carica il JSON di configurazione e normalizza la mappa classi in chiavi/valori interi
def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)

    class_map = config["data"].get("class_map", {})
    config["data"]["class_map"] = {int(k): int(v) for k, v in class_map.items()}
    return config