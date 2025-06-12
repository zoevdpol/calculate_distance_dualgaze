import json
import pathlib

class CustomTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, pathlib.Path):
            return str(obj)
        # Voeg hier andere custom types toe indien nodig
        return super().default(obj)
