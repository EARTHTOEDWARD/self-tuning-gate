import json
import io
import os

class TelemetryWriter:
    def __init__(self, path: str, fmt: str = "jsonl"):
        self.path = path
        self.fmt = fmt
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = io.open(self.path, "w", encoding="utf8")

    def write(self, row: dict):
        if self.fmt == "jsonl":
            self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            if not hasattr(self, "_csv_header_written"):
                self._csv_keys = list(row.keys())
                self.f.write(",".join(self._csv_keys) + "\n")
                self._csv_header_written = True
            vals = [str(row.get(k, "")) for k in self._csv_keys]
            self.f.write(",".join(vals) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
