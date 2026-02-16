import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from pprint import pformat


class StoreParams:
    def __init__(self):
        self._params = {}
        self._extra_files = {}

    def add(self, key, value):
        """Store any Python object under a key."""
        self._params[key] = value

    def add_classification_report(self, report_str=None, report_dict=None):
        """
        Store classification report (string and/or dict).
        """
        if report_str is not None:
            self._extra_files["classification_report.txt"] = report_str

        if report_dict is not None:
            self._extra_files["classification_report.json"] = report_dict

    def save_params(self, base_dir="saved_params"):
        """
        Creates a unique directory and dumps:
        - params.json
        - any extra files (like classification report)
        Returns the directory path.
        """
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        save_path = base_path / unique_id
        save_path.mkdir()

        # Save main params
        params_file = save_path / "params.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(self._params, f, indent=4, default=str)

        # Save extra files
        for filename, content in self._extra_files.items():
            file_path = save_path / filename

            if filename.endswith(".json"):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=4)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(content))

        return str(save_path)
