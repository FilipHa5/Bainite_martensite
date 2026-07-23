import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from pprint import pformat


class StoreParams:
    def __init__(self, auto_save=True, base_dir="saved_params"):
        self._path = ""
        self._params = {}
        self._extra_files = {}
        self._auto_save = auto_save
        self._base_dir = base_dir

    def get_dir(self):
        return self._base_dir

    def add(self, key, value):
        self._params[key] = value
        if self._auto_save:
            self.save_params(self._base_dir)

    def add_classification_report(self, report_str=None, report_dict=None):
        if report_str is not None:
            self._extra_files["classification_report.txt"] = report_str

        if report_dict is not None:
            self._extra_files["classification_report.json"] = report_dict

        if self._auto_save:
            self.save_params(self._base_dir)

    def save_params(self, base_dir="saved_params"):
        """
        Creates a unique directory on the first call and saves:
        - params.json
        - any extra files (like classification report)

        Subsequent calls reuse the same directory.
        Returns the directory path.
        """
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Reuse existing path if available
        if self._path:
            save_path = Path(self._path)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            save_path = base_path / unique_id
            save_path.mkdir(parents=True, exist_ok=True)
            self._path = str(save_path)

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
