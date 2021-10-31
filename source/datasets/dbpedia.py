import numpy as np
import random
from imblearn.under_sampling import RandomUnderSampler
from typing import Any, Dict, List, Tuple


class DBpedia:

    def __init__(self, file_path: str):
        self.file_path_ = file_path
        self._load_text()
        self._extract_content()
        self._assign_labels()
        self._select_data()

    def _load_text(self) -> None:
        with open(self.file_path_, "r", encoding="utf-8") as text_file:
            self.text_ = text_file.read()

    def _extract_content(self) -> None:

        KEYS = [
            "SENTENCE",
            "MANUALLY CHECKED",
            "ENTITY1",
            "TYPE1",
            "ENTITY2",
            "TYPE2",
            "REL TYPE"
        ]

        self.raw_samples_ = []
        sample: Dict[str, Any] = {}
        for line in self.text_.split("\n"):

            if line.find(":") > 0:  # content
                key = KEYS[len(sample)]
                value = line.split(":")[-1][1:]
                sample[key] = value

            if line.find("****") == 0:  # break
                if sample:
                    self.raw_samples_.append(sample)
                    sample = {}

    def _assign_labels(self) -> None:
        for sample in self.raw_samples_:
            sample["LABEL"] = 0 if sample["REL TYPE"] == "other" else 1

    def _select_data(self) -> None:

        samples_filtered = self._filter_inconsistent_samples(self.raw_samples_)
        samples_resampled = self._resample(samples_filtered)

        self.data = []
        for sample in samples_resampled:
            self.data.append({
                "sentence": sample["SENTENCE"],
                "entity_1": sample["ENTITY1"].strip(),
                "entity_2": sample["ENTITY2"].strip()
            })

        self.labels = [sample["LABEL"] for sample in samples_resampled]

    @staticmethod
    def _filter_inconsistent_samples(samples: List[dict]) -> List[dict]:
        samples_filtered = []
        for sample in samples:
            if (
                (sample["ENTITY1"] in sample["SENTENCE"]) and
                (sample["ENTITY2"] in sample["SENTENCE"]) and
                (sample["ENTITY1"] != sample["ENTITY2"])
            ):
                samples_filtered.append(sample)
        return samples_filtered

    @staticmethod
    def _resample(samples: List[dict]) -> List[dict]:
        labels = np.array([sample["LABEL"] for sample in samples]).reshape(-1, 1)
        indexes = np.array([i for i, _ in enumerate(labels)]).reshape(-1, 1)
        selected_indexes, _ = RandomUnderSampler(random_state=42).fit_resample(indexes, labels)
        selected_indexes = sorted(selected_indexes[:, 0])
        return [samples[i] for i in selected_indexes]

    @staticmethod
    def _get_subsample(samples: List[dict], n_samples: int) -> List[dict]:
        random.seed(42)
        return random.sample(samples, n_samples)
