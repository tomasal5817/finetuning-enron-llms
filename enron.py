from typing import Tuple
import datasets
import pandas as pd
from datasets import load_dataset
from dataclasses import dataclass
import json
import random

def print_highlighted(text):
    print("\033[93m" + text + "\033[0m")

def rnd_idx(N, seed=None):
    """
    Return a list of shuffled indices from 0 to N-1.
    """
    idx = list(range(N))
    if seed is not None:
        random.seed(seed)
    random.shuffle(idx)
    return idx

@dataclass
class CustomEnronBuilder(datasets.BuilderConfig):
    name: str = None
    sample_duplication_rate: int = 1    # number of times a sample is repeated
    shuffle_facts_seed: int = 42
    chat_template: str = "none"
    pseudonymize: bool = False

class CustomEnron(datasets.GeneratorBasedBuilder):
    """ A wrapper around the Enron dataset without anonymization. """

    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the Enron dataset (no scrubbing)."
    _TEXT = "text"

    BUILDER_CONFIGS = [
        CustomEnronBuilder(name="default", sample_duplication_rate=1, version=VERSION, description="Raw Enron data"),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def __init__(self, *args, **kwargs):
        self.df: pd.DataFrame = pd.DataFrame()
        super().__init__(*args, **kwargs)

    def _info(self):
        features = datasets.Features({self._TEXT: datasets.Value("string")})
        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=features
        )

    def _split_generators(self, dl_manager):
        self.df = load_dataset("LLM-PBE/enron-email")
        print("done load data")
        print_highlighted(f"self.config.pseudonymize: {self.config.pseudonymize}")
        self.data = [item for item in self.df["train"]["text"]]
        
        all_texts = (
            self.df["train"]["text"]
            + self.df.get("test", {"text": []})["text"]
            + self.df.get("validation", {"text": []})["text"]
        )
        
        # Shuffle
        if self.config.shuffle_facts_seed > 0:
            all_texts = [all_texts[i] for i in rnd_idx(N=len(all_texts), seed=self.config.shuffle_facts_seed)]

        self.data = all_texts

        return [
            datasets.SplitGenerator(  
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "start": 0.0,
                    "end": 0.45  
                },
            ),
            datasets.SplitGenerator(  
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "start": 0.45,
                    "end": 0.55
                },
            ),
            datasets.SplitGenerator( 
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "start": 0.55,
                    "end": 1.0  
                },
            ),
        ]
    def _generate_examples(self, split: str, start: float, end: float):
        start_pos, end_pos = int(len(self.data) * start), int(len(self.data) * end)
        print_highlighted(
            f"Length of data: {len(self.data)}. Loading from {start_pos} to {end_pos} (Total={end_pos - start_pos})."
        )

        unique_identifier = start_pos
        for i, text in enumerate(self.data[start_pos:end_pos]):
            for _ in range(self.config.sample_duplication_rate):
                yield f"{unique_identifier}", {self._TEXT: text}
                unique_identifier += 1

 