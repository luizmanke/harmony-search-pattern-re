import spacy
from tqdm import tqdm
from typing import List, Tuple


class Tagger:

    def __init__(self) -> None:
        self.spacy_model_ = spacy.load("pt_core_news_sm")
        self.TOKEN_MASK = "TOKEN_MASK"

    def tag(self, samples: List[dict]) -> List[dict]:
        tags = []
        for sample in tqdm(samples, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            try:
                sentence_masked = self._get_sentence_with_mask(sample)
                doc = self._get_doc(sentence_masked)
                start_index, end_index = self._get_boundary(doc)
                tags_list = self._get_tags(doc)
                tags_selected = self._select_tags(tags_list, start_index, end_index)
            except Exception:
                tags_selected = {}
            tags.append(tags_selected)
        return tags

    def convert_tags_to_strings(self, tags: dict) -> dict:
        strings = {}
        for key, values in tags.items():
            strings[key] = [self.spacy_model_.vocab.strings[value] for value in values]
        return strings

    def _get_sentence_with_mask(self, sample: dict) -> str:
        sentence = sample["sentence"]
        sentence = sentence.replace(sample["entity_1"], self.TOKEN_MASK, 1)
        sentence = sentence.replace(sample["entity_2"], self.TOKEN_MASK, 1)
        return sentence

    def _get_doc(self, sentence: str) -> spacy.tokens.doc.Doc:
        return self.spacy_model_(sentence)

    def _get_boundary(self, doc: spacy.tokens.doc.Doc) -> Tuple[int, int]:
        mask_count = 0
        for i, token in enumerate(doc):
            if token.text == self.TOKEN_MASK:
                if mask_count == 0:
                    start_index = i
                    mask_count += 1
                elif mask_count == 1:
                    end_index = i
                    break
        return start_index, end_index

    @staticmethod
    def _get_tags(doc: spacy.tokens.doc.Doc) -> List[int]:
        tags = [token.pos for token in doc]
        return tags

    @staticmethod
    def _select_tags(tags: List[int], start_index: int, end_index: int) -> dict:
        N_TAGS_BEFORE, N_TAGS_AFTER = 2, 2
        return {
            "between": tags[start_index+1:end_index]
        }
