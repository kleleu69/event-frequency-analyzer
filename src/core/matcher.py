"""Matching modes for event words and phrases."""

import re
from typing import List

import pandas as pd
from fuzzywuzzy import fuzz


class WordMatcher:
    def __init__(self, words: List[str], mode: str = "partial",
                 threshold: int = 80, case_sensitive: bool = False):
        if mode not in ("exact", "partial", "fuzzy"):
            raise ValueError("Mode must be exact, partial, or fuzzy.")
        if not 0 <= threshold <= 100:
            raise ValueError("Fuzzy threshold must be between 0 and 100.")
        self.mode = mode
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.words = []
        seen = set()
        for word in words:
            word = str(word).strip()
            key = word if case_sensitive else word.casefold()
            if word and key not in seen:
                self.words.append(word)
                seen.add(key)

    def match(self, text) -> List[str]:
        if text is None or pd.isna(text):
            return []
        text = str(text)
        if not self.case_sensitive:
            text = text.casefold()
        matched = []
        for original in self.words:
            word = original if self.case_sensitive else original.casefold()
            if self.mode == "exact":
                found = re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None
            elif self.mode == "partial":
                found = word in text
            else:
                found = fuzz.partial_ratio(word, text) >= self.threshold
            if found:
                matched.append(original)
        return matched
