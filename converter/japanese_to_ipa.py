# Learning IPA from Japanese and Polish                                                                                                                                                                    
from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random

# For Japanese processing
import MeCab
import unidic
import romkan
mecab = MeCab.Tagger()

import re

class Japanese2IPA():
    IGNORE_JA_REGEX = "[、,。]"
    NON_PUNCT = "\s\w"
    roman_to_ipa = {# consonants
        "pp": "pː",
        "tt": "tː",
        "dd": "dː",
        "kk": "kː",
        "r": "ɾ",
        "y": "ɰ", # temporary conversion for avoiding confusion                                                                                                                                     
        # "hi": "çi",                                                                                                                                                                               
        "hy": "ç",
        "sh": "ɕ",
        "ssh": "ɕː",
        "j": "d͡ʑ",
        "ts": "t͡s",
        "ch": "t͡ɕ",
        "cch": "t͡ɕː",
        "n'n": "nː",
        "ni": "ɲi",
        "nni": "ɲːi",
        "ny": "ɲ",
        "n'ny": "ɲː",
        "ng": "ŋg",
        "nk": "ŋk",
        "nm": "mː",
        "f": "ɸ",
        # vowels
        "u": "ɯ",
        "a": "ä",
        "e": "e̞",
        "o": "o̞",
        # long vowel
        "aa": "äː",
        "ii": "iː",
        "uu": "ɯː",
        "ee": "e̞ː",
        "ou": "o̞ː",
        "oo": "o̞ː",
        "wo": "o̞",
        "-": "ː"
    }

    def remove_ja_punct(self, sent: str) -> str:
        """
        Remove Japanese punctuation symbols.
        """
        sent = re.sub(self.IGNORE_JA_REGEX, "", sent).lower() + " "
        return sent

    def convert_sentence_to_ipa(self, sent: str) -> str:
        s = mecab.parse(sent)
        kana = ""
        for line in s.split("\n"):
            if line.find("\t") <= 0:
                kana = kana.rstrip(" ")
                continue
            columns = line.split(",")
            if len(columns) < 10:
                kana += line.split("\t")[0]
                kana += " "
            else:
                pos = columns[0].split("\t")[1]
                if pos == "助動詞" or pos == "助詞":
                    kana = kana.rstrip(" ")
                kana += columns[9]
                kana += " "
        roman = romkan.to_roma(kana)
        # from longest                                                                                                                                                                                    
        four = dict([(k, v) for k, v in self.roman_to_ipa.items() if len(k) == 4])
        three = dict([(k, v) for k, v in self.roman_to_ipa.items() if len(k) == 3])
        two = dict([(k, v) for k, v in self.roman_to_ipa.items() if len(k) == 2])
        one = dict([(k, v) for k, v in self.roman_to_ipa.items() if len(k) == 1])
        trans = [four, three, two, one]
        ipa = roman
        for t in trans:
            for k, v in t.items():
                ipa = ipa.replace(k, v)
        ipa = ipa.replace("ɰ", "j")
        ipa = ipa.replace("hi", "çi")
        ipa = ipa.replace("nw", "ɴw")
        tokens = ipa.split()
        for i, t in enumerate(tokens):
            if t[-1] == "n":
                tokens[i] = t[:-1] + "ɴ"
        ipa = " ".join(tokens)
        return ipa

    @classmethod
    def convert(self, batch: dict) -> dict:
        sent = batch["sentence"]
        sent = self.remove_ja_punct(self, sent)
        ipa = self.convert_sentence_to_ipa(self, sent)
        batch["ipa"] = ipa
        return batch
