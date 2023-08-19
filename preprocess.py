from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from argparse import ArgumentParser
import json
import re
from multiprocessing import Pool
import time
import pandas as pd
import os
import shutil
from epitran import Epitran

import utils

import sys
sys.path.insert(0, "./converter")

from japanese_to_ipa import Japanese2IPA
from maltese_to_ipa import Maltese2IPA
from finnish_to_ipa import Finnish2IPA
from greek_to_ipa import Greek2IPA
from tamil_to_ipa import Tamil2IPA
from english_to_ipa import English2IPA

parser = ArgumentParser(description="Create dataset locally.")
parser.add_argument("-l", "--languages", nargs="+", default=["ja", "pl", "mt", "hu", "fi", "el", "ta", "en"],
                    help="Specify the languages to include in the test dataset.")
parser.add_argument("--output_dir", type=str, default="data_new",
                    help="Specify the output directory in which the preprocessed data will be stored.")
parser.add_argument("--num_proc", type=int, default=1,
                    help="Specify the number of cores to use for multiprocessing. The default is set to 1 (no multiprocessing).")
parser.add_argument("--clear_cache", action="store_true",
                    help="Use this option if you want to clear the dataset cache after loading to prevent memory from crashing.")
parser.add_argument("--cache_dir", type=str,
                    help="Specify the cache directory's path if you choose to clear the cache.")
args = parser.parse_args()
assert args.clear_cache and args.cache_dir is not None, "Cache directory's path is not defined."

def transliterate(sample: dict):
    if "chapter_id" in sample.column_names:
        lang = "en"
    else:
        lang = sample["locale"]
    sent = sample["sentence"]
    if lang == "ja":
        converter = Japanese2IPA()
        ipa = converter.remove_ja_punct(sent)
        ipa = converter.convert_sentence_to_ipa(ipa)
    elif lang == "mt":
        ipa = Maltese2IPA.maltese_generate_ipa(sent)
    elif lang == "fi":
        ipa = Finnish2IPA.finnish_generate_ipa(sent)
    elif lang == "el":
        ipa = Greek2IPA.greek_generate_ipa(sent)
    elif lang == "hu":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("hun-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "pl":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("pol-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "ta":
        ipa = Tamil2IPA.tamil_generate_ipa(sent)
    elif lang == "en":
        ipa = English2IPA.english_generate_ipa(sent)
    else:
        raise Exception("Unknown locale (language) found")
    sample["ipa"] = "".join(ipa.split())
    return sample

def remove_tamil_special_char(train, valid) -> tuple:
    """Remove sentences including "ச" since its pronunciation
    seems to be irregular/erratic
    """
    train = train.filter(lambda batch: "ச" not in batch["sentence"])
    valid = valid.filter(lambda batch: "ச" not in batch["sentence"])
    return train, valid

def remove_audio_column(train, valid) -> tuple:
    """Remove ["audio"] column from the dataset so that it can be
    saved to json.
    Apparently `array` causes `OverflowError: Unsupported UTF-8 sequence length when encoding string`
    so we need to remove it.
    This column will be restored by directly downloaded data upon training.
    """
    train = train.remove_columns(["audio"])
    valid = valid.remove_columns(["audio"])
    return train, valid
    
# Dataset
if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    stats_file = "{}/presave_trainvalid_stats.tsv".format(args.output_dir)
    with open(stats_file, "w") as f:
        f.write("lang\ttrain\tvalid\ttime\n")
        
    # test data split creation
    for l in args.languages:
        start = time.time()
        if l == "en":
            train = load_dataset("librispeech_asr",
                                 split="train.clean.100")
            valid = load_dataset("librispeech_asr",
                                 split="validation.clean")
        elif l == "ta":
            # Tamil dataset is too big and reaches AFS file path limit
            train = load_dataset("mozilla-foundation/common_voice_11_0", l,
                                 split="train",
                                 streaming=True)
            valid = load_dataset("mozilla-foundation/common_voice_11_0", l,
                                 split="validation",
                                 streaming=True)
            ds_train = []
            ds_valid = []
            for i, batch in enumerate(train):
                if i >= 30000:
                    break
                ds_train.append(batch)
            for i, batch in enumerate(valid):
                if i >= 30000:
                    break
                ds_valid.append(batch)
            train = Dataset.from_pandas(pd.DataFrame(data=ds_train))
            valid = Dataset.from_pandas(pd.DataFrame(data=ds_valid))
            
        else:
            train = load_dataset("mozilla-foundation/common_voice_11_0", l,
                                 split="train")
            valid = load_dataset("mozilla-foundation/common_voice_11_0", l,
                                 split="validation")

        if l == "ta":
            train, valid = remove_tamil_special_char(train, valid)

        # Remove audio column (non-writable to json)
        train, valid = remove_audio_column(train, valid)

        train = train.map(transliterate,
                          num_proc=args.num_proc)
        valid = valid.map(transliterate,
                          num_proc=args.num_proc)

        # Export to json
        train.to_json("{}/{}_train.json".format(args.output_dir, l))
        valid.to_json("{}/{}_valid.json".format(args.output_dir, l))

        print("{}\ttrain: {}\tvalid: {}\n".format(l, len(train), len(valid)))
        end = time.time()
        duration = end - start
        print("Elapsed time for {}: {}".format(l, duration))
        with open(stats_file, "a") as f:
            f.write("{}\t{}\t{}\t{}\n".format(l, len(train), len(valid), duration))

        # Clear cache
        print("Clearing the cache...")
        if args.clear_cache:
            shutil.rmtree(args.cache_dir, ignore_errors=True)
        print("Cache cleared")
