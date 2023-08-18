from datasets import load_dataset, Dataset
from japanese_to_ipa import Japanese2IPA
from maltese_to_ipa import Maltese2IPA
from finnish_to_ipa import Finnish2IPA
from greek_to_ipa import Greek2IPA
import random
import epitran
import re

def selection(dataset, selectsize: int):
    trainsize = len(dataset)
    samples = sampling(trainsize, selectsize)
    selected = dataset.select(samples)
    return selected

def filter_low_quality(dataset):
    dataset = dataset.filter(lambda batch: batch["down_votes"] == 0)
    return dataset

def remove_alpha_punct(batch: dict) -> dict:
    """
    Remove punctuation symbols from batch["sentence"],
    which contains the sentence (transcription) of the
    speech data.
    This function can only be used for alphabetical orthographies.
    """
    non_punct = "[\s\w]"
    sent = re.findall(non_punct, batch["sentence"].lower(), re.MULTILINE)
    batch["sentence"] = "".join(sent)
    return batch

def convert_maltese_to_ipa(batch: dict) -> dict:
    """
    Convert Maltese sentences to IPA.
    Note that `maltese_generate_ipa` also handles punctuation removal.
    """
    sent = batch["sentence"]
    batch["ipa"] = Maltese2IPA.maltese_generate_ipa(sent)
    return batch

def convert_polish_to_ipa(batch: dict) -> dict:
    """                                                                                                                                                                                                        Convert Polish sentences to IPA.                                                                                                                                                                           """
    epi = epitran.Epitran('pol-Latn')
    batch["ipa"] = epi.transliterate(batch["sentence"])
    return batch

def convert_hungarian_to_ipa(batch: dict) -> dict:
    """
    Convert Hungarian sentences to IPA.
    """
    epi = epitran.Epitran("hun-Latn")
    batch["ipa"] = epi.transliterate(batch["sentence"])
    return batch

def convert_finnish_to_ipa(batch: dict) -> dict:
    """
    Convert Finnish sentences to IPA.
    Note that `finnish_generate_ipa` also handles punctuation removal.
    """
    sent = batch["sentence"]
    batch["ipa"] = Finnish2IPA.finnish_generate_ipa(sent)
    return batch

def convert_greek_to_ipa(batch: dict) -> dict:
    """
    Convert Greek sentences to IPA.
    Note that `greek_generate_ipa` also ahndles punctualtion removal.
    """
    sent = batch["sentence"]
    batch["ipa"] = Greek2IPA.greek_generate_ipa(sent)
    return batch

def convert_tamil_to_ipa(batch: dict) -> dict:
    """
    Convert Tamil sentences to IPA.
    """
    epi = epitran.Epitran("tam-Taml")
    sent = epi.transliterate(batch["sentence"])
    voiceable = {"k": "g",
                 "q": "d͡ʑ",
                 "x": "d̪",
                 "p": "b",
                 "ʈ": "ɖ", 
                 "t": "d"}

    convertable = {"ŋk": "ŋg",
                   "n̪x": "n̪d̪",
                   "ɲq": "ɲd͡ʑ",
                   "ɳʈ": "ɳɖ",
                   "rr": "tːr",
                   "pp": "pː",
                   "kk": "kː",
                   "xx": "t̪ː",
                   "ʈʈ": "ʈː",
                   "qq": "t͡ɕː",
                   "nr": "ndr",
                   "ɯː": "uː",}
    
    sent = sent.replace("t͡ʃ", "q")
    sent = sent.replace("t̪", "x")
    
    for k, v in convertable.items():
        sent = sent.replace(k, v)

    sonorants = ["a", "ɯ", "i", "e", "o", "j", "ɾ" "ː"]
    newsent = list(sent)
    for i, c in enumerate(sent):
        if i >= 1 and i < len(sent) - 1:
            if sent[i-1] in sonorants and sent[i+1] in sonorants and sent[i] in voiceable.keys():
                newsent[i] = voiceable[c]
    sent = "".join(newsent)
    sent = sent.replace("q", "t͡ɕ")
    sent = sent.replace("x", "t̪")

    tokens = sent.split()
    for i, t in enumerate(tokens):
        if t.startswith("e"):
            tokens[i] = "j" + t
    sent = " ".join(tokens)

    # remove punctuation for IPA 
    non_punct = r"[\s\w\u0250-\u02AF\u02B0-\u02FF\u1D00-\u1D7F\u1D80–\u1DBF\u0300-\u036F]"
    sent = re.findall(non_punct, sent, re.MULTILINE)
    sent = "".join(sent)

    batch["ipa"] = sent
    return batch

def sampling(size: int, n: int) -> list:
    """
    size: The length (number of samples) in the original dataset
    n: The length of the samples you want to get
    The output is a list of integers (indices of the dataset) without duplication.
    Use this function to reduce the sample size of the dataset when training with
    multiple languages and expected to be time-consuming.
    """
    numlist = [i for i in range(size)]
    random_samples = random.sample(numlist, n)
    return random_samples

def selection(dataset, selectsize: int):
    trainsize = len(dataset)
    samples = sampling(trainsize, selectsize)
    selected = dataset.select(samples)
    return selected

def downsampling(dataset: Dataset, samples: int):
    size = len(dataset)
    if size == None or size < samples:
        samples = size
    dataset = selection(dataset, samples)
    return dataset

class Preprocessors:
    @classmethod
    def japanese(cls, train_samples, test_samples, quality_filter=False):
        # should the first variable be `cls`?
        # load dataset
        ja_train = load_dataset("common_voice", "ja", split="train")
        ja_test = load_dataset("common_voice", "ja", split="validation")

        # Filter low quality samples
        if quality_filter:
            ja_train = filter_low_quality(ja_train)
            ja_test = filter_low_quality(ja_test)

        # downsampling if necessary
        # train_size = len(ja_train)
        # test_size = len(ja_test)
        # if train_size == None or train_size < train_samples:
        #     # not specified on the command line or too big value
        #     train_samples = train_size
        # if test_size == None or test_size < test_samples:
        #     # not specified on the command line or too big value
        #     test_samples = test_size # maximum
        # ja_train = selection(ja_train, train_samples)
        # ja_test = selection(ja_test, test_samples)

        # to IPA
        ja_train = ja_train.map(Japanese2IPA.convert)
        ja_test = ja_test.map(Japanese2IPA.convert)

        return ja_train, ja_test

    @classmethod
    def polish(cls, train_samples, test_samples, quality_filter=False):
        # load dataset
        pl_train = load_dataset("common_voice", "pl", split="train")
        pl_test = load_dataset("common_voice", "pl", split="validation")

        # Filter low-quality samples
        if quality_filter:
            pl_train = filter_low_quality(pl_train)
            pl_test = filter_low_quality(pl_test)

        # Downsampling if necessary
        # train_size = len(pl_train)
        # test_size = len(pl_test)
        # if train_size < train_samples:
        #     train_samples = train_size # maximu
        #     print("Train samples are larger than the available data samples.\nWe use the maximum size instead.")
        # if test_size < test_samples:
        #     test_samples = test_size # maximum  
        #     print("Test samples are larger than the available data samples.\nWe use the maximum size instead.")
        # pl_train = selection(pl_train, train_samples)
        # pl_test = selection(pl_test, test_samples)

        # Remove punctuation
        pl_train = pl_train.map(remove_alpha_punct)
        pl_test = pl_test.map(remove_alpha_punct)

        # to IPA
        pl_train = pl_train.map(convert_polish_to_ipa)
        pl_test = pl_test.map(convert_polish_to_ipa)

        return pl_train, pl_test

    @classmethod
    def maltese(cls, train_samples, test_samples, quality_filter=False):
        # Load dataset
        mt_train = load_dataset("common_voice", "mt", split="train")
        mt_test = load_dataset("common_voice", "mt", split="validation")

        # Filter low-quality audio                                                                                                                                                                         
        if quality_filter:
            mt_train = filter_low_quality(mt_train)
            mt_test = filter_low_quality(mt_test)

        # Downsampling if necessary
        # train_size = len(mt_train)
        # test_size = len(mt_test)
        # if train_size < train_samples:
        #     train_samples = train_size # maximum
        #     print("Train samples are larger than the available data samples.\nWe use the maximum size instead.")
        # if test_size < test_samples:
        #     test_samples = test_size # maximum
        #     print("Test samples are larger than the available data samples.\nWe use the maximum size instead.")
        # mt_train = selection(mt_train, train_samples)
        # mt_test = selection(mt_test, test_samples)

        # Remove punctuation and convert to IPA
        mt_train = mt_train.map(convert_maltese_to_ipa)
        mt_test = mt_test.map(convert_maltese_to_ipa)

        return mt_train, mt_test

    @classmethod
    def hungarian(cls, train_samples, test_samples, quality_filter=False):
        # Load dataset
        hu_train = load_dataset("common_voice", "hu", split="train")
        hu_test = load_dataset("common_voice", "hu", split="validation")

        # Filter low-quality audio
        if quality_filter:
            hu_train = filter_low_quality(hu_train)
            hu_test = filter_low_quality(hu_test)

        # Downsampling if necessary
        # train_size = len(hu_train)
        # test_size = len(hu_test)
        # if train_size < train_samples:
        #     train_samples = train_size # maximum
        #     print("Train samples are larger than the available data samples.\nWe use the maximum size instead.")
        # if test_size < test_samples:
        #     test_samples = test_size # maximum
        #     print("Test samples are larger than the available data samples.\nWe use the maximum size instead.")
        # hu_train = selection(hu_train, train_samples)
        # hu_test = selection(hu_test, test_samples)

        # Remove punctuation
        hu_train = hu_train.map(remove_alpha_punct)
        hu_test = hu_test.map(remove_alpha_punct)

        # to IPA
        hu_train = hu_train.map(convert_hungarian_to_ipa)
        hu_test = hu_test.map(convert_hungarian_to_ipa)

        return hu_train, hu_test

    @classmethod
    def finnish(cls, train_samples, test_samples, quality_filter=False):
        # Load dataset
        train = load_dataset("common_voice", "fi", split="train")
        test = load_dataset("common_voice", "fi", split="validation")

        # Filter low-quality audio
        if quality_filter:
            train = filter_low_quality(train)
            test = filter_low_quality(test)

        # Downsampling if necessary
        # train_size = len(train)
        # test_size = len(test)
        # if train_size < train_samples:
        #     train_samples = train_size
        #     print("Train samples are larger than the available data samples.\nWe use the maximum size instead.")
        # if test_size < test_samples:
        #     test_samples = test_size
        #     print("Test samples are larger than the available data samples.\nWe use the maximum size instead.")
        # train = selection(train, train_samples)
        # test = selection(test, test_samples)

        # punctuation removal and conversion to IPA 
        train = train.map(convert_finnish_to_ipa)
        test = test.map(convert_finnish_to_ipa)

        return train, test

    @classmethod
    def greek(cls, train_samples, test_samples, quality_filter=False):
        # Load dataset
        train = load_dataset("common_voice", "el", split="train")
        test = load_dataset("common_voice", "el", split="validation")

        # Filter low-quality audio
        if quality_filter:
            train = filter_low_quality(train)
            test = filter_low_quality(test)

        # Downsampling if necessary
        # train_size = len(train)
        # test_size = len(test)
        # if train_size < train_samples:
        #     train_samples = train_size
        #     print("Train samples are larger than the available data samples.\nWe use the maximum size instead.")
        # if test_size < test_samples:
        #     test_samples = test_size
        #     print("Test samples are larger than the available data samples.\nWe use the maximum size instead.")
        # train = selection(train, train_samples)
        # test = selection(test, test_samples)

        # punctuation removal and conversion to IPA                                                                                                                                                        
        train = train.map(convert_greek_to_ipa)
        test = test.map(convert_greek_to_ipa)

        return train, test

    @classmethod
    def tamil(cls, train_samples, test_samples, quality_filter=False):
        # Load dataset
        train = load_dataset("common_voice", "ta", split="train")
        test = load_dataset("common_voice", "ta", split="validation")

        # Filter low-quality audio
        if quality_filter:
            train = filter_low_quality(train)
            test = filter_low_quality(test)

        # # Downsampling if necessary
        # train_size = len(train)
        # test_size = len(test)
        # if train_size < train_samples:
        #     train_samples = train_size # maximum
        #     print("Train samples are larger than the available data samples.\nWe use the maximum size instead.")
        # if test_size < test_samples:
        #     test_samples = test_size # maximum
        #     print("Test samples are larger than the available data samples.\nWe use the maximum size instead.")
        # train = selection(train, train_samples)
        # test = selection(test, test_samples)

        # to IPA
        print("Converting to IPA...")
        train = train.map(convert_tamil_to_ipa)
        test = test.map(convert_tamil_to_ipa)
        print("IPA conversion done")

        print(type(train["ipa"]))

        # Remove punctuation
        # train = train.map(remove_tamil_punct)
        # test = test.map(remove_tamil_punct)

        # Filter IPA (don't include samples containing "t͡ɕ" and "d͡ʑ";
        # they are subject to allophonic variations)
        train = train.filter(lambda batch: "t͡ɕ" not in batch["ipa"] and "d͡ʑ" not in batch["ipa"])
        test = test.filter(lambda batch: "t͡ɕ" not in batch["ipa"] and "d͡ʑ" not in batch["ipa"])

        return train, test
