from datasets import load_dataset, load_metric, Audio, concatenate_datasets, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Union
import random
import argparse
import pandas as pd
import os
import multiprocess

# local
from data_utils import filter_low_quality, downsampling
import add_forvo

IPA_LIST = "data/full_vocab_ipa.txt"

def extract_all_chars_ipa(batch: dict) -> dict:
    """Extract IPA characters from the train/valid dataset.

    TODO: Change this function so that it creates vocabulary
    based on phonemes, not on characters.
    (though I think this is taken care of by adding all the
    possible IPAs in the tokenizer later.)
    """
    all_text = " ".join(batch["ipa"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset_ipa(batch: dict) -> dict:
    """Prepare the dataset for the training by adding the input values
    and the labels.
    """
    audio = batch["audio"]
    batch["input_values"] = processor_ipa(audio["array"],
                                          sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor_ipa.as_target_processor():
        batch["labels"] = processor_ipa(batch["ipa"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> \
                 Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
                )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

    
def remove_long_data(dataset: Dataset,
                     max_seconds: Union[int, float] = 6) -> Dataset:
    """Remove samples that are too long.
    You can customize the max audio length depending on your computing environment.
    """
    # convert pyarrow table to pandas
    dftest = dataset.to_pandas()
    # find out length of input_values
    dftest["len"] = dftest['input_values'].apply(len)
    # for wav2vec training we already resampled to 16khz
    # remove data that is longer than max_seconds (6 seconds ideal)
    max_length = max_seconds * 16000 
    dftest = dftest[dftest["len"] < max_length]
    dftest = dftest.drop("len", 1)
    # convert back to pyarrow table to use in trainer
    dataset = dataset.from_pandas(dftest)
    # directly remove do not wait for gc
    del dftest
    return dataset


def concatenate_common_voice(datasetlist: List[Dataset]) -> Dataset:
    """
    Concatenate more than one datasets from Common Voice.
    Also consider using datasets.interleave_datasets(datasets: List[DatasetType]
    so that the new dataset is constructed by cycling between each source to get the examples.
    """
    init_data = datasetlist[0]
    for dl in datasetlist:
        assert dl.features.type == init_data.features.type
    concatenated = concatenate_datasets(datasetlist)
    return concatenated


def remove_space(batch: dict) -> dict:
    """Remove whitespace from the IPAs in the dataset."""
    ipa = batch["ipa"]
    ipa = ipa.split()
    ipa = "".join(ipa)
    batch["ipa"] = ipa
    return batch


def dataload_test(train_data, train_ipa, valid_data, valid_ipa):
    assert len(train_data) == len(train_ipa), \
        print("Length of train_data and train_ipa does not match")
    assert len(valid_data) == len(valid_ipa), \
        print("Length of valid_data and valid_ipa does not match")
    if l == "en":
        for j in range(len(train_data)):
            filename = train_data[j]["file"]
            ipa_filename = train_ipa[j]["file"]
            assert filename == ipa_filename
        for j in range(len(valid_ipa)):
            filename = valid_data[j]["file"]
            ipa_filename = valid_ipa[j]["file"]
            assert filename == ipa_filename
    else:
        for j in range(len(train_data)):
            filename = train_data[j]["path"].split("/")[-1]
            ipa_filename = train_ipa[j]["path"].split("/")[-1]
            assert filename == ipa_filename
        for j in range(len(valid_data)):
            filename = valid_data[j]["path"].split("/")[-1]
            ipa_filename = valid_ipa[j]["path"].split("/")[-1]
            assert filename == ipa_filename


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Specify languages to use and options for each language"
    )

    parser.add_argument("-l", "--languages", nargs="*", type=str, required=True,
                        help="Specify language code (split by space).",
                        "Typically ISO639-1, or ISO639-2 if not found in ISO639-1.")
    parser.add_argument("-tr", "--train_samples", nargs="*", type=int,
                        help="Specify the number of samples to be used as the training data for each language." \
                        "For example, if you want to use 1000, 2000, 3000 training samples for Japanese, Polish," \
                        "and Maltese, then specify as -l ja pl mt -tr 1000 2000 3000." \
                        "You can type an irrationally large number to pick up the maximum value.")
    parser.add_argument("-te", "--test_samples", nargs="*", type=int,
                        help="Specify the number of samples to be used as the test data for each language." \
                        "For example, if you want to use 1000, 2000, 3000 test samples for Japanese, Polish," \
                        "and Maltese, then specify as -l ja pl mt -tr 1000 2000 3000." \
                        "You can type an irrationally large number to pick up the maximum value.")
    parser.add_argument("-qf", "--quality_filter", nargs="*", type=bool, default=True,
                        help="Specify if you want to remove low quality audio (at least having 1 down vote) from the dataset." \
                        "True if you want to, False if you do not want to.")
    parser.add_argument("-a", "--additional_data", nargs=1, type=bool, default=False,
                        help="Specify if you want to use additional data fetched from Forvo.")
    parser.add_argument("-s", "--suffix", type=str, default="",
                        help="Specify a suffix to identify your training. This suffix will be added to the checkpoint file directory.")
    parser.add_argument("-ns", "--no_space", type=bool, default=False,
                        help="Set True if you want to remove spaces from the training and test data.")
    parser.add_argument("-v", "--vocab_file", type=str,
                        help="Specify the vocab file name to be created")
    parser.add_argument("-dd", "--data_dir", type=str, default="data_new/",
                        help="Specify the directory path for the training/validation data files." \
                        "Default is set to `data_new/`, which stores the data from the as-of-now newest" \
                        "`mozilla-foundation/common_voice_11_0`.")
    parser.add_argument("-ds", "--dataset", type=str, default="mozilla-foundation/common_voice_11_0",
                        help="Specify the dataset name. Default is set to" \
                        "`mozilla-foundation/common_voice_11_0`.")
    parser.add_argument("-e", "--num_train_epochs", type=int, default=30,
                        help="Specify the number of train epochs. By default it's set to 30.")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Specify the number of CPUs for preprocessing. Default set to 24.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size. Default set to 4.")
    parser.add_argument("--learning_date", type=float, default=3e-4,
                        help="Learning rate. Default set to 3e-4.")
    args = parser.parse_args()

    assert len(args.train_samples) <= len(args.languages), "`train_samples` argument is longer than the number of languages"
    assert len(args.test_samples) <= len(args.languages), "`test_samples` argument is longer than the number of languages"
    assert len(args.quality_filter) <= len(args.languages), "`quality_filter` argument is longer than the number of languages"

    if args.additional_data:
        from add_forvo import add_language
    return args


def load_data() -> Tuple[Dataset, Dataset]:
    """Load the training and validation data."""
    train_list = []
    valid_list = []
    stats_file = "stats_train_valid_{}.csv".format(args.suffix)
    with open(stats_file, "w") as f:
        f.write("lang,train,valid\n")
    for i, l in enumerate(args.languages):
        train_sample = args.train_samples[i]
        test_sample = args.test_samples[i]
        q_filter = args.quality_filter[i]
        if l == "en":
            q_filter = False

        train_ipa = load_dataset("json",
                                 data_files="{}{}_train.json".format(args.data_dir, l),
                                 split="train")
        valid_ipa = load_dataset("json",
                                 data_files="{}{}_valid.json".format(args.data_dir, l),
                                 split="train")
        train_data, valid_data = get_audio_dataset(l, train_ipa, valid_ipa)

        if l == "ta":
            train_data, valid_data = clean_tamil_data(train_data, valid_data)

        # tests
        dataload_test(train_data, train_ipa, valid_data, valid_ipa)

        train_data, valid_data = add_ipa_column(train_data, valid_data, train_ipa, valid_ipa)
        if l == "en":
            train_data = train_data.rename_column("file", "path")
            valid_data = valid_data.rename_column("file", "path")

        if q_filter:
            train_data = filter_low_quality(train_data)
            valid_data = filter_low_quality(valid_data)

        # Clipping to the specified sample size using datasets's Dataset.select()
        train_limit = min(args.train_samples, len(train_data))
        valid_limit = min(args.test_samples, len(valid_data))
        train_data = train_data.select(range(train_limit))
        valid_data = valid_data.select(range(valid_limit))

        train_list.append(train_data)
        valid_list.append(valid_data)

        with open(stats_file, "a") as f:
            f.write(l + " " + str(len(train_data)) + " " + str(len(valid_data))  + "\n")

    # Concatenate the languages
    print("Concatenating datasets for each language...")
    train = concatenate_common_voice(train_list)
    valid = concatenate_common_voice(valid_list)
    print("Concatenation done")

    if args.additional_data:
        train, valid = add_additional_data(train, valid)

    # remove samples that are too long
    train = remove_long_data(train)
    valid = remove_long_data(valid)

    train, valid = remove_unnecessary_columns(train, valid)
    if args.no_space:
        train = train.map(remove_space)
        valid = valid.map(remove_space)
    assert " " not in train[0]["ipa"], \
        print("Space removal did not seem to work correctly")

    train = train.shuffle(seed=42)
    valid = valid.shuffle(seed=35)
    return train, valid


def get_audio_dataset(lang: str,
                      train_ipa: Dataset,
                      valid_ipa: Dataset) -> Tuple[Dataset, Dataset]:
    """Get audio dataset from Common Voice or Librispeech (for English)."""
    if lang == "en":
        # load from Librispeech
        train_ipa = train_ipa.sort("file")
        valid_ipa = valid_ipa.sort("file")
        train_data = load_dataset("librispeech_asr",
                                  split="train.clean.100",
                                  num_proc=args.num_proc)
        train_data = train_data.sort("file")
        train_data = train_data.rename_column("text", "sentence")
        valid_data = load_dataset("librispeech_asr",
                                  split="validation.clean",
                                  num_proc=args.num_proc)
        valid_data = valid_data.sort("file")
        valid_data = valid_data.rename_column("text", "sentence")
    else:
        train_ipa = train_ipa.sort("path")

        # Get raw training dataset                                                                                                                                                                      
        train_data = load_dataset(args.dataset,
                                  lang,
                                  split="train",
                                  num_proc=args.num_proc)
        train_data = train_data.sort("path")

        # Get raw validation dataset from common_voice_11_0
        valid_data = load_dataset(args.dataset,
                                  lang,
                                  split="validation",
                                  num_proc=args.num_proc)
        valid_data = valid_data.sort("path")

    assert train_data[0]["sentence"] == train_ipa[0]["sentence"], \
        (train_data[0]["sentence"], train_ipa[0]["sentence"])
    assert valid_data[0]["sentence"] == valid_ipa[0]["sentence"], \
        (valid_data[0]["sentence"], valid_ipa[0]["sentence"])

    # Adjust the sampling rate
    train_data = train_data.cast_column("audio", Audio(sampling_rate=16_000))
    valid_data = valid_data.cast_column("audio", Audio(sampling_rate=16_000))
    return train_data, valid_data


def clean_tamil_data(train_data: Dataset,
                     valid_data: Dataset) -> Tuple[Dataset, Dataset]:
    """Remove Tamil sentences containing 'ச'"""
    train_data = train_data.filter(lambda batch: "ச" not in batch["sentence"])
    valid_data = valid_data.filter(lambda batch: "ச" not in batch["sentence"])
    return train_data, valid_data


def add_ipa_column(train_data: Dataset,
                   valid_data: Dataset,
                   train_ipa: Dataset,
                   valid_ipa: Dataset) -> Tuple[Dataset, Dataset]:
    """Add a new column for IPA transcription labels."""
    train_ipa = [train_ipa[i]["ipa"] for i in range(len(train_ipa))]
    valid_ipa = [valid_ipa[i]["ipa"] for i in range(len(valid_ipa))]

    train_data = train_data.add_column("ipa", train_ipa)
    valid_data = valid_data.add_column("ipa", valid_ipa)
    return train_data, valid_data


def add_additional_data(train_data: Dataset,
                        valid_data: Dataset) -> Tuple[Dataset, Dataset]:
    """Add additional data from Forvo."""
    print("Concatenating the additional data from Forvo...")
    new_ds: dict = add_forvo.add_language()
    new_ds = new_ds.train_test_split(test_size=0.2)
    train_data = concatenate_datasets([train_data, new_ds["train"]])
    valid_data = concatenate_datasets([valid_data, new_ds["test"]])
    print("Concatenated additional data from Forvo")
    return train_data, valid_data


def remove_unnecessary_columns(train_data: Dataset,
                               valid_data: Dataset) -> Tuple[Dataset, Dataset]:
    """Remove unnecessary columns."""
    unnecessary_columns = ["accent", "age", "client_id", "down_votes", "gender",
                           "locale", "segment", "up_votes",
                           "speaker_id", "chapter_id", "id" # for Librispeech
                           ]
    train_data = train_data.remove_columns(unnecessary_columns)
    valid_data = valid_data.remove_columns(unnecessary_columns)
    assert train_data.features.type == valid_data.features.type
    return train_data, valid_data


def create_vocab(train_data: Dataset,
                 valid_data: Dataset) -> None:
    """Create vocabulary for training."""
    vocab_train_ipa = train_data.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_data.column_names
        )
    vocab_valid_ipa = valid_data.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=valid_data.column_names
        )
    vocab_list_ipa = list(
        set(vocab_train_ipa["vocab"][0]) | set(vocab_valid_ipa["vocab"][0])
        )
    # Add other IPAs including multi-character IPAs
    with open(IPA_LIST, "r") as f:
        lines = f.readlines()
        ipa_all = set([l.strip() for l in lines])
    vocab_list_ipa = set(vocab_list_ipa) | ipa_all
    vocab_list_ipa = list(vocab_list_ipa)
    vocab_dict_ipa = {v: k for k, v in enumerate(vocab_list_ipa)}
    vocab_dict_ipa["[UNK]"] = len(vocab_dict_ipa)
    vocab_dict_ipa["[PAD]"] = len(vocab_dict_ipa)

    with open(args.vocab_file, "w") as f:
        json.dump(vocab_dict_ipa, f)

        
def main(args: argparse.Namespace) -> None:
    """Main function."""
    train_data, valid_data = load_data()
    create_vocab(train_data, valid_data)

    print("Creating the Tokenizer...")
    tokenizer_ipa = Wav2Vec2CTCTokenizer("./{}".format(args.vocab_file),
                                         unk_token="[UNK]",
                                         pad_token="[PAD]",
                                         word_delimiter_token="|")
    print("Tokenizers created")

    print("Creating Feature Extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16_000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=True)
    print("Feature Extractor created")

    print("Creating the Processor...")
    processor_ipa = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer_ipa)
    print("Processor created")

    print("Creating the data collator")
    data_collator = DataCollatorCTCWithPadding(processor=processor_ipa, padding=True)
    print("Data collator created")

    print("Preprocessing the dataset...")
    # Try removing `num_proc=` if you encounter any errors while running this part
    train_data = train_data.map(
        prepare_dataset_ipa,
        remove_columns=train_data.column_names,
        num_proc=args.num_proc
    )
    valid_data = valid_data.map(
        prepare_dataset_ipa,
        remove_columns=valid_data.column_names,
        num_proc=args.num_proc
    )
    print("Dataset lengths to be trained and tested:")
    print("Train:", len(train_data))
    print("Valid:", len(valid_data))
    print("Preprocessing done")

    print("Defining the model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor_ipa.tokenizer.pad_token_id,
        vocab_size=len(processor_ipa.tokenizer)
        )
    print("Model defined")

    # Freeze the feature extractor so that it won't be changed by the fine-tuning
    print("Freezing the feature extractor...") 
    model.freeze_feature_extractor()
    print("Feature extractor frozen")

    output_dir = "./wav2vec2-large-xlsr-{}-ipa".format("".join(args.languages))
    if args.suffix:
        output_dir += args.suffix
    # Training
    print("Beginning the training...") 
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=processor_ipa.feature_extractor,
        )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    args = get_args()
    main(args)
