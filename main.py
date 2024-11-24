from datasets import load_dataset, Audio, concatenate_datasets, Dataset, DatasetDict
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import argparse
import pandas as pd
import os
import multiprocess
from tqdm import tqdm

from data_utils import filter_low_quality, downsampling

def extract_all_chars_ipa(batch: dict) -> dict:
    # Change this function later at some point to create vocabulary based on
    # phonemes, not on characters
    all_text = " ".join(batch["ipa"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset_ipa(batch: dict) -> dict:
    audio = batch["audio"]

    # batched output is unbatched
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

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def remove_long_data(dataset, max_seconds=6):
    # convert pyarrow table to pandas
    dftest = dataset.to_pandas()
    # find out length of input_values
    dftest['len'] = dftest['input_values'].apply(len)
    # for wav2vec training we already resampled to 16khz
    # remove data that is longer than max_seconds (6 seconds ideal)
    maxLength = max_seconds * 16000 
    dftest = dftest[dftest['len'] < maxLength]
    dftest = dftest.drop('len', 1)
    # convert back to pyarrow table to use in trainer
    dataset = dataset.from_pandas(dftest)
    # directly remove do not wait for gc
    del dftest
    return dataset

def concatenate_common_voice(datasetlist: list):
    """
    Concatenate more than one datasets from Common Voice.
    Also consider using datasets.interleave_datasets(datasets: List[DatasetType]
    so that the new dataset is constructed by cycling between each source to get the examples.
    """
    init_data = datasetlist[0]
    for d in datasetlist:
        assert d.features.type == init_data.features.type
    concatenated = concatenate_datasets(datasetlist)
    return concatenated

def remove_space(batch: dict) -> dict:
    ipa = batch["ipa"]
    ipa = ipa.split()
    ipa = "".join(ipa)
    batch["ipa"] = ipa
    return batch

def dataload_test(train_data, train_ipa, valid_data, valid_ipa):
    assert len(train_data) == len(train_ipa), "Length of train_data and train_ipa does not match"
    assert len(valid_data) == len(valid_ipa), "Length of valid_data and valid_ipa does not match"
    
    if l == "en":
        print("Validating train data...")
        for j in tqdm(range(len(train_data)), desc="Train Data Validation", unit="file"):
            filename = train_data[j]["file"]
            ipa_filename = train_ipa[j]["file"]
            assert filename == ipa_filename
        
        print("Validating valid data...")
        for j in tqdm(range(len(valid_data)), desc="Valid Data Validation", unit="file"):
            filename = valid_data[j]["file"]
            ipa_filename = valid_ipa[j]["file"]
            assert filename == ipa_filename
    else:
        print("Validating train data...")
        for j in tqdm(range(len(train_data)), desc="Train Data Validation", unit="file"):
            filename = train_data[j]["path"].split("/")[-1]
            ipa_filename = train_ipa[j]["path"].split("/")[-1]
            assert filename == ipa_filename
        
        print("Validating valid data...")
        for j in tqdm(range(len(valid_data)), desc="Valid Data Validation", unit="file"):
            filename = valid_data[j]["path"].split("/")[-1]
            ipa_filename = valid_ipa[j]["path"].split("/")[-1]
            assert filename == ipa_filename

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Specify languages to use and options for each language")

    parser.add_argument("-l", "--languages", nargs="*", type=str, required=True,
                        help="Specify language code (split by space). Typically ISO639-1, or ISO639-2 if not found in ISO639-1.")
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
    args = parser.parse_args()
    lgx = args.languages
    suffix = args.suffix
    
    assert len(args.train_samples) <= len(lgx), "`train_samples` argument is longer than the number of languages"
    assert len(args.test_samples) <= len(lgx), "`test_samples` argument is longer than the number of languages"
    assert len(args.quality_filter) <= len(lgx), "`quality_filter` argument is longer than the number of languages"

    if args.additional_data:
        from add_forvo import add_language
    
    train_list = []
    valid_list = []
    # Data loading
    stats_file = "stats_train_valid_{}.txt".format(suffix)
    with open(stats_file, "w") as f:
        f.write("lang train valid\n")
    for i, l in enumerate(lgx):
        train_sample = args.train_samples[i]
        test_sample = args.test_samples[i]
        q_filter = args.quality_filter[i]
        if l == "en":
            q_filter = False

        # Get preprocessed training dataset with IPA
        train_ipa = load_dataset("json",
                                 data_files="{}{}_train.json".format(args.data_dir, l),
                                 split="train",)
                                #  cache_dir=os.getenv("CACHE_DIR"))
        valid_ipa = load_dataset("json",
                                 data_files="{}{}_valid.json".format(args.data_dir, l),
                                 split="train",)
                                #  cache_dir=os.getenv("CACHE_DIR"))
        if l == "en":
            # Librispeech's file name column is "file"
            train_ipa = train_ipa.sort("file")
            valid_ipa = valid_ipa.sort("file")

            # Get raw training dataset
            train_data = load_dataset("librispeech_asr",
                                      split="train.clean.100",
                                      num_proc=args.num_proc)
            train_data = train_data.sort("file")
            train_data = train_data.rename_column("text", "sentence")

            # Get raw validation dataset
            valid_data = load_dataset("librispeech_asr",
                                      split="validation.clean",
                                      num_proc=args.num_proc)
            valid_data = valid_data.sort("file")
            valid_data = valid_data.rename_column("text", "sentence")
        else:
            train_ipa = train_ipa.sort("path")
            valid_ipa = valid_ipa.sort("path")

            # Get raw training dataset
            train_data = load_dataset(args.dataset,
                                      l,
                                      split="train",
                                      num_proc=args.num_proc)
            train_data = train_data.sort("path")

            # Get raw validation dataset from common_voice_11_0
            valid_data = load_dataset(args.dataset,
                                     l,
                                     split="validation",
                                     num_proc=args.num_proc)
            valid_data = valid_data.sort("path")

        assert train_data[0]["sentence"] == train_ipa[0]["sentence"], (train_data[0]["sentence"], train_ipa[0]["sentence"])
        assert valid_data[0]["sentence"] == valid_ipa[0]["sentence"], (valid_data[0]["sentence"], valid_ipa[0]["sentence"])

        # Remove Tamil sentences containing "ச"
        if l == "ta":
            train_data = train_data.filter(lambda batch: "ச" not in batch["sentence"])
            valid_data = valid_data.filter(lambda batch: "ச" not in batch["sentence"])

        # tests
        print("Testing dataset")
        # dataload_test(train_data, train_ipa, valid_data, valid_ipa)

        train_ipa = [train_ipa[i]["ipa"] for i in tqdm(range(len(train_ipa)), desc="Processing train_ipa")]
        valid_ipa = [valid_ipa[i]["ipa"] for i in tqdm(range(len(valid_ipa)), desc="Processing valid_ipa")]
        # train_ipa = [train_ipa[i]["ipa"] for i in range(len(train_ipa))]
        # valid_ipa = [valid_ipa[i]["ipa"] for i in range(len(valid_ipa))]

        # Combine the IPA column
        print("Combining columns")
        train_data = train_data.add_column("ipa", train_ipa)
        valid_data = valid_data.add_column("ipa", valid_ipa)
        if l == "en":
            train_data = train_data.rename_column("file", "path")
            valid_data = valid_data.rename_column("file", "path")

        if q_filter:
            train_data = filter_low_quality(train_data)
            valid_data = filter_low_quality(valid_data)

        # Clipping to the specified sample size using datasets's Dataset.select()
        train_limit = min(train_sample, len(train_data))
        valid_limit = min(test_sample, len(valid_data))
        train_data = train_data.select(range(train_limit))
        valid_data = valid_data.select(range(valid_limit))
        
        train_list.append(train_data)
        valid_list.append(valid_data)

        with open(stats_file, "a") as f:
            f.write(l + " " + str(len(train_data)) + " " + str(len(valid_data))  + "\n")
    
    # Concatenate the languages
    print("Concatenating datasets for each language...")
    common_voice_train = concatenate_common_voice(train_list)
    common_voice_valid = concatenate_common_voice(valid_list)
    print("Concatenation done")

    # if args.additional_data:
    #     print("Concatenating the additional data from Forvo...")
    #     new_ds = add_language() # -> dict
    #     new_ds = new_ds.train_test_split(test_size=0.2)
    #     common_voice_train = concatenate_datasets([common_voice_train, new_ds["train"]])
    #     common_voice_valid = concatenate_datasets([common_voice_valid, new_ds["test"]])
    #     print("Concatenated additional data from Forvo")

    # Remove unnecessary columns
    available_columns_train = common_voice_train.column_names
    available_columns_valid = common_voice_valid.column_names
    columns_to_remove = [
        "accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes",
        "speaker_id", "chapter_id", "id" 
    ]

    filtered_columns_train = [col for col in columns_to_remove if col in available_columns_train]
    filtered_columns_valid = [col for col in columns_to_remove if col in available_columns_valid]

    print("Removing unnecessary columns...")
    common_voice_train = common_voice_train.remove_columns(filtered_columns_train)
    common_voice_valid = common_voice_valid.remove_columns(filtered_columns_valid)
    print("Unnecessary columns removed. Data preview:")
    print(common_voice_train[0])
    assert common_voice_train.features.type == common_voice_valid.features.type

    # Remove spaces if specified
    if args.no_space:
        common_voice_train = common_voice_train.map(remove_space)
        common_voice_valid = common_voice_valid.map(remove_space)
    assert " " not in common_voice_train[0]["ipa"], print("Apparently space removal did not work correctly")
        
    # Shuffle the dataset
    print("Shuffling the dataset...")
    common_voice_train = common_voice_train.shuffle(seed=42)
    common_voice_valid = common_voice_valid.shuffle(seed=35)
    print("Shuffling done")

    # Preprocessing 
    print("Creating vocabulary...")
    vocab_train_ipa = common_voice_train.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=common_voice_train.column_names
    )
    vocab_valid_ipa = common_voice_valid.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=common_voice_train.column_names
    )
    vocab_list_ipa = list(
        set(vocab_train_ipa["vocab"][0]) | set(vocab_valid_ipa["vocab"][0])
    )
    # add multiletter IPAs and other IPAs
    with open("full_vocab_ipa.txt", "r") as f:
        lines = f.readlines()
        ipa_all = set([l.strip() for l in lines])
    vocab_list_ipa = set(vocab_list_ipa) | ipa_all
    vocab_list_ipa = list(vocab_list_ipa)
    vocab_dict_ipa = {v: k for k, v in enumerate(vocab_list_ipa)}

    print("Vocab created. Details:")
    print("vocab_dict_ipa: {}".format(len(vocab_dict_ipa)))

    # Preprocessing necessary for CTC
    # Add [UNK], [PAD]
    print("Adding [UNK] and [PAD]...")
    vocab_dict_ipa["[UNK]"] = len(vocab_dict_ipa)
    vocab_dict_ipa["[PAD]"] = len(vocab_dict_ipa)
    print("[UNK] and [PAD] added")

    print("Writing vocab json files...")
    # Don't forget to change the file name when you use different languages,
    # otherwise the vocab file will be lost
    # filename = "vocab_ipa_{}.json".format("".join(lgx))
    with open(args.vocab_file, 'w') as vocab_file_ipa:
        json.dump(vocab_dict_ipa, vocab_file_ipa)
    print("Vocab json files created")

    # Create Tokenizers
    print("Creating Tokenizers...")
    # Be careful to load the correct vocab file.
    tokenizer_ipa = Wav2Vec2CTCTokenizer("./{}".format(args.vocab_file),
                                         unk_token="[UNK]",
                                         pad_token="[PAD]",
                                         word_delimiter_token="|")
    print("Tokenizers created") 

    # Create a Feature Extractor
    print("Creating Feature Extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16_000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=True)
    print("Feature Extractor created") 

    # Define Processors
    print("creating Processors...")
    processor_ipa = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer_ipa)
    print("Processors created") 

    # Set the sampling rate to 16,000Hz
    print("Adjusting the sampling rate to 16,000Hz...")
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_valid = common_voice_valid.cast_column("audio", Audio(sampling_rate=16_000))
    print("Sampling rate adjustment done")

    print("Preprocessing the dataset...")
    # Try removing `num_proc=` if you encounter any errors while running this part
    common_voice_train = common_voice_train.map(
        prepare_dataset_ipa,
        remove_columns=common_voice_train.column_names,
        # num_proc=args.num_proc
    )
    common_voice_valid = common_voice_valid.map(
        prepare_dataset_ipa,
        remove_columns=common_voice_valid.column_names,
        # num_proc=args.num_proc
    )
    print("Removing audio files longer than 6 secs...")
    common_voice_train = remove_long_data(common_voice_train)
    common_voice_valid = remove_long_data(common_voice_valid)
    print("Dataset lengths to be trained and tested:")
    print("Train:", len(common_voice_train))
    print("Valid:", len(common_voice_valid))
    print("Preprocessing done")

    print("Creating the data collator")
    data_collator = DataCollatorCTCWithPadding(processor=processor_ipa, padding=True)
    print("Data collator created")
    
    # Model
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

    output_dir = "./wav2vec2-large-xlsr-{}-ipa".format("".join(lgx))
    if suffix:
        output_dir += suffix
    # Training
    print("Beginning the training...") 
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=4, #4
        gradient_accumulation_steps=2, #2
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        fp16=False, #True only if CUDA-capable GPU
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_valid,
        tokenizer=processor_ipa.feature_extractor,
        )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()
    # trainer.push_to_hub(repo_name="wav2vec2-ipa")
