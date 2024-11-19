import os
import glob
import subprocess
from datasets import Dataset, Audio, concatenate_datasets
import pandas as pd
from typing import List

def add_language() -> List[dict]:
    # a path looks like this: "additional/Xhosa/Audio/{$AUDIOFILE}.wav" or "additional/Xhosa/{$PRONFILE}.txt
    # Used in the article: Adyghe(Adygean), Arabic, Burmese, Icelandic, Xhosa, and Zulu
    path = "./additional/"
    if not os.path.exists(path):
        path = "../additional/"
    langs = [l for l in os.listdir(path) if not l.startswith(".")]
    for l in langs:
        assert l.istitle(), print("The first letter of the language must be upper-case (i.e., title case)")

        audio_path = path + "/" + l + "/Audio/wav"
        text_file = path + "/" + l + "/ipa.txt"
        audio_files = os.listdir(audio_path) # -> list of files in the directory

        with open(text_file, "r") as f:
            # a line will look like this:
            # pronunciation_xh_amabhaca amabaːǀa
            ipas = [l.strip() for l in f.readlines() if l != "\n"]
            assert not ipas[0].endswith("\n"), print("line break letter found at the end of the line.")
            assert len(ipas) == len(audio_files), print("Numbers of audio files and IPAs don't match")
    
        ds = dict()
        ds["path"] = list()
        ds["ipa"] = list()
        ds["sentence"] = list()
        for l in ipas:
            # print(l.split(" "))
            filename = l.split()[0]
            pron = " ".join(l.split()[1:])
            sent = " ".join(filename.split("_")[2:])
            print(filename)
            filename = filename + ".wav"
            assert filename in audio_files, print("Audio file not found. Check the file name or the directory.")
            file_path = audio_path + "/" + filename
            ds["path"].append(file_path)
            ds["ipa"].append(pron)
            ds["sentence"].append(sent)
        df = pd.DataFrame(ds) # -> DataFrame
        ds = Dataset.from_pandas(df) # -> Dataset
        
        # Read binary data (array) of the audio files
        audio_files_with_path = glob.glob(audio_path + "/*")
        audio_data = Dataset.from_dict({"audio": audio_files_with_path}).cast_column("audio", Audio(sampling_rate=48000))
        # -> Dataset

        # Concatenate ds and the audio column w.r.t. the column
        ds = concatenate_datasets([ds, audio_data], axis=1)

    return ds
