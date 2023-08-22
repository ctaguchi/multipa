# multipa
MultIPA is yet another automatic speech transcription model into phonetic IPA.
The idea is that, if we train a multilingual speech-to-IPA model with enough amount of good phoneme representations, the model's output will be approximated to phonetic transcriptions.
Please check out the [Paper](https://arxiv.org/abs/2308.03917) for details.

## Available training languages
At this moment, we have the following languages incorporated available in the training data:
- Finnish
- Hungarian
- Japanese
- Maltese
- Modern Greek
- Polish
- Tamil

We aim to include more languages to take into account linguistic diversity.

## How to run
You will first need to convert the transcription in the CommonVoice dataset into IPA.
To do so, run `preprocess.py`; for example,
```
python preprocess.py \
       -l ja pl mt hu fi el ta \
       --num_proc 48
```

Then, run `main.py` to train a model.
For example:
```
python3 main_general_preprocessed_allipa.py \
        -l ja pl mt hu fi el ta \
        -tr 1000 1000 1000 1000 1000 1000 1000 \
        -te 200 200 200 200 200 200 200 \
        -qf False False False False False False False \
        -a True \
        -s "japlmthufielta-nq-ns" \
        -ns True \
        -v vocab.json \
        -e 10
```
for training with 7 languages, 1000 training samples and 200 validation samples for each, where audio samples with bad quality are not filtered out, additional data from Forvo are included, the suffix for the output model folder name is `japlmthufielta-nq-ns`, orthographic spaces are removed, the name of the vocab file is `vocab.json`, and the number of epochs is set to 10.

## Model
You can run the model (trained on 1k samples for each language, 9h in total) [here](https://huggingface.co/ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns).

## Notes
- If you are using AFS, `preprocess.py` might cause `OS Error: File too large` due to reaching the limit of the number of files that a directory can accommodate.
- Additional data from Forvo themselves are not uploaded in this repository.

## Citation
Chihiro Taguchi, Yusuke Sakai, Parisa Haghani, David Chiang. "Universal Automatic Phonetic Transcription into the International Phonetic Alphabet". INTERSPEECH 2023.\
For the time being, you may cite our arXiv paper:
```
@misc{taguchi2023universal,
      title={Universal Automatic Phonetic Transcription into the International Phonetic Alphabet}, 
      author={Chihiro Taguchi and Yusuke Sakai and Parisa Haghani and David Chiang},
      year={2023},
      eprint={2308.03917},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
Feel free to raise issues if you find any bugs.
Also, feel free to contact me `ctaguchi at nd.edu` for collaboration.