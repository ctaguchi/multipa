# multipa
MultIPA is yet another automatic speech transcription model into phonetic IPA.
The idea is that, if we train a multilingual speech-to-IPA model with enough amount of good phoneme representations, the model's output will be approximated to phonetic transcriptions.

Note that the codes in this repository are incomplete; we are still cleaning up the original codes for publishing in this repository.
The finalized codes will be prepared by the day of the presentation at INTERSPEECH 2023 (mid-August).
We appreciate your patience!

## Available training languages
At this moment, we have the following languages incorporated available in the training data:
- Finnish
- Hungarian
- Japanese
- Maltese
- Modern Greek
- Polish
- Tamil
- (English)

Note that English was added after the INTERSPEECH 2023 paper was submitted.
We aim to include more languages to take into account linguistic diversity.

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