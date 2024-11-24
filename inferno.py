import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import jiwer
import numpy as np
import gc
import csv

from converter.japanese_to_ipa import Japanese2IPA
from converter.maltese_to_ipa import Maltese2IPA
from converter.finnish_to_ipa import Finnish2IPA
from converter.greek_to_ipa import Greek2IPA
from converter.tamil_to_ipa import Tamil2IPA
from converter.english_to_ipa import English2IPA
from epitran import Epitran
import re

cache_dir = "./cache"

# Caminho do modelo
model_dir = "/mnt/5fc7fd01-6487-4f39-8109-556023ff1f7f/puc/7 sem/topicos/multipa/wav2vec2-large-xlsr-japlmthufiel-ipajaplmthufielta-nq-ns"

# Inicializar o processador e o modelo
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.eval()

def transliterate_sentence(lang, sentence):
    """
    Função para converter a sentença para IPA de acordo com o idioma.
    """
    if lang == "ja":
        converter = Japanese2IPA()
        ipa = converter.remove_ja_punct(sentence)
        ipa = converter.convert_sentence_to_ipa(ipa)
    elif lang == "mt":
        ipa = Maltese2IPA.maltese_generate_ipa(sentence)
    elif lang == "fi":
        ipa = Finnish2IPA.finnish_generate_ipa(sentence)
    elif lang == "el":
        ipa = Greek2IPA.greek_generate_ipa(sentence)
    elif lang == "hu":
        ipa = re.findall(r"[\s\w]", sentence.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("hun-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "pl":
        ipa = re.findall(r"[\s\w]", sentence.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("pol-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "ta":
        ipa = Tamil2IPA.tamil_generate_ipa(sentence)
    elif lang == "en":
        ipa = English2IPA.english_generate_ipa(sentence)
    else:
        raise Exception("Unknown locale (language) found")
    
    return "".join(ipa.split())

def calculate_per_and_pfer(reference_transcriptions, predicted_transcriptions):
    """
    Calcula o WER (Word Error Rate) e PFER (Phone Error Rate).
    """
    wer_scores = []
    pfer_scores = []

    for ref, pred in zip(reference_transcriptions, predicted_transcriptions):
        # Calcular WER
        wer_score = jiwer.wer(ref, pred)
        wer_scores.append(wer_score)

        # PFER é calculado de maneira semelhante ao WER
        pfer_score = jiwer.wer(ref, pred)  # Usando WER como aproximação de PFER
        pfer_scores.append(pfer_score)

    return np.mean(wer_scores), np.mean(pfer_scores)

def test_model_on_samples(model_dir, languages, num_samples, cache_dir='./cache'):
    """
    Realiza inferência do modelo e calcula WER/PFER para as transcrições do dataset.
    """
    results = {}

    for lang in languages:
        print(f"\nTesting language: {lang}")

        # Carregar o dataset
        try:
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="test", cache_dir=cache_dir)
            print(f"Dataset {lang} loaded.")
        except Exception as e:
            print(f"Error loading dataset {lang}: {e}")
            continue

        reference_transcriptions = []
        predicted_transcriptions = []

        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            # Preprocessamento do áudio
            input_values = processor(sample["audio"]["array"], return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            # Gerar transcrição IPA da sentença original
            ipa_transcription = transliterate_sentence(lang, sample["sentence"])

            # Armazenar as transcrições de referência e previstas
            reference_transcriptions.append(ipa_transcription)
            predicted_transcriptions.append(transcription)

            print(f"Sample {i+1}: {sample['sentence']}")
            print(f"Predicted IPA: {transcription}")
            print(f"Reference IPA: {ipa_transcription}")

            # Limpeza de memória
            del sample, input_values, logits, predicted_ids, transcription, ipa_transcription
            gc.collect()

        # Calcular WER e PFER para o idioma
        wer, pfer = calculate_per_and_pfer(reference_transcriptions, predicted_transcriptions)
        results[lang] = {
            "wer": wer,
            "pfer": pfer
        }
        print(f"Results for {lang}:")
        print(f"WER: {wer:.4f}")
        print(f"PFER: {pfer:.4f}")

    return results

# Definir idiomas e número de amostras
languages = ["ja", "pl", "mt", "hu", "fi", "el"]
num_samples = 5
output_file = "model_data.csv"

# Realizar testes e salvar resultados
results = test_model_on_samples(model_dir, languages, num_samples, cache_dir)

# Salvar resultados em um arquivo CSV
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Language", "WER", "PFER"])
    for lang, metrics in results.items():
        writer.writerow([lang, metrics["wer"], metrics["pfer"]])
        print(f"\nResults for {lang}:")
        print(f"WER: {metrics['wer']:.4f}")
        print(f"PFER: {metrics['pfer']:.4f}")

print(f"\nResults saved to {output_file}")
