import numpy as np
import pandas as pd
import evaluate
from transformers import Wav2Vec2CTCTokenizer

# df = pd.read_csv("features.csv", index_col=0)

def cos_sim(v1, v2):
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0:
        denominator = 0.001
    return np.dot(v1, v2) / denominator

def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]

def LPhD(token1, token2, df):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            # penalty mitigation
            t1_f = df.loc[token1[t1-1], :].to_numpy()[1:]
            t2_f = df.loc[token2[t2-1], :].to_numpy()[1:]
            penalty = 1 - cos_sim(t1_f, t2_f)

            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + penalty
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + penalty
                else:
                    distances[t1][t2] = c + penalty
    
    return distances[len(token1)][len(token2)]

# Spacing Modifier Letters
sml = set()
for i in range(int(0x2b0), int(0x36f)+1):
    sml.add(chr(i))

def retokenize_ipa(sent: str):
    tie_flag = False
    modified = []
    for i in range(len(sent)):
        if tie_flag:
            tie_flag = False
            continue
        if sent[i] in sml:
            if i == 0:
                # when the space modifier letter comes at the index 0
                modified.append(sent[i])
                continue
            modified[-1] += sent[i]
            if sent[i] == "\u0361":
                # tie bar
                modified[-1] += sent[i+1]
                tie_flag = True
        else:
            modified.append(sent[i])
    return modified

def combine_features(phone: str, df):
    # global phone_not_found
    features = np.array([0] * (df.shape[1] - 1))
    for p in phone:
        if p not in set(df.index):
            print("The IPA {} (U+{}) not found in the feature table. We will use zeroed out feature vector instead.".format(p, hex(ord(p))))
            f = np.array([0] * (df.shape[1] - 1))
            # add the unknown phone and its unicode to the dict so that at the end of the evaluation
            # we can get the list of phones unsupported in the feature table
            # phone_not_found[p] = hex(ord(p))
        else:
            f = df.loc[p, :].to_numpy()[1:]
        # print(f)
        features = np.add(features, f)
        # ReLU if necessary
    return features

def preprocessing_combine(sent: str, df) -> pd.DataFrame:
    # df is the feature table
    sent_index = retokenize_ipa(sent)
    sent_array = [[0 for i in range(1, df.shape[1])] for j in range(len(sent_index))]
    sent_df = pd.DataFrame(sent_array, index=sent_index, columns=df.columns[1:])
    # print(sent_df)
    for i, phone in enumerate(sent_df.index):
        if phone in df.index:
            sent_df.iloc[i] = df.loc[phone].to_numpy()[1:]
        else:
            features = combine_features(phone, df)
            sent_df.iloc[i] = features
        # print(phone, features)
    return sent_df

def LPhD_combined(df1, df2):
    distances = np.zeros((df1.shape[0] + 1, df2.shape[0] + 1))

    for t1 in range(df1.shape[0] + 1):
        distances[t1][0] = t1
    for t2 in range(df2.shape[0] + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, df1.shape[0] + 1):
        for t2 in range(1, df2.shape[0] + 1):
            # penalty mitigation
            t1_f = df1.iloc[t1-1]
            t2_f = df2.iloc[t2-1]
            penalty = 1 - cos_sim(t1_f, t2_f)

            if np.equal(df1.iloc[t1-1].to_numpy()[1:], df2.iloc[t2-1].to_numpy()[1:]).all():
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + penalty
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + penalty
                else:
                    distances[t1][t2] = c + penalty
    
    return distances[df1.shape[0]][df2.shape[0]]

def phoneme_error_rate(df1, df2):
    # df2 should be the reference
    df1_list = df1.index
    df2_list = df2.index
    for i, c in enumerate(df1_list):
        if pd.isna(c):
            df1_list.pop(i)
    for i, c in enumerate(df2_list):
        if pd.isna(c):
            df2_list.pop(i)
    phone_LD = levenshteinDistanceDP(df1_list, df2_list)
    ref_length = len(df2_list)
    return phone_LD / ref_length

def compute_all_metrics(pred: str, gold: str, df) -> dict:
    pred = pred.replace("g", "ɡ") # different unicode characters!
    gold = gold.replace("g", "ɡ")
    
    # Levenshtein distance
    ld = levenshteinDistanceDP(pred, gold)

    # Character Error Rate
    cer = ld / len(gold)

    # CER by the evaluation library
    cer_evaluator = evaluate.load("cer")
    predictions = [pred] # arguments must be of list type
    references = [gold]
    cer_eval_score = cer_evaluator.compute(predictions=predictions, references=references)

    # Phoneme Error Rate
    df_pred = preprocessing_combine(pred, df)
    df_gold = preprocessing_combine(gold, df)
    per = phoneme_error_rate(df_pred, df_gold)

    # Levenshtein Phone Distance
    lphd = LPhD_combined(df_pred, df_gold)

    # Feature-weighted Phone Error Rate based on LPhD
    fper = lphd / df_gold.shape[0]
    # shape[0] gives the length of the gold transcription

    output = {"Levenshtein Distance": ld,
              "Character Error Rate": cer,
              "Character Error Rate (evaluate)": cer_eval_score,
              "Phoneme Error Rate": per,
              "Levenshtein Phone Distance": lphd,
              "Feature-weighted Phone Error Rate": fper}

    return output

def compute_only_fper(pred: str, gold: str, df) -> int:
    """Compute Feature-weighted Phone Error Rate.
    
    Args:
        pred: Predicted IPAs
        gold: Gold (label) IPAs
        df: IPA table with features
    Returns:
        fper: Feature-weighted Phone Error Rate for pred
    """
    pred = pred.replace("g", "ɡ") # different unicode characters!
    gold = gold.replace("g", "ɡ")
    df_pred = preprocessing_combine(pred, df)
    df_gold = preprocessing_combine(gold, df)

    # Levenshtein Phone Distance
    lphd = LPhD_combined(df_pred, df_gold)

    # Feature-weighted Phone Error Rate based on LPhD
    fper = lphd / df_gold.shape[0] * 100
    
    return fper
