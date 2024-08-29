import epitran
import re


class Tamil2IPA:
    def tamil_convert(self, sent):
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
                       "mp": "mb",
                       "rr": "tːr",
                       "pp": "pː",
                       "kk": "kː",
                       "xx": "t̪ː",
                       "ʈʈ": "ʈː",
                       "qq": "t͡ɕː",
                       "nr": "ndr",
                       "ɯː": "uː",
        }

        sent = sent.replace("t͡ʃ", "q")
        sent = sent.replace("t̪", "x")
        sent = sent.replace("u", "ɯ")
        
        for k, v in convertable.items():
            sent = sent.replace(k, v)

        sonorants = set(["a", "ɯ", "i", "e", "o", "j", "ɾ"])
        vowel = set(["a", "ɯ", "i", "e", "o"])
        newsent = list(sent)
        for i, c in enumerate(sent):
            if i >= 1 and i < len(sent) - 1:
                if sent[i-1] in sonorants and sent[i+1] in sonorants and sent[i] in voiceable.keys():
                    newsent[i] = voiceable[c]
            # when the preceding vowel is long
            if i >= 2 and i < len(sent) - 1:
                if sent[i-2] in vowel and sent[i-1] == "ː" and sent[i+1] in sonorants and sent[i] in voiceable.keys():
                    newsent[i] = voiceable[c]
        sent = "".join(newsent)
        sent = sent.replace("q", "t͡ɕ")
        sent = sent.replace("x", "t̪")

        tokens = sent.split()
        for i, t in enumerate(tokens):
            if t.startswith("e"):
                tokens[i] = "j" + t
                tokens[i].strip()
        sent = " ".join(tokens)

        alpha = r"[\s\w\u0250-\u02AF\u02B0-\u02FF\u1D00-\u1D7F\u1D80–\u1DBF\u0300-\u036F]"
        sent = re.findall(alpha, sent, re.MULTILINE)
        sent = "".join(sent)
    
        return sent

    @classmethod
    def tamil_generate_ipa(cls, sent):
        epi_ta = epitran.Epitran("tam-Taml")
        epi_ta_ipa = epi_ta.transliterate(sent)
        epi_ta_ipa = epi_ta_ipa.replace("u", "ɯ")
        converted = cls.tamil_convert(cls, epi_ta_ipa)
        return converted
