import re

class Maltese2IPA:
    # rule map
    pron_dict = {"a": "a",
                 "à": "a",
                 "b": "b",
                 "bb": "bː",
                 "b#": "p",
                 "bb#": "pː",
                 "ċ": "t͡ʃ",
                 "ċċ": "t͡ʃː",
                 "d": "d",
                 "dd": "dː",
                 "d#": "t",
                 "dd#": "tː",
                 "e": "ɛ",
                 "è": "ɛ",
                 "f": "f",
                 "ff": "fː",
                 "ġ": "d͡ʒ",
                 "ġġ": "d͡ʒː",
                 "ġ#": "t͡ʃ",
                 "g": "g",
                 "għ": "",
                 "għħ": "ħː",
                 "għh": "ħː",
                 "gg#": "kː",
                 "g#": "k",
                 "h": "",
                 "ħ": "ħ",
                 "ħħ": "ħː",
                 "i": "i",
                 "ì": "i",
                 "ie": "iː",
                 "j": "j",
                 "jj": "jː",
                 "k": "k",
                 "kk": "kː",
                 "l": "l",
                 "ll": "lː",
                 "m": "m",
                 "mm": "mː",
                 "n": "n",
                 "nn": "nː",
                 "o": "o",
                 "ò": "o",
                 "p": "p",
                 "pp": "pː",
                 "q": "ʔ",
                 "qq": "ʔː",
                 "r": "ɾ",
                 "rr": "R",
                 "s": "s",
                 "ss": "sː",
                 "t": "t",
                 "tt": "tː",
                 "u": "u",
                 "ù": "u",
                 "v": "v",
                 "vv": "vː",
                 "v#": "f",
                 "w": "w",
                 "ww": "wː",
                 "x": "ʃ",
                 "xx": "ʃː",
                 "ż": "Z",
                 "żż": "Zː",
                 "żż#": "sː",
                 "ż#": "s",
                 "z": "t͡s",
                 "zz": "t͡sː"}
    PUNCT_REGEX = "[\?,\-\"\'\.]"
    NON_PUNCT = "[\s\w]"
    voiced = set(["b", "d", "g", "ż", "ġ", "v"])
    voiceless = set(["p", "t", "k", "s", "x", "f", "ħ", "h"])
    
    def remove_punct(self, sent):
        sent = re.findall(self.NON_PUNCT, sent.lower(), re.MULTILINE)
        sent = "".join(sent)
        return sent

    def convert_voiceless(self, sent):
        sent = sent.strip()
        newsent = list(sent)
        for i in range(len(sent)):
            if set(sent[i]) & self.voiced:
                if i == len(sent) - 1:
                    newsent[i] = sent[i] + "#"
                else:
                    if sent[i+1] == " ":
                        next = sent[i+2]
                    elif sent[i] == "g" and sent[i+1] == "ħ":
                        continue
                    else:
                        next = sent[i+1]
                    if next in self.voiceless and sent[i] != next:
                        newsent[i] = sent[i] + "#"
        return "".join(newsent)

    def convert_ipa(self, sent):
        three = {k:v for k, v in self.pron_dict.items() if len(k) == 3}
        two = {k: v for k, v in self.pron_dict.items() if len(k) == 2}
        one = {k: v for k, v in self.pron_dict.items() if len(k) == 1}
        for k, v in three.items():
            sent = sent.replace(k, v)
        for k, v in two.items():
            sent = sent.replace(k, v)
        for k, v in one.items():
            sent = sent.replace(k, v)
        # for escaping /z/ /r/
        sent = sent.replace("Z", "z")
        sent = sent.replace("R", "r")
        return sent

    @classmethod
    def maltese_generate_ipa(self, sent):
        """
        This function processes the conversion above at once.
        """
        sent = self.remove_punct(self, sent)
        sent = self.convert_voiceless(self, sent)
        sent = self.convert_ipa(self, sent)
        return sent
