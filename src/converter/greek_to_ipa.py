import re

class Greek2IPA:
    # rule map
    translit_dict = {"μ": "m",
                 "ν": "n",
                 "π": "p",
                 "τ": "t",
                 "κ": "k",
                 "#μπ": "b",
                #  "μπ": "mb",
                 "μπ": "b",
                 "#ντ": "d",
                #  "ντ": "nd",
                 "ντ": "d",
                 "#γκ": "g",
                #  "γκ": "ŋg",
                 "γκ": "g",
                 "γγ": "ŋg",
                 "φ": "f",
                 "θ": "θ",
                 "σ": "s",
                 "χ": "x",
                 "β": "v",
                 "δ": "ð",
                 "ζ": "z",
                 "γ": "ɣ",
                 "ρ": "ɾ",
                 "λ": "l",
                 "τζ": "d͡z",
                 "τσ": "t͡s",
                 "ο": "o",
                 "ε": "e",
                 "α": "a",
                 "η": "i",
                 "ι": "i",
                 "υ": "i",
                 "ω": "o",
                 "αι": "e",
                 "ει": "i",
                 "οι": "i",
                 "ου": "u",
                 "αυ": "av",
                 "αυ#": "af",
                 "ευ": "ev",
                 "ευ#": "ef",
                 "μφ": "ɱf",
                 "ψ": "ps",
                 "ξ": "ks",
    }
    
    postprocess = {"xi": "çi",
               "xe": "çe",
               "xia": "ça",
               "xiu": "çu",
               "xio": "ço",
               "ɣi": "ʝi",
               "ɣe": "ʝe",
               "ɣia": "ʝa",
               "ɣio": "ʝo",
               "ɣiu": "ʝu",
               "li": "ʎi",
               "lia": "ʎa",
               "lio": "ʎo",
               "liu": "ʎu",
               }

    voiceless = set(["π", "τ", "κ", "φ", "θ", "σ", "χ"])
    mutable_v = set(["ευ", "αυ"])
    mutable_c = set(["γκ", "ντ", "μπ"])

    accent_dict = {"ό": "ο",
               "έ": "ε",
               "ά": "α",
               "ή": "η",
               "ί": "ι",
               "ύ": "υ",
               "ώ": "ω",
               "ς": "σ",
               }

    def remove_accent(self, sent: str) -> str:
        for k, v in self.accent_dict.items():
            sent = sent.replace(k, v)
        return sent

    def translit_greek(self, sent: str) -> str:
        three = {k: v for k, v in self.translit_dict.items() if len(k) == 3}
        two = {k: v for k, v in self.translit_dict.items() if len(k) == 2}
        one = {k: v for k, v in self.translit_dict.items() if len(k) == 1}
        newsent = list(sent)
        for i in range(len(newsent)):
            # sentence final devoicing of ev, av
            if i == len(newsent) - 1 and newsent[i-1] + newsent[i] in self.mutable_v:
                newsent[i-1] = "#" + newsent[i-1]
            else:
                if newsent[i-1] + newsent[i] in self.mutable_v and newsent[i+1] in self.voiceless:
                    newsent[i] = newsent[i] + "#"
                # if i >= 2:
                #     if newsent[i-1] + newsent[i] in mutable_c and newsent[i-2] == " ":
                #         newsent[i-1] = "#" + newsent[i-1]
                # elif i == 1 and newsent[i-1] + newsent[i] in mutable_c:
                #     newsent[i-1] = "#" + newsent[i-1]
    
        sent = "".join(newsent)
        for k, v in three.items():
            sent = sent.replace(k, v)
        for k, v in two.items():
            sent = sent.replace(k, v)
        for k, v in one.items():
            sent = sent.replace(k, v)
    
        three_p = {k: v for k, v in self.postprocess.items() if len(k) == 3}
        two_p = {k: v for k, v in self.postprocess.items() if len(k) == 2}
        for k, v in three_p.items():
            sent = sent.replace(k, v)
        for k, v in two_p.items():
            sent = sent.replace(k, v)
        return sent

    def remove_punct(self, sent: str) -> str:
        only_alpha = r"[\s\w]"
        sent = re.findall(only_alpha, sent.lower(), re.MULTILINE)
        sent = "".join(sent)
        return sent

    @classmethod
    def greek_generate_ipa(self, sent: str) -> str:
        """
        This function processes punctuation removal and IPA conversion at once.
        """
        sent = self.remove_punct(self, sent)
        sent = self.remove_accent(self, sent)
        sent = self.translit_greek(self, sent)
        return sent
