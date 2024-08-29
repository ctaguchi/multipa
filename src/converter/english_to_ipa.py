import eng_to_ipa as ipa

class English2IPA:
    def __init__(cls):
        filename = "cmudict-0.7b-ipa.txt"
        prondict = cls.make_prondict(filename)

    @classmethod
    def english_generate_ipa(cls, sent: str):
        words = sent.lower().split()
        transcription = []
        keys = prondict.keys()
        for w in words:
            if w not in keys:
                addendum = ipa.convert(w)
            else:
                addendum = prondict[w][0]
            if not suprasegmental:
                addendum.replace("ˈ", "").replace("ˌ", "")
            transcription.append(addendum)
        output = " ".join(transcription)
        return output

    def make_prondict(cls, filename: str) -> dict:
        with open(filename, "r") as f:
            entries = f.readlines()
            prondict = dict()
            for e in entries:
                e = e.strip()
                k, v = e.split("\t")
                if "," in v:
                    v = v.split(", ")
                    prondict[k.lower()] = v
                else:
                    prondict[k.lower()] = [v]
        return prondict
