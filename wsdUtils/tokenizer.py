import os
import nltk
import treetaggerwrapper as ttw

DEFAULT_TAGGER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../TreeTagger"))


"""
Tokenize with NLTK and use TreeTagger for POS and lemma
"""


class Tokenizer:

    def __init__(self, tagger_path: str = DEFAULT_TAGGER_PATH, lang: str = "de"):
        self.tagger = ttw.TreeTagger(TAGLANG="de", TAGDIR=tagger_path)
        if lang != "de":
            raise NotImplementedError
        self.lang = lang

    def tokenize_raw(self, text: str):
        """
        Tokenizes and lemma/pos tags a raw string. Returns a list of (form, lemma, pos) tuples
        :param text:
        :return:
        """
        nltk_lang_table = {"de": "german"}

        tokens = nltk.word_tokenize(text, language=nltk_lang_table[self.lang])
        tags = ttw.make_tags(self.tagger.tag_text(tokens, tagonly=True))
        output = []
        for token, tag in zip(tokens, tags):
            lemma = tag.lemma
            if (token == "``" and "``" not in text) or (token == "´´" and "´´" not in text):
                token = "\""
                lemma = "\""
            output.append((token, lemma, tag.pos))
        return output


if __name__ == "__main__":
    tok = Tokenizer()
    out = tok.tokenize_raw("Dann ruf ihn doch an.")
    for tup in out:
        print("\t".join(tup))
