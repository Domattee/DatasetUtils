import math
import json
import jsonpickle
import argparse
import os
import random

from lxml import etree as et
from typing import List, Dict, Union
from nltk.corpus import wordnet as wn


"""
Defines the data format for the pipeline scripts train/eval for wsd.
Files are JSON Objects containing the following fields:
'name': Arbitary name for the dataset
'lang': Language for the dataset
'labeltype': What type of labels this dataset uses. Must be one of ["wnoffsets", "bnids", "gn"].
'entries': A list of Objects, each representing a single training/disambiguation instance that must contain the 
    following fields for EWISER processing:
    'label': String, Gold label for disambiguation. This must be either wordnet offsets, babelnet ids or germanet ids 
    'lemma': String, target lemma for disambiguation.
    'upos': String, target pos. Must be either 'NOUN', 'VERB', 'ADJ' or 'ADV'
    'tokens': A list of json objects which each object representing a token. Tokens must have the fields:
        'form': String, Wordform
        'lemma': String, lemma
        'pos': String, pos. Assumed to be TIGER STTS
        'upos': String, coarse upos. If this field is missing we will try to produce it from 'pos'.
        'begin': Integer, index of the first character of the token in the text.
        'end': Integer, index of the last character of the token in the text
        'is_pivot': Boolean, whether or not this is the target token for the label
Optional fields for instances are:
    'pivot_start': Integer, index of the first character of the target word
    'pivot_end': Integer, index of the last character of the target word
    'sentence': String, complete sentence.
    'source_id': String, arbitary identification for the source of the instance
Instances can contain additional fields without interfering, but they will not be used/considered by these functions.
Its important that the 'lemma' and 'pos' field for the whole instance matches those for the specific target token. 
"""
# TODO: Should turn valid labeltypes into an enum or something of the sort, so we don't have to adjust a
#  dozen files if we change names
# TODO: Loading and saving seems very slow
# TODO: Raganato saving and loading, with pivot info, etc...
# TODO: Get rid of sentences as string and construct them from tokens maybe?
VALID_LABELTYPES = ["wnoffsets", "bnids", "gn"]


class WSDToken:
    def __init__(self, form: str, lemma: str, pos: str, begin: int, end: int, upos: str = None):
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.upos = upos
        self.begin = begin
        self.end = end

    def __str__(self):
        rep_items = [self.form, self.lemma, self.pos, self.upos, self.begin, self.end]
        return "\t".join(map(lambda x: str(x), rep_items))

    def is_pivot(self, entry):
        """Check if this token is the pivot token of entry"""
        return self in entry.tokens and entry.pivot_start == self.begin and entry.pivot_end == self.end


class SentenceWithTokens:
    def __init__(self, sentence: str, tokens: List[WSDToken] = None):
        self.sentence = sentence
        self.tokens = tokens


class WSDEntry:
    # TODO: Replace sentence with a function based on tokens, make tokens non-optional
    def __init__(self, dataset: "WSDData", sentence_idx: int, label: str, lemma: str, upos: str,
                 source_id: str = None, pivot_start: int = None, pivot_end: int = None):
        self.dataset = dataset
        self.label = label
        self.lemma = lemma
        self.sentence_idx = sentence_idx
        self.upos = upos
        self.source_id = source_id
        self.pivot_start = pivot_start
        self.pivot_end = pivot_end

    def __str__(self):
        rep_items = [self.label, self.lemma, self.pivot_start, self.pivot_end, self.source_id,
                     "{} tokens".format(0 if self.tokens is None else len(self.tokens))]
        return "\t".join(map(lambda x: str(x), rep_items))

    # This is used by jsonpickle to determine what gets written out. @property functions are not in __dict__ and aren't
    # written and attributes manually excluded from __dict__ here are not written either
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["dataset"]  # Don't need this in the file to restore
        return state

    def get_dict(self):
        return {"label": self.label,
                "lemma": self.lemma,
                "upos": self.upos,
                "sentence": self.sentence,
                "tokens": self.tokens,
                "source_id": self.source_id,
                "pivot_start": self.pivot_start,
                "pivot_end": self.pivot_end}

    @property
    def sentence(self):
        return self.dataset._sentences[self.sentence_idx].sentence

    @sentence.setter
    def sentence(self, new_sentence: str):
        if new_sentence in self.dataset._sentence_cache:
            # Iterate over sentences and check if new sentence is identical to any
            self.sentence_idx = self.dataset._sentence_cache[new_sentence]
        else:
            self.sentence_idx = self.dataset._add_sentence(new_sentence, None)

    @property
    def tokens(self):
        return self.dataset._sentences[self.sentence_idx].tokens

    @tokens.setter
    def tokens(self, new_tokens: List[WSDToken]):
        # This sets or corrects tokens for the sentence
        self.dataset._sentences[self.sentence_idx].tokens = new_tokens

    def is_tokenized(self):
        return self.dataset._sentences[self.sentence_idx].tokens is not None

        
class WSDData:
    """ Cache works as such: OrderedDict of sentences and tokens"""
    def __init__(self, name: str, lang: str, labeltype: str):
        assert labeltype in VALID_LABELTYPES
        self.name = name
        self.lang = lang
        self.labeltype = labeltype
        self.entries: List[WSDEntry] = []
        self._sentences: Dict[int, SentenceWithTokens] = {}
        self._sentence_cache = {}

    def __str__(self):
        return "Dataset {}, language {}, labels {} with {} entries".format(self.name,
                                                                           self.lang, self.labeltype, len(self.entries))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_sentence_cache"]  # Prevents this attribute from being printed to file
        return state

    def _add_sentence(self, sentence: str, tokens: List[WSDToken] = None, replace_tokens=False):
        if sentence in self._sentence_cache:
            idx = self._sentence_cache[sentence]
            if replace_tokens:
                self._sentences[idx].tokens = tokens
            return idx
        else:
            idx = 0
            while idx in self._sentences:
                idx += 1
            self._sentences[idx] = SentenceWithTokens(sentence, tokens)
            self._sentence_cache[sentence] = idx
            return idx

    def _clean_sentences(self):
        """ Removes all sentences from the caches which are not being used by entries"""
        used_idxs = [entry.sentence_idx for entry in self.entries]
        for sentence, idx in list(self._sentence_cache.items()):
            if idx not in used_idxs:
                del self._sentence_cache[sentence]
                del self._sentences[idx]

    def add_entry(self, label: str, lemma: str, upos: str, sentence: str, tokens: List[WSDToken] = None,
                  source_id: str = None, pivot_start: int = None, pivot_end: int = None):
        idx = self._add_sentence(sentence, tokens)
        self.entries.append(WSDEntry(self, idx, label, lemma, upos,
                                     source_id=source_id, pivot_start=pivot_start, pivot_end=pivot_end))

    @classmethod
    def load(cls, json_path: str):
        with open(json_path, "r", encoding="utf8") as f:
            loaded = json.load(f)
            lang = loaded["lang"]
            name = loaded["name"]
            labeltype = loaded["labeltype"]
            assert labeltype in VALID_LABELTYPES
            dataset = cls(name, lang, labeltype)
            sentences = {}
            for sentence_id, sent_with_tokens in loaded["_sentences"].items():
                sentence = sent_with_tokens["sentence"]
                tokens = []
                for json_token in sent_with_tokens["tokens"]:
                    tokens.append(WSDToken(json_token["form"],
                                           json_token["lemma"],
                                           json_token["pos"],
                                           json_token["begin"],
                                           json_token["end"],
                                           upos=json_token["upos"]
                                           ))
                sentences[int(sentence_id)] = SentenceWithTokens(sentence, tokens)

            for entry in loaded["entries"]:
                label = entry["label"]
                target_lemma = entry["lemma"]
                entry_pos = entry["upos"]
                sentence_idx = int(entry["sentence_idx"])
                sentence = sentences[sentence_idx].sentence
                tokens = sentences[sentence_idx].tokens
                if sentence in dataset._sentence_cache:
                    assert dataset._sentence_cache[sentence] == sentence_idx
                else:
                    dataset._sentence_cache[sentence] = sentence_idx
                source = load_opt(entry, "source_id", default=None)
                pivot_start = load_opt(entry, "pivot_start", default=None)
                if pivot_start is not None:
                    pivot_start = int(pivot_start)
                pivot_end = load_opt(entry, "pivot_end", default=None)
                if pivot_end is not None:
                    pivot_end = int(pivot_end)

                dataset.add_entry(
                    label,
                    target_lemma,
                    entry_pos,
                    sentence,
                    tokens=tokens,
                    source_id=source,
                    pivot_start=pivot_start,
                    pivot_end=pivot_end
                )
            dataset._sentences = sentences
        return dataset

    @classmethod
    def load_legacy(cls, json_path: str, correct_pivot_tokens: bool = True):
        """ correct_pivot_tokens replaces upos and lemma data for tokens with is_pivot==True with the entry upos and
        lemma fields. This can help correct issues where the token information is incorrect due to automatic taggers.
        We are assumming that entry information is correct here due to manual annotation"""
        with open(json_path, "r", encoding="utf8") as f:
            loaded = json.load(f)
            lang = loaded["lang"]
            name = loaded["name"]
            labeltype = loaded["labeltype"]
            assert labeltype in VALID_LABELTYPES
            dataset = cls(name, lang, labeltype)
            for entry in loaded["entries"]:
                label = entry["label"]
                target_lemma = entry["lemma"]
                entry_pos = entry["upos"]
                
                sentence = entry["sentence"]
                source = load_opt(entry, "source_id", default=None)
                pivot_start = load_opt(entry, "pivot_start", default=None)
                if pivot_start is not None:
                    pivot_start = int(pivot_start)
                pivot_end = load_opt(entry, "pivot_end", default=None)
                if pivot_end is not None:
                    pivot_end = int(pivot_end)

                l_tokens = []
                if "tokens" in entry:
                    tokens = entry["tokens"]
                    for token in tokens:
                        form = token["form"]
                        lemma = token["lemma"]
                        pos = token["pos"]
                        if "upos" in token and token["upos"] is not None and token["upos"] != "None":
                            upos = token["upos"]
                        else:
                            # Do the pos to upos conversion, assuming STTS tagset
                            upos = pos_2_upos(pos)
                        begin = int(token["begin"])
                        end = int(token["end"])
                        l_tokens.append(WSDToken(form, lemma, pos, begin, end, upos=upos))
                        
                dataset.add_entry(
                        label, 
                        target_lemma, 
                        entry_pos,
                        sentence,
                        tokens=l_tokens,
                        source_id=source,
                        pivot_start=pivot_start,
                        pivot_end=pivot_end
                        )

            if correct_pivot_tokens:
                # Correct lemmatization/pos tagging errors from tokenizer by setting them to entry data
                for entry in dataset.entries:
                    lemma = entry.lemma
                    upos = entry.upos
                    tokens = entry.tokens
                    for token in tokens:
                        if token.is_pivot(entry):
                            token.lemma = lemma
                            token.upos = upos

            return dataset

    @classmethod
    def load_raganato(cls, xml_path: str, target_upos: Union[str, List[str]] = None, lang: str = "en",
                      input_keys: str = "sensekeys", name: str = None):
        assert xml_path.endswith(".data.xml"), "Must provide path to raganato xml"
        label_path = xml_path.replace('.data.xml', '.gold.key.txt')
        if name is None:
            name = os.path.basename(xml_path).replace(".data.xml", "")

        if isinstance(target_upos, str):  # Wrap single input with list for later check
            target_upos = [target_upos]

        dataset = WSDData(name=name, lang=lang, labeltype="wnoffsets")
        # Load in gold labels
        gold_labels = {}
        with open(label_path, "rt", encoding="utf8") as f:
            for line in f:
                line = line.strip().split(" ")
                instance_id = line[0]
                labels = line[1:]  # Currently ignore all labels other than last
                if input_keys == "wnoffsets":
                    gold_labels[instance_id] = labels[0]
                elif input_keys == "sensekeys":
                    # Convert sensekeys
                    offsets = [wnoffset_from_sense_key(label) for label in labels]
                    gold_labels[instance_id] = offsets[0]
                else:
                    raise NotImplementedError

        parser = et.XMLParser()
        xml_corpus = et.parse(xml_path, parser).getroot()
        # Go through each sentence
        for i, xml_text in enumerate(xml_corpus.getchildren()):
            for xml_sentence in xml_text.getchildren():
                if "id" in xml_sentence.attrib:
                    source = xml_sentence.attrib["id"]
                elif "source" in xml_text.attrib:
                    source = xml_text.attrib["source"]
                else:
                    source = name + "_" + str(i)
                # Create token list and find disambiguation instances
                tokens = []
                pivots = []
                char_position = 0
                for xml_token in xml_sentence.getchildren():
                    token_type = xml_token.tag

                    lemma = xml_token.attrib["lemma"]
                    upos = xml_token.attrib["pos"]
                    if upos == ".":
                        upos = "PUNCT"
                    form = xml_token.text
                    begin = char_position
                    end = char_position + len(form)
                    char_position += len(form) + 1
                    if token_type == "instance":
                        instance_id = xml_token.attrib["id"]
                        if (target_upos is None or upos in target_upos) and instance_id in gold_labels:
                            label = gold_labels[instance_id]
                            pivots.append((instance_id, label, lemma, upos, begin, end))
                    tokens.append(WSDToken(form, lemma, upos, begin, end, upos))
                # Create new entry for each instance
                sentence = " ".join([token.form for token in tokens])
                for pivot in pivots:
                    instance_id, label, lemma, upos, pivot_start, pivot_end = pivot
                    dataset.add_entry(label,
                                      lemma,
                                      upos,
                                      sentence,
                                      tokens=tokens,
                                      source_id=source,
                                      pivot_start=pivot_start,
                                      pivot_end=pivot_end)
        return dataset

    def save(self, outpath: str):
        out = jsonpickle.encode(self, unpicklable=False, indent=1)
        with open(outpath, "w+", encoding="utf8", newline="\n") as f:
            f.write(out)

    #def save_raganato(self, outpath: str):
    #    assert outpath.endswith(".data.xml")
    #    for
            
    def add(self, other: 'WSDData'):
        """ Merges the other dataset into this one. This can only be done if both have the same language"""
        assert self.lang == other.lang, "Can only merge two datasets of the same language!"
        assert self.labeltype == other.labeltype, "Can only merge to datasets with the same labeltype!"
        self.name = self.name + "+" + other.name
        for entry in other.entries:
            self.add_entry(**entry.get_dict())

    def intersection(self, other: 'WSDData'):
        """ Removes all entries from this dataset with source-ids not occuring in the other dataset """
        ids_other = set([entry.source_id for entry in other.entries])
        matching_entries = []
        for entry in self.entries:
            if entry.source_id in ids_other:
                matching_entries.append(entry)
        self.entries = matching_entries
        self._clean_sentences()

    @classmethod
    def split_on_sentences(cls, dataset: 'WSDData', sentence_limit: int):
        """ Splits the input dataset into smaller datasets with at most sentence_limit distinct sentences"""
        datasets = []
        n_sentences = len(dataset._sentence_cache.keys())
        entries_by_sentence = {}
        for entry in dataset.entries:
            if entry.sentence_idx in entries_by_sentence:
                entries_by_sentence[entry.sentence_idx].append(entry)
            else:
                entries_by_sentence[entry.sentence_idx] = [entry]

        n_splits = n_sentences // sentence_limit + 1
        split_size = n_sentences // n_splits

        idxs = list(dataset._sentences.keys())
        for i in range(n_splits-1):
            new = cls(dataset.name, dataset.lang, dataset.labeltype)
            for idx in idxs[i*split_size:(i+1)*split_size]:
                for entry in entries_by_sentence[idx]:
                    new.add_entry(**entry.get_dict())
            datasets.append(new)
        last_split = cls(dataset.name, dataset.lang, dataset.labeltype)
        for idx in idxs[(n_splits - 1) * split_size:]:
            for entry in entries_by_sentence[idx]:
                last_split.add_entry(**entry.get_dict())
        datasets.append(last_split)

        return datasets

    def sentence_count(self):
        return len(self._sentence_cache)

    def map_labels(self, mapping_dict, new_labeltype: str, no_map="skip", in_place: bool = True):
        # TODO: in_place. Need to deep copy entries and tokens to avoid list copy type bugs, don't want to do that
        #  though because of memory. May not actually be a concern. Don't need this feature anymore either though.
        if not in_place:
            raise NotImplementedError
        mapped_entries = []
        assert new_labeltype in VALID_LABELTYPES
        for entry in self.entries:
            if entry.label in mapping_dict:
                entry.label = mapping_dict[entry.label]
                mapped_entries.append(entry)
            else:
                if no_map == "skip":
                    continue
                elif no_map == "raise":
                    raise RuntimeWarning("No mapping for entries with label {}".format(entry.label))
                elif no_map == "warn":
                    print("No mapping for entries with label {}".format(entry.label))
                    continue
                else:
                    raise NotImplementedError
        self.entries = mapped_entries
        self.labeltype = new_labeltype

    def filter(self, ambiguous: bool = False, label_count_limit: int = None, in_place: bool = True):
        filtered = self.entries
        # TODO: in_place. Need to deep copy entries and tokens to avoid list copy type bugs, don't want to do that
        #  though because of memory
        if not in_place:
            raise NotImplementedError
        # Filter out all labels with a count below the limit
        if label_count_limit:
            assert label_count_limit >= 0
            label_counts = {}
            for entry in self.entries:
                if entry.label in label_counts:
                    label_counts[entry.label] += 1
                else:
                    label_counts[entry.label] = 1
            tmp = [entry for entry in filtered if label_counts[entry.label] >= label_count_limit]
            filtered = tmp

        # Filter out lemmas which are not ambiguous in dataset
        if ambiguous:
            lemma_sense_map = {}
            for entry in filtered:
                key = entry.lemma + "#" + entry.upos
                if key in lemma_sense_map:
                    lemma_sense_map[key].add(entry.label)
                else:
                    lemma_sense_map[key] = {entry.label}
            tmp = [entry for entry in filtered if len(lemma_sense_map[entry.lemma + "#" + entry.upos]) > 1]
            filtered = tmp
        self.entries = filtered
        self._clean_sentences()

    def get_lemmas(self):
        lemma_frequencies = {}
        for entry in self.entries:
            if entry.lemma in lemma_frequencies:
                lemma_frequencies[entry.lemma] += 1
            else:
                lemma_frequencies[entry.lemma] = 1
        return lemma_frequencies

    def get_statistics(self, pprint=True):
        lemma_sense_map = {}
        lemma_sense_dist = {}
        lemma_dist = {}
        word_sense_dist = {}
        labels = set()
        statistics_dict = {}

        for entry in self.entries:
            key = entry.lemma + "#" + entry.upos
            labels.add(entry.label)
            if key in lemma_sense_map:
                lemma_sense_map[key].add(entry.label)
            else:
                lemma_sense_map[key] = {entry.label}
            if key in lemma_dist:
                lemma_dist[key] += 1
            else:
                lemma_dist[key] = 1

        for key in lemma_sense_map:
            count = len(lemma_sense_map[key])
            if count in lemma_sense_dist:
                lemma_sense_dist[count] += 1
            else:
                lemma_sense_dist[count] = 1
            if count in word_sense_dist:
                word_sense_dist[count] += lemma_dist[key]
            else:
                word_sense_dist[count] = lemma_dist[key]

        lemma_min = math.inf
        lemma_max = - math.inf
        for key in lemma_dist:
            if lemma_dist[key] < lemma_min:
                lemma_min = lemma_dist[key]
            if lemma_dist[key] > lemma_max:
                lemma_max = lemma_dist[key]

        average_word_polysemy = 0
        for key in lemma_dist:
            average_word_polysemy += lemma_dist[key] * len(lemma_sense_map[key])
        average_word_polysemy = average_word_polysemy / len(self.entries)

        average_lemma_polysemy = 0
        for i in lemma_sense_dist:
            average_lemma_polysemy += i * lemma_sense_dist[i]
        average_lemma_polysemy = average_lemma_polysemy / len(lemma_sense_map)

        statistics_dict["Number of Instances"] = len(self.entries)
        statistics_dict["Distinct senses"] = len(labels)
        statistics_dict["Distinct lemmas"] = len(lemma_sense_map)
        statistics_dict["Lemma polysemy distribution"] = lemma_sense_dist
        statistics_dict["Average lemma polysemy"] = average_lemma_polysemy
        statistics_dict["Word Polysemy distribution"] = word_sense_dist
        statistics_dict["Average word polysemy"] = average_word_polysemy
        statistics_dict["Lemma distribution"] = lemma_dist

        if pprint:
            print("Instances: {}".format(len(self.entries)))
            print("Distinct senses: {}".format(len(labels)))
            print("Distinct lemmas: {}".format(len(lemma_sense_map)))
            print("Average lemma polysemy: {:.2f}".format(average_lemma_polysemy))
            print("Average word polysemy: {:.2f}".format(average_word_polysemy))
            print("Lemma frequency range: {} - {}".format(lemma_min, lemma_max))
            print("Distribution of lemmas with x senses:")
            for i in sorted(lemma_sense_dist):
                print("{} lemmas with {} senses".format(lemma_sense_dist[i], i))
            for i in sorted(word_sense_dist):
                print("{} words with {} senses".format(word_sense_dist[i], i))

        return statistics_dict

    def mfs(self):
        per_lemma_sense_counts = {}
        # Count sense occurences for each lemma
        for entry in self.entries:
            if entry.lemma in per_lemma_sense_counts:
                if entry.label in per_lemma_sense_counts[entry.lemma]:
                    per_lemma_sense_counts[entry.lemma][entry.label] += 1
                else:
                    per_lemma_sense_counts[entry.lemma][entry.label] = 1
            else:
                per_lemma_sense_counts[entry.lemma] = {entry.label: 1}

        # Get most frequent sense
        most_frequent_senses = {}
        for lemma in per_lemma_sense_counts:
            mfs = None
            count = -1
            for sense in per_lemma_sense_counts[lemma]:
                if per_lemma_sense_counts[lemma][sense] > count:
                    mfs = sense
                    count = per_lemma_sense_counts[lemma][sense]
            most_frequent_senses[lemma] = mfs

        return most_frequent_senses


def load_opt(entry, key: str, default=None):
    if key in entry:
        return entry[key]
    else:
        return default


def load_mapping(map_path: str, first_only=True):
    map_dict = {}
    with open(map_path, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            key = line[0]
            if first_only:
                value = line[1]
            else:
                value = line[1:]
            map_dict[key] = value
    return map_dict         


def train_test_split(dataset: WSDData, ratio_eval=0.2, ratio_test=0.2, shuffle=True, lemma_key=False):
    # Split dataset into train/eval/test datasets with stratification using the gold labels
    assert ratio_eval + ratio_test <= 1.0
    assert ratio_eval >= 0.0
    assert ratio_test >= 0.0

    entries_by_label = {}

    for entry in dataset.entries:
        if lemma_key:
            key = (entry.lemma, entry.label)
        else:
            key = entry.label
        if key in entries_by_label:
            entries_by_label[key].append(entry)
        else:
            entries_by_label[key] = [entry]

    trainset = WSDData(dataset.name + "_train", dataset.lang, dataset.labeltype)
    evalset = WSDData(dataset.name + "_eval", dataset.lang, dataset.labeltype)
    testset = WSDData(dataset.name + "_test", dataset.lang, dataset.labeltype)

    for key, entries in entries_by_label.items():
        # Dump labels with single instance
        if len(entries) <= 1:
            continue

        # Fix sizes for low count labels to ensure we have at least one in train/eval/test if at all possible
        elif len(entries) == 2:
            eval_size = 0
            test_size = 1
        else:
            eval_size = math.floor(len(entries)*ratio_eval)
            test_size = math.floor(len(entries) * ratio_test)
            if test_size == 0 and ratio_test > 0.0:
                test_size = 1
            if eval_size == 0 and ratio_eval > 0.0:
                eval_size = 1
            train_size = len(entries) - eval_size - test_size

        if shuffle:
            random.shuffle(entries)

        for entry in entries[:eval_size]:
            evalset.add_entry(**entry.get_dict())
        for entry in entries[eval_size:eval_size+test_size]:
            testset.add_entry(**entry.get_dict())
        for entry in entries[eval_size+test_size:]:
            trainset.add_entry(**entry.get_dict())

    if shuffle:
        random.shuffle(trainset.entries)
        random.shuffle(evalset.entries)
        random.shuffle(testset.entries)
    return trainset, evalset, testset


def pos_2_upos(pos: str):
    STTS = {"$(": "PUNCT",
            "$,": "PUNCT",
            "$.": "PUNCT",
            "ADJA": "ADJ",
            "ADJD": "ADJ",
            "ADV": "ADV",
            "APPO": "ADP",
            "APPR": "ADP",
            "APPRART": "ADP",
            "APZR": "ADP",
            "ART": "DET",
            "CARD": "NUM",
            "FM": "X",
            "ITJ": "INTJ",
            "KOKOM": "CCONJ",
            "KON": "CCONJ",
            "KOUI": "SCONJ",
            "KOUS": "SCONJ",
            "NE": "PROPN",
            "NN": "NOUN",
            "PAV": "ADV",
            "PDAT": "DET",
            "PDS": "PRON",
            "PIAT": "DET",
            "PIDAT": "DET",
            "PIS": "PRON",
            "PPER": "PRON",
            "PPOSAT": "DET",
            "PPOSS": "PRON",
            "PRELAT": "DET",
            "PRELS": "PRON",
            "PRF": "PRON",
            "PROAV": "ADV",
            "PTKA": "PART",
            "PTKANT": "PART",
            "PTKNEG": "PART",
            "PTKVZ": "ADP",
            "PTKZU": "PART",
            "PWAT": "DET",
            "PWAV": "ADV",
            "PWS": "PRON",
            "TRUNC": "X",
            "VAFIN": "AUX",
            "VAIMP": "AUX",
            "VAINF": "AUX",
            "VAPP": "AUX",
            "VMFIN": "VERB",
            "VMINF": "VERB",
            "VMPP": "VERB",
            "VVFIN": "VERB",
            "VVIMP": "VERB",
            "VVINF": "VERB",
            "VVIZU": "VERB",
            "VVPP": "VERB",
            "XY": "X"
            }
    return STTS[pos]


def wnoffset_from_sense_key(sense_key: str):
    syn = wn.synset_from_sense_key(sense_key)
    offset = "wn:" + "{:08d}".format(syn.offset()) + syn.pos()
    return offset


def cli():
    parser = argparse.ArgumentParser(description="Dataset utility script for splitting datasets into train/eval/test or"
                                                 " converting labels according to some mapping")
    subparsers = parser.add_subparsers(help="Either 'split' to split a dataset into train/test eval "
                                            "or 'convert' to convert dataset labels",
                                       dest="action", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("-d", "--data", required=True, type=str,
                              help="JSON Datafile that will be split into train/eval/test.")
    split_parser.add_argument("-re", "--ratio-eval", required=True, type=float,
                              help="ratio of the datasets that will be used as evaluation data")
    split_parser.add_argument("-rt", "--ratio-test", required=True, type=float,
                              help="ratio of the datasets that will be used as test data")
    split_parser.add_argument("-o", "--out", required=True, type=str,
                              help="Output directory. Three files will be created with train/eval/test suffixes.")
    split_parser.add_argument("-s", "--shuffle", required=False, action="store_true",
                              help="If this flag is set the dataset is shuffled before splitting.")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("-d", "--data", required=True, type=str,
                                help="JSON Datafile that will be split into train/eval/test.")
    convert_parser.add_argument("-m", "--mapping", required=True, type=str,
                                help="Mapping file which will be used for converting labels.")
    convert_parser.add_argument("-n", "--new-labeltype", required=True, type=str,
                                help="New labeltype after the mapping. Must be one of {}".format(VALID_LABELTYPES))
    convert_parser.add_argument("-o", "--out", required=True, type=str,
                                help="Output path")
    convert_parser.add_argument("-s", "--skip", required=False, action="store_true",
                                help="If this flag is set, entries which cannot be mapped will be skipped")

    args = parser.parse_args()
    
    dataset = WSDData.load(os.path.abspath(args.data))

    if args.action == "split":
        basename = os.path.splitext(os.path.basename(args.data))[0]
        trainset, evalset, testset = train_test_split(dataset, args.ratio_eval, args.ratio_test, shuffle=args.shuffle)
        trainset.save(os.path.abspath(os.path.join(args.out, basename + "_train.json")))
        evalset.save(os.path.abspath(os.path.join(args.out, basename + "_eval.json")))
        testset.save(os.path.abspath(os.path.join(args.out, basename + "_test.json")))
    elif args.action == "convert":
        mapping = load_mapping(args.mapping, first_only=True)
        if args.skip:
            no_map = "skip"
        else:
            no_map = "raise"
        dataset.map_labels(mapping, args.new_labeltype, no_map=no_map)
        dataset.save(args.out)


if __name__ == "__main__":
    cli()
