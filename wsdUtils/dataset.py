import math
import json
import jsonpickle
import argparse
import os
import random
from typing import List

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
# TODO: tokenizing
# TODO: Tokenizer
# TODO: Eval other non-java tokenizers,
#  maybe nltk german tokenizer/pos-taggers are fine? Would make process much simpler.

VALID_LABELTYPES = ["wnoffsets", "bnids", "gn"]


class WSDToken:
    def __init__(self, form: str, lemma: str, pos: str, begin: int, end: int, upos: str = None, is_pivot: bool = False):
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.upos = upos
        self.begin = begin
        self.end = end
        self.is_pivot = is_pivot

    def __str__(self):
        rep_items = [self.form, self.lemma, self.pos, self.upos, self.begin, self.end, self.is_pivot]
        return "\t".join(map(lambda x: str(x), rep_items))
        
        
class WSDEntry:
    def __init__(self, label: str, lemma: str, upos: str, tokens: List[WSDToken] = None,
                 sentence: str = None, source_id: str = None, pivot_start: int = None, pivot_end: int = None):
        if tokens is None:
            tokens = []
        self.label = label
        self.lemma = lemma
        self.tokens = tokens
        self.sentence = sentence
        self.upos = upos
        self.source_id = source_id
        self.pivot_start = pivot_start
        self.pivot_end = pivot_end

    def __str__(self):
        rep_items = [self.label, self.lemma, self.pivot_start, self.pivot_end, self.source_id,
                     "{} tokens".format(len(self.tokens))]
        return "\t".join(map(lambda x: str(x), rep_items))
        
        
class WSDData:
    def __init__(self, name: str, lang: str, labeltype: str, entries: List[WSDEntry] = None):
        assert labeltype in VALID_LABELTYPES
        if entries is None:
            entries = []
        self.name = name
        self.entries = entries
        self.lang = lang
        self.labeltype = labeltype

    def __str__(self):
        return "Dataset {}, language {}, labels {} with {} entries".format(self.name,
                                                                           self.lang, self.labeltype, len(self.entries))

    @classmethod
    def _load_opt(cls, entry, key: str, default=None):
        if key in entry:
            return entry[key]
        else:
            return default
        
    @classmethod
    def load(cls, json_path: str):
        with open(json_path, "r", encoding="utf8") as f:
            loaded = json.load(f)
            lang = loaded["lang"]
            name = loaded["name"]
            labeltype = loaded["labeltype"]
            entries = []
            for entry in loaded["entries"]:
                label = entry["label"]
                target_lemma = entry["lemma"]
                entry_pos = entry["upos"]
                
                sentence = cls._load_opt(entry, "sentence", default=None)
                source = cls._load_opt(entry, "source_id", default=None)
                pivot_start = cls._load_opt(entry, "pivot_start", default=None)
                if pivot_start is not None:
                    pivot_start = int(pivot_start)
                pivot_end = cls._load_opt(entry, "pivot_end", default=None)
                if pivot_end is not None:
                    pivot_end = int(pivot_end)

                l_tokens = []
                if "tokens" in entry:
                    tokens = entry["tokens"]
                    for token in tokens:
                        form = token["form"]
                        lemma = token["lemma"]
                        pos = token["pos"]
                        if "upos" in token:
                            upos = token["upos"]
                        else:
                            # Do the pos to upos conversion, assuming STTS tagset
                            upos = pos_2_upos(pos)
                        begin = int(token["begin"])
                        end = int(token["end"])
                        is_pivot = bool(token["is_pivot"])
                        l_tokens.append(WSDToken(form, lemma, pos, begin, end, upos=upos, is_pivot=is_pivot))
                        
                entries.append(
                    WSDEntry(
                        label, 
                        target_lemma, 
                        entry_pos, 
                        tokens=l_tokens, 
                        sentence=sentence, 
                        source_id=source,
                        pivot_start=pivot_start,
                        pivot_end=pivot_end
                        ))
            return cls(name, lang, labeltype, entries)
                
    def save(self, outpath: str):
        out = jsonpickle.encode(self, unpicklable=False, indent=2)
        with open(outpath, "w+", encoding="utf8") as f:
            f.write(out)
            
    def add(self, other):
        """ Merges the other dataset into this one. This can only be done if both have the same language"""
        assert self.lang == other.lang, "Can only merge two datasets of the same language!"
        assert self.labeltype == other.labeltype, "Can only merge to datasets with the same labeltype!"
        self.name = self.name + "+" + other.name
        self.entries.extend(other.entries)
        
    def map_labels(self, mapping_dict, new_labeltype: str, no_map="skip"):
        # TODO: What to do if we have multiple values for keys in dict?
        mapped_entries = []
        for entry in self.entries:
            if entry.label in mapping_dict:
                entry.label = mapping_dict[entry.label]
                mapped_entries.append(entry)
            else:
                if no_map == "skip":
                    continue
                elif no_map == "raise":
                    raise RuntimeWarning("No mapping for entries with label {}".format(entry.label)) 
        self.entries = mapped_entries
        self.labeltype = new_labeltype

    def filter(self, ambiguous: bool = False, label_count_limit: int = None):
        filtered = self.entries

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

        # Apply filter to current dataset
        self.entries = filtered

    def get_statistics(self):
        lemma_sense_map = {}
        sense_dist = {}
        labels = set()

        for entry in self.entries:
            key = entry.lemma + "#" + entry.upos
            labels.add(entry.label)
            if key in lemma_sense_map:
                lemma_sense_map[key].add(entry.label)
            else:
                lemma_sense_map[key] = {entry.label}

        for key in lemma_sense_map:
            count = len(lemma_sense_map[key])
            if count in sense_dist:
                sense_dist[count] += 1
            else:
                sense_dist[count] = 1
        print("Instances: {}".format(len(self.entries)))
        print("Distinct senses: {}".format(len(labels)))
        print("Distinct lemmas: {}".format(len(lemma_sense_map)))
        print("Distribution of lemmas with x senses:")
        for i in sorted(sense_dist):
            print("{} lemmas with {} senses".format(sense_dist[i], i))

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


def train_test_split(dataset: WSDData, ratio_eval=0.2, ratio_test=0.2, shuffle=True):
    # Split dataset into train/eval/test datasets with stratification using the gold labels
    assert ratio_eval + ratio_test <= 1.0
    assert ratio_eval >= 0.0
    assert ratio_test >= 0.0

    entries_by_label = {}

    for entry in dataset.entries:
        label = entry.label
        if label in entries_by_label:
            entries_by_label[label].append(entry)
        else:
            entries_by_label[label] = [entry]

    trainset = WSDData(dataset.name + "_train", dataset.lang, dataset.labeltype, entries=[])
    evalset = WSDData(dataset.name + "_eval", dataset.lang, dataset.labeltype, entries=[])
    testset = WSDData(dataset.name + "_test", dataset.lang, dataset.labeltype, entries=[])

    for label, entries in entries_by_label.items():
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
        evalset.entries.extend(entries[:eval_size])
        testset.entries.extend(entries[eval_size:eval_size+test_size])
        trainset.entries.extend(entries[eval_size+test_size:])

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
