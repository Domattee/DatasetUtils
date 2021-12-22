# Datastructure based on wsdUtils.WSDData WSDEntry
# TODO: Compute IAA scores with randolphs kappa -> Special value -2 (skip) should be ignored
#       Score 1: IAA including "no valid sense"
#       Score 2: IAA excluding "no valid sense"
#       IAA Matrix for each annotator
# TODO: Functions to export Dataset in annotation format:
#       sentence id / lemma / pivot start / pivot end / gold annotation / list of annotations from each annotator
# Functions to manage gold labels in dataset:
#       Map senses using merge map. Should track merged labels and why they were merged
#       TODO: Sentences with only skips or no valid sense -> skip for WSD format, still output in annotation format
#       TODO: Sentences with two differing labels -> Review these somehow and manage labels:
#           Merge labels if too similar/one is metaphorical use of other -> Add appropriate merge map entries
#           Merging should happen consistently, i.e. if one of the two senses was already merged or is the target of a
#             merge we should merge to that one.
#           If senses do not get merged -> Pick a "correct" one for gold label, or skip if can't decide
#       Sentences with single label or two identical labels -> Gold label is that label.
# TODO: Make lemma id file for all germanet verbs

from wsdUtils.dataset import WSDData, WSDEntry, WSDToken

from typing import List, Union
import json


class AnnotEntry(WSDEntry):

    SPECIAL_LABELS = ["-1", "-2"]

    def __init__(self, dataset: "AnnotData", sentence_idx: int, label: str, lemma: str, upos: str,
                 raw_labels: List[str] = [], annotators: List[str] = [], source_id: str = None,
                 pivot_start: int = None, pivot_end: int = None):
        super().__init__(dataset, sentence_idx, label, lemma, upos,
                         source_id=source_id, pivot_start=pivot_start, pivot_end=pivot_end)
        if raw_labels is not None or annotators is not None:
            assert len(raw_labels) == len(annotators)
        self.raw_labels = raw_labels
        self.annotators = annotators

    @property
    def can_be_labeled(self):
        return any([label not in AnnotEntry.SPECIAL_LABELS for label in self.raw_labels])


class AnnotData(WSDData):

    def __init__(self, name: str, lang: str, labeltype: str):
        super().__init__(name, lang, labeltype)
        self.entries: List[AnnotEntry] = []
        self.merge_map = {}
        self.merge_targets = set()
        self.merge_reasons = {}
        self.lemma_ids = {}
        self.load_lemma_labels("")

    def add_entry(self, label: Union[str, None], lemma: str, upos: str, sentence: str, tokens: List[WSDToken] = None,
                  raw_labels=[], annotators=[], source_id: str = None, pivot_start: int = None, pivot_end: int = None):
        idx = self._add_sentence(sentence, tokens)
        self.entries.append(AnnotEntry(self, idx, label, lemma, upos, raw_labels=raw_labels, annotators=annotators,
                                       source_id=source_id, pivot_start=pivot_start, pivot_end=pivot_end))

    def unprocessed_entries(self):
        out = []
        for entry in self.entries:
            if entry.label is None and entry.can_be_labeled:
                out.append(entry)
        return out

    def anonymize(self):
        """ Replaces all annotators with numbers and generates a dictionary containing the replacement mapping """
        annotator_labels = {}
        annotator_count = 1
        for entry in self.entries:
            annotators = entry.annotators
            anonymized = []
            for annotator in annotators:
                if annotator in annotator_labels:
                    anonymized.append(annotator_labels[annotator])
                else:
                    annotator_label = "A_{}".format(annotator_count)
                    anonymized.append(annotator_label)
                    annotator_labels[annotator] = annotator_label
                    annotator_count += 1
            entry.annotators = anonymized
        with open("annotator_names.txt", "wt", encoding="utf8", newline="") as f:
            for annotator_name in annotator_labels:
                f.write(annotator_name + "\t" + annotator_labels[annotator_name] + "\n")

    # =========== Merging related functions ========================================================================
    # ==============================================================================================================

    def _merge_labels(self, label1, label2):
        # Handles the bureaucracy of selecting two labels for merging
        # If both label1 and label2 were already used as merge targets:
        #   Chose label1 as target
        #   Go over merge map and replace all label2 targets with label1
        #   Remove label2 from merge targets
        # If only label1 was already used, proceed as above but without the complicated check
        # If only label2 was already used, choose label 2 and proceed as above, but without the complicated check
        # If neither label was used, choose label1
        if label1 in self.merge_targets or label2 not in self.merge_targets:
            target_label = label1
            mapped_label = label2
            if label2 in self.merge_targets:
                for key in self.merge_map:
                    if self.merge_map[key] == label2:
                        self.merge_map[key] = label1
                self.merge_targets.remove(label2)
        else:
            target_label = label2
            mapped_label = label1
        self.merge_targets.add(target_label)
        self.merge_map[mapped_label] = target_label
        return target_label

    def _remap_labels(self):
        for entry in self.entries:
            if entry.label in self.merge_map:
                entry.label = self.merge_map[entry.label]

    def write_merge_map(self, outfile):
        inverted_map = {}
        for mapped in self.merge_map:
            target = self.merge_map[mapped]
            if target in inverted_map:
                inverted_map[target].add(mapped)
            else:
                inverted_map[target] = {mapped}
        with open(outfile, "wt", encoding="utf8", newline="") as f:
            for key in inverted_map:
                outline = key + "\t" + "\t".join(inverted_map[key]) + "\n"
                f.write(outline)

    def load_merge_map(self, infile):
        with open(infile, "rt", encoding="utf8") as f:
            for line in f:
                line = line.strip().split("\t")
                target = line[0]
                mapped = line[1:]
                for tmp in mapped:
                    self.merge_map[tmp] = target

    # =========== Processing entries without gold ==================================================================
    # ==============================================================================================================

    def process_easy_entries(self):
        # Easy entries are those where all manual annotations are identical, excluding special labels
        counter = 0
        for entry in self.unprocessed_entries():
            labels = set()
            for label in entry.raw_labels:
                if label in self.merge_map:
                    labels.add(self.merge_map[label])
                else:
                    labels.add(label)
            if len(labels) == 1:
                counter += 1
                label = list(labels)[0]
                entry.label = label
        print("Filled in {} labels".format(counter))

    def process_hard_entries(self):
        # While there are unprocessed entries with differing annotations:
        # Display the sentence, the two annotations and info on the two senses
        # Ask user on what to do:
        #   1. Pick gold sense      -> Add label to entry
        #   2. Merge senses         -> Do merge process and add target label to entry
        #   3. Skip                 -> ???
        # If we merged -> assign new easy labels and remap entries
        # Continue

        # Final step: Go over all entries and map them according to merge map to ensure consistency
        #       Write out merge information in file
        pass

    # =========== I/O ==============================================================================================
    # ==============================================================================================================

    @classmethod
    def _load_db_dump(cls, filepath, name, labeltype, lang, upos, anonymize=True):
        """ Load mongo dump. If anonymize is true we replace actual annotator names with numbers and write out a file
        with the labeling """
        with open(filepath, "r", encoding="utf8") as f:
            dataset = cls(name=name, labeltype=labeltype, lang=lang)
            loaded = json.load(f)
            for entry in loaded:
                source_id = entry["sentence_id"]
                sentence = entry["sentence"]
                entry_lemma = entry["verb_clean"]
                pivot_start = int(entry["pivot_start"])
                pivot_end = int(entry["pivot_end"])

                annotations = []
                annotators = []
                for annotation in entry["annotations"]:
                    annotations.append(annotation["annotation"])
                    annotators.append(annotation["annotator"])

                dataset.add_entry(label=None, lemma=entry_lemma, upos=upos, sentence=sentence, raw_labels=annotations,
                                  annotators=annotators, source_id=source_id, pivot_start=pivot_start,
                                  pivot_end=pivot_end)

            if anonymize:
                dataset.anonymize()

    @classmethod
    def _load_verb_db_dump(cls, filepath, anonymize=True):
        return cls._load_db_dump(filepath, "ttvc_2", "gn", "de", "VERB", anonymize=anonymize)

    def load(self, outpath):
        pass

    def load_lemma_labels(self, inpath):
        with open(inpath, "rt", encoding="utf8") as f:
            for line in f:
                line = line.strip().split("\t")
                lemmapos = line[0]
                labels = line[1:]
                self.lemma_ids[lemmapos] = labels

    # =========== Metrics and Related ==============================================================================
    # ==============================================================================================================

    def iaa(self, with_merging=True):
        """ Randolphs Kappa for whole set"""

        annotations = []
        total_labels = set()
        for entry in self.entries:
            if entry.raw_labels is None:
                continue
            total_labels.update(self._count_possible_labels(entry, with_merging=with_merging))
            annotations.append(self._count_labels(entry, with_merging=with_merging))
        score = _randolphs_kappa(annotations, len(total_labels))
        # TODO: Check that this makes any sense at all? Seems as though we should calculate kappa for each instance or
        #  each lemma and then average them or something. p_e is effectively 0 due to very large number of total labels
        #  currently.
        return score

    def iaa_lemma(self, lemma: str, with_merging=True):
        """ Randolphs Kappa for specific lemma"""
        annotations = []
        total_labels = set()
        for entry in self.entries:
            if entry.lemma == lemma and entry.raw_labels is not None:
                total_labels.update(self._count_possible_labels(entry, with_merging=with_merging))
                annotations.append(self._count_labels(entry, with_merging=with_merging))
        score = _randolphs_kappa(annotations, len(total_labels))
        return score

    def _count_labels(self, entry, with_merging=True):
        label_counts = {}
        for label in entry.raw_labels:
            if with_merging and label in self.merge_map:
                label = self.merge_map[label]
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        counts = [value for key, value in label_counts.items()]
        return counts

    def _count_possible_labels(self, entry, with_merging=True):
        entry_labels = set()
        lemmapos = entry.lemma + "#" + entry.upos
        if lemmapos not in self.lemma_ids:
            raise RuntimeError("Could not find lemmapos {} in mapping!".format(lemmapos))
        else:
            for label in self.lemma_ids[lemmapos]:
                if with_merging and label in self.merge_map:
                    entry_labels.add(self.merge_map[label])
                else:
                    entry_labels.add(label)
        return entry_labels


def _randolphs_kappa(annotations, n_labels):
    """ Computes randolphs kappa for a given annotation matrix. Annotation matrix stores the number of annotations per
    category for each instance. Categories do not have to be consistent between instances.
    Assumes a rater can only rate each item once
    Pythonized version of the DKPro implementation"""
    p_e = 1.0 / n_labels

    nom = 0.0
    denom = 0.0
    for instance in annotations:
        rater_count = sum(instance)
        if rater_count < 2:
            continue
        nom += sum(count * (count + 1) for count in instance) / (rater_count - 1)
        denom += rater_count
    p_o = nom / denom
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa
