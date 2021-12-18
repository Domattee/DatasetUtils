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

from wsdUtils.dataset import WSDData, WSDEntry, WSDToken

from typing import List


class AnnotEntry(WSDEntry):

    SPECIAL_LABELS = ["-1", "-2"]

    def __init__(self, dataset: "AnnotData", sentence_idx: int, label: str, lemma: str, upos: str,
                 raw_labels: List[str] = [], annotators: List[str] = [], source_id: str = None,
                 pivot_start: int = None, pivot_end: int = None):
        super().__init__(dataset, sentence_idx, label, lemma, upos,
                         source_id=source_id, pivot_start=pivot_start, pivot_end=pivot_end)

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

    def add_entry(self, label: str, lemma: str, upos: str, sentence: str, tokens: List[WSDToken] = None,
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

    # =========== Merging related functions ========================================================================
    # ==============================================================================================================

    def merge_labels(self, label1, label2):
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

    # =========== Processing entries without gold ==================================================================
    # ==============================================================================================================

    def process_easy_entries(self):
        # Easy entries are those where all manual annotations are identical, excluding special labels
        for entry in self.unprocessed_entries():
            labels = set()
            for label in entry.raw_labels:
                if label in self.merge_map:
                    labels.add(self.merge_map[label])
                else:
                    labels.add(label)
            if len(labels) == 1:
                label = list(labels)[0]
                entry.label = label

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
        pass

    # =========== I/O ==============================================================================================
    # ==============================================================================================================

    def _load_db_dump(self, filepath):
        pass

    def load(self, outpath):
        pass
