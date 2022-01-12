# Datastructure based on wsdUtils.WSDData WSDEntry
# Compute IAA scores with randolphs kappa -> Special value -2 (skip) should be ignored
#       IAA Matrix for each annotator pair
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

from wsdUtils.dataset import WSDData, WSDEntry, WSDToken, LABEL_PREFIXES

from typing import List, Union
import json
import os


LEMMA_ID_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "lemmaIds.txt"))


class AnnotEntry(WSDEntry):

    SPECIAL_LABELS = ["gn:-1", "gn:-2"]

    def __init__(self, dataset: "AnnotData", sentence_idx: int, label: str, lemma: str, upos: str,
                 raw_labels: List[str] = [], annotators: List[str] = [], source_id: str = None,
                 pivot_start: int = None, pivot_end: int = None):
        super().__init__(dataset, sentence_idx, label, lemma, upos,
                         source_id=source_id, pivot_start=pivot_start, pivot_end=pivot_end)
        if annotators is None:
            annotators = []
        if raw_labels is None:
            raw_labels = []
        assert len(raw_labels) == len(annotators)
        self.raw_labels = raw_labels
        self.annotators = annotators

    @property
    def can_be_labeled(self):
        return any([label not in AnnotEntry.SPECIAL_LABELS for label in self.raw_labels])


class AnnotData(WSDData):

    def __init__(self, name: str, lang: str, labeltype: str, load_lemma_ids=False):
        super().__init__(name, lang, labeltype)
        self.entries: List[AnnotEntry] = []
        self.merge_map = {}
        self.merge_targets = set()
        self.merge_reasons = {}
        self.lemma_ids = {}
        if load_lemma_ids:
            self.load_lemma_labels(LEMMA_ID_PATH)

    def add_entry(self, label: Union[str, None], lemma: str, upos: str, sentence: str, tokens: List[WSDToken] = None,
                  raw_labels: List[str] = [], annotators: List[str] = [], source_id: str = None,
                  pivot_start: int = None, pivot_end: int = None):
        idx = self._add_sentence(sentence, tokens)
        self.entries.append(AnnotEntry(self, idx, label, lemma, upos, raw_labels=raw_labels, annotators=annotators,
                                       source_id=source_id, pivot_start=pivot_start, pivot_end=pivot_end))

    def unprocessed_entries(self, include_without_valid=False):
        out = []
        for entry in self.entries:
            if entry.label is None and (include_without_valid or (not include_without_valid and entry.can_be_labeled)):
                out.append(entry)
        return out

    def annotators(self):
        annotators = set()
        for entry in self.entries:
            annotators.update(entry.annotators)
        return sorted(list(annotators))

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

    def _merge_labels(self, label1, label2, reason: str = "", force_label1=False):
        # Handles the bureaucracy of selecting two labels for merging
        # If both label1 and label2 were already used as merge targets:
        #   Chose label1 as target
        #   Go over merge map and replace all label2 targets with label1
        #   Remove label2 from merge targets
        # If only label1 was already used, proceed as above but without the complicated check
        # If only label2 was already used, choose label 2 and proceed as above, but without the complicated check
        # If neither label was used, choose label1
        # If force_label1 is set we always choose label1 as merge target
        if label1 in self.merge_targets or label2 not in self.merge_targets or force_label1:
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
        self.merge_reasons[mapped_label] = reason
        return mapped_label, target_label

    def _remap_labels(self):
        for entry in self.entries:
            if entry.label in self.merge_map:
                entry.label = self.merge_map[entry.label]

    def _remap_raw_labels(self, entry):
        labels = []
        for label in entry.raw_labels:
            if label in self.merge_map:
                labels.append(self.merge_map[label])
            else:
                labels.append(label)
        return labels

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
            labels = self._remap_raw_labels(entry)
            if len(set(labels)) == 1:
                counter += 1
                label = labels[0]
                entry.label = label
        print("Filled in {} labels".format(counter))

    def process_hard_entries(self):
        # While there are unprocessed entries with differing annotations:
        # Display the sentence, the two annotations and info on the corresponding senses
        # Ask user on what to do:
        #   1. Pick gold sense      -> Add label to entry
        #   2. Merge senses         -> Do merge process and add target label to entry, also ask for reason?
        #   3. Skip                 -> Get next, do later
        #   4. Stop processing, continue later
        # If we merged -> assign new easy labels and remap entries
        # Continue with next entry

        # Final step: Go over all entries and map them according to merge map to ensure consistency

        # sort entries by label first
        lemma_entries = {}
        for entry in self.unprocessed_entries(include_without_valid=True):
            if entry.lemma in lemma_entries:
                lemma_entries[entry.lemma].append(entry)
            else:
                lemma_entries[entry.lemma] = [entry]
        sorted_entries = []
        for key in lemma_entries:
            sorted_entries.extend(lemma_entries[key])

        counter = 0
        for entry in sorted_entries:
            # Get raw labels
            labels = set(self._remap_raw_labels(entry))
            if len(labels) < 2:  # Boring entry, nothing to do
                continue
            # Do the whole thing
            option, value = self._select_how_to_handle_entry(entry)
            if option == "gold":
                entry.label = value
            elif option == "merge":
                mapped, target = self._merge_labels(*labels, reason=value)
                entry.label = target
            elif option == "skip":
                continue
            else:
                break
            counter += 1
        print("Manually processed {} entries".format(counter))
        self.process_easy_entries()
        self._remap_labels()

    def _show_label_info(self, labels):
        # Can't be bothered, just look it up on web api
        pass

    def _select_how_to_handle_entry(self, entry):
        labels = list(self._remap_raw_labels(entry))
        options = []
        for label in labels:
            options.append(label)

        options.append("Merge these senses")
        merge_option = len(options)
        options.append("Insufficient context")
        missing_context = len(options)
        options.append("No valid sense")
        no_sense = len(options)
        options.append("Skip this sentence for now")
        skip_option = len(options)
        options.append("Stop processing for now")
        break_option = len(options)

        print("Select how to handle the following sentence:")
        print(entry.sentence)
        print(entry.lemma + "\t" + "\t".join(labels))
        self._show_label_info(labels)

        option = _get_numbered_options("Select the appropriate option from below:", options)
        # We now have our option.
        if option == -1:
            raise RuntimeError("Something went horribly wrong during input parsing")
        if option < merge_option:
            return "gold", labels[option - 1]
        elif option == merge_option:
            reasons = ["Indistinguishable", "Circular", "Metaphorical", "Obsolete or dialectical"]
            reason_choice = _get_numbered_options("\nWhy should these be merged?", reasons) - 1
            reason = reasons[reason_choice]
            # If we have a metaphorical or obsolete/dialectical we have to ask for base label
            if reason == "Metaphorical" or reason == "Obsolete or dialectical":
                label_choice = _get_numbered_options("\nWhich label is the base label?", labels) - 1
                base_label = labels[label_choice]
                other_label = labels[not label_choice]
                self._merge_labels(base_label, other_label, force_label1=True)
                return "gold", base_label
            else:
                return "merge", reason
        elif option == missing_context:
            return "gold", "-2"
        elif option == no_sense:
            return "gold", "-1"
        elif option == skip_option:
            return "skip", 0
        elif option == break_option:
            return "break", 0
        else:
            raise RuntimeError("Something else went horribly wrong during input parsing")

    # =========== I/O ==============================================================================================
    # ==============================================================================================================

    @classmethod
    def _load_db_dump(cls, filepath, name, labeltype, lang, upos, anonymize=True, min_count=50):
        """ Load mongo dump. If anonymize is true we replace actual annotator names with numbers and write out a file
        with the labeling """
        annotator_counts = {}
        with open(filepath, "r", encoding="utf8") as f:
            dataset = cls(name=name, labeltype=labeltype, lang=lang, load_lemma_ids=True)
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
                    annotations.append(LABEL_PREFIXES[dataset.labeltype] + annotation["annotation"])
                    annotators.append(annotation["annotator"])
                    if annotation["annotator"] in annotator_counts:
                        annotator_counts[annotation["annotator"]] += 1
                    else:
                        annotator_counts[annotation["annotator"]] = 1

                dataset.add_entry(label=None, lemma=entry_lemma, upos=upos, sentence=sentence, raw_labels=annotations,
                                  annotators=annotators, source_id=source_id, pivot_start=pivot_start,
                                  pivot_end=pivot_end)

            # Filter out annotators with low annotation count (gets rid of test annotators if they got through to
            # this stage, as well as any entries that are no longer annotated after this filtering)
            low_count_annotators = [key for key, value in annotator_counts.items() if value < min_count]
            filtered_entries = []
            for entry in dataset.entries:
                filtered_labels = []
                filtered_annotators = []
                for i in range(len(entry.raw_labels)):
                    if entry.annotators[i] in low_count_annotators:
                        continue
                    else:
                        filtered_labels.append(entry.raw_labels[i])
                        filtered_annotators.append(entry.annotators[i])

                if len(filtered_labels) > 0:
                    entry.raw_labels = filtered_labels
                    entry.annotators = filtered_annotators
                    filtered_entries.append(entry)

            dataset.entries = filtered_entries
            dataset._clean_sentences()

            if anonymize:
                dataset.anonymize()

            return dataset

    @classmethod
    def _load_verb_db_dump(cls, filepath, anonymize=True):
        return cls._load_db_dump(filepath, "ttvc_2", "gn", "de", "VERB", anonymize=anonymize, min_count=100)

    def _load_entry_dict(self, json_entry, sentences):
        entry_dict = super()._load_entry_dict(json_entry, sentences)
        raw_labels = json_entry["raw_labels"]
        annotators = json_entry["annotators"]
        entry_dict["raw_labels"] = raw_labels
        entry_dict["annotators"] = annotators
        return entry_dict

    @classmethod
    def _load(cls, json_path):
        dataset, loaded_json = super()._load(json_path)
        dataset.merge_map = loaded_json["merge_map"]
        dataset.merge_targets = set(loaded_json["merge_targets"])
        dataset.merge_reasons = loaded_json["merge_reasons"]
        dataset.lemma_ids = loaded_json["lemma_ids"]
        return dataset, loaded_json

    def save(self, outpath):
        # Save as with super, but also separately write out merge file
        super().save(outpath)
        with open(os.path.join(os.path.dirname(outpath), "mergeMap.txt"), "wt", encoding="utf8", newline="") as f:
            f.write("old\tnew\treason\n")
            for key in self.merge_map:
                f.write(key + "\t" + self.merge_map[key] + "\t" + self.merge_reasons[key] + "\n")

    def save_as_publishable(self, outpath):
        # Print out a csv file with: source-id, gold label, annotations for each annotator
        annotators = self.annotators()
        with open(outpath, "wt", encoding="utf8", newline="") as f:
            f.write("Sentence\tlemma\tgold\t" + "\t".join(annotators) + "\n")
            for entry in self.entries:
                annotations = []
                for annotator in annotators:
                    if annotator not in entry.annotators:
                        annotations.append("-")
                    else:
                        annotations.append(entry.raw_labels[entry.annotators.index(annotator)])
                f.write(entry.source_id +
                        "\t" + entry.lemma +
                        "\t" + entry.label + 
                        "\t" + "\t".join(annotations) + "\n")

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
            if label == "-2":
                continue
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

    def count_unprocessed_entries(self):
        return len(self.unprocessed_entries())

    def count_unprocessable_entries(self):
        counter = 0
        for entry in self.entries:
            if not entry.can_be_labeled:
                counter += 1
        return counter

    def count_entries_to_review(self):
        counter = 0
        for entry in self.entries:
            if entry.raw_labels is not None and entry.can_be_labeled and len(set(entry.raw_labels)) > 1:
                counter += 1
        return counter

    def count_multiply_annotated(self):
        counter = 0
        for entry in self.entries:
            if entry.raw_labels is not None and len(entry.raw_labels) > 1:
                counter += 1
        return counter


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
        nom += sum(count * (count - 1) for count in instance) / (rater_count - 1)
        denom += rater_count
    p_o = nom / denom
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def _get_numbered_options(question, options):
    print(question)
    print("\nOptions are:")
    for i, option in enumerate(options):
        print(str(i + 1) + "\t" + options[i])
    print("\n")

    input_valid = False
    option = -1
    while not input_valid:
        option_string = input("Please select one of the options above: ")
        try:
            option = int(option_string)
            if 0 < option <= len(options):
                input_valid = True
        except ValueError:
            input_valid = False
            print("You must enter one of the numbers for the options above!")
    return option


def load_and_process_data(inputfile, outputfile):
    # TODO: Convenience function for loading a dataset from drive and processing annotations,
    #  saving after we finish

    dataset = AnnotData.load(inputfile)
    print("{} entries left to annotate!".format(len(dataset.unprocessed_entries())))
    # Do the whole thing
    dataset.process_easy_entries()
    dataset.process_hard_entries()
    print("{} entries left to annotate!".format(len(dataset.unprocessed_entries())))
    dataset.save(outputfile)
