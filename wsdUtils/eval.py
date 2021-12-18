from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
from typing import List


def compute_scores(gold: List, preds: List):
    """
    Expects two lists of equal length, one containing the gold labels, the other the predictions.
    We do not allow multi label scoring as such.
    Returns a dictionary containing the precision, recall, f1-scores and support with micro, macro and weighted
    averaging. Also returns the micro accuracy specifically.
    """
    assert len(gold) == len(preds), "Gold and predictions do not have equal length!"
    results = {
        "Accuracy": accuracy_score(gold, preds),
        "Micro": [*precision_recall_fscore_support(gold, preds, average="micro")[:-1], len(gold)],
        "Macro": [*precision_recall_fscore_support(gold, preds, average="macro")[:-1], len(gold)],
        "Weighted": [*precision_recall_fscore_support(gold, preds, average="weighted")[:-1], len(gold)]
    }
    return results


def pretty_print_results(results):
    rows = [["Averaging", "Precision", "Recall", "F1 Score", "Support"]]
    rows.append(["Micro", *results["Micro"]])
    rows.append(["Macro", *results["Macro"]])
    rows.append(["Weighted", *results["Weighted"]])
    outstring = "=================================================================\n"
    outstring += "{:<9}\t{:>9}\t{:>9}\t{:>9}\t{:>9}\n".format(*rows[0])
    for row in rows[1:]:
        outstring += "{:>9}\t{:>9.4f}\t{:>9.4f}\t{:>9.4f}\t{:>9}\n".format(*row)
    outstring += "=================================================================\n"
    print(outstring + "\n")
    return outstring


def export_latex(results):
    headers = ["Averaging", "Precision", "Recall", "F1 Score", "Support", "Accuracy"]
    rows = []
    rows.append(
        ["Micro", *[item * 100 for item in results["Micro"][:-1]], results["Micro"][-1], results["Accuracy"] * 100])
    rows.append(["Macro", *[item * 100 for item in results["Macro"][:-1]], results["Macro"][-1]])
    rows.append(["Weighted", *[item * 100 for item in results["Weighted"][:-1]], results["Weighted"][-1]])
    print(tabulate(rows, headers=headers, tablefmt="latex_booktabs", floatfmt=".2f"))
