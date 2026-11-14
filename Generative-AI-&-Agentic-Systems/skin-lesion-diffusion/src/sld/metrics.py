"""Per-class recall matters more than overall accuracy here: a model that calls
everything a common mole is 67% accurate on HAM10000 and useless."""
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def per_class_report(y_true: list, y_pred: list, classes: list, rare: list) -> dict:
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    out = {f"recall_{c}": round(float(r), 3) for c, r in zip(classes, recalls)}
    out["rare_class_recall"] = round(sum(out[f"recall_{c}"] for c in rare) / len(rare), 3)
    out["balanced_accuracy"] = round(float(balanced_accuracy_score(y_true, y_pred)), 3)
    out["macro_f1"] = round(float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)), 3)
    out["accuracy"] = round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 3)
    return out
