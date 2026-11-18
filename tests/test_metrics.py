import numpy as np

from visdet.metrics import best_f1, match_counts, precision_recall


def _gt():
    return {1: (np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), np.array([0, 1]))}


def test_perfect_predictions():
    preds = {1: (np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), np.array([0.9, 0.8]), np.array([0, 1]))}
    assert match_counts(_gt(), preds, 0.25) == (2, 0, 0)
    pr = precision_recall(_gt(), preds, 0.25)
    assert pr["precision"] == 1 and pr["recall"] == 1


def test_wrong_class_and_missing():
    preds = {1: (np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), np.array([0.9, 0.8]), np.array([1, 1]))}
    tp, fp, fn = match_counts(_gt(), preds, 0.25)
    assert (tp, fp, fn) == (0, 2, 2)


def test_conf_threshold_and_duplicates():
    preds = {1: (np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], float),
                 np.array([0.9, 0.8, 0.1]), np.array([0, 0, 1]))}
    assert match_counts(_gt(), preds, 0.25) == (1, 1, 1)     # doublon = FP, la 3e boite est sous le seuil
    assert best_f1(_gt(), preds)["f1"] > 0
