"""
Train the scratch ID3 tree on the generated PopOut MCTS dataset.

The model trained here replaces sklearn.tree.DecisionTreeClassifier for the
assignment requirement that the decision tree learner is implemented manually.
"""

from __future__ import annotations

import csv
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from id3 import ID3DecisionTreeClassifier


DATASET_PATH = Path("popout_mcts_dataset.csv")
MODEL_PATH = Path("dt_id3_scratch_model.pkl")
TREE_TEXT_PATH = Path("dt_id3_scratch_tree.txt")


def load_popout_dataset(path: Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    feature_names = [name for name in reader.fieldnames if name.startswith("r") or name == "current_player"]
    X = np.asarray([[int(row[name]) for name in feature_names] for row in rows], dtype=int)
    y = np.asarray([int(row["action"]) for row in rows], dtype=int)
    return X, y, feature_names


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for cls in sorted(set(y.tolist())):
        indices = np.where(y == cls)[0].tolist()
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_size))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.mean(y_true_arr == y_pred_arr)) if y_true_arr.size else 0.0


def class_report(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    lines = ["class  precision  recall  f1-score  support", "-" * 48]

    for label in labels:
        tp = int(np.sum((y_true_arr == label) & (y_pred_arr == label)))
        fp = int(np.sum((y_true_arr != label) & (y_pred_arr == label)))
        fn = int(np.sum((y_true_arr == label) & (y_pred_arr != label)))
        support = int(np.sum(y_true_arr == label))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        lines.append(f"{label:>5}  {precision:>9.3f}  {recall:>6.3f}  {f1:>8.3f}  {support:>7}")

    return "\n".join(lines)


def legal_action_labels(features: Sequence[int]) -> List[int]:
    """Return encoded legal drop/pop actions for a PopOut feature row."""
    row = np.asarray(features, dtype=int)
    board = row[:42].reshape(6, 7)
    player = int(row[42])
    labels: List[int] = []

    for col in range(7):
        if board[0, col] == 0:
            labels.append(col)
    for col in range(7):
        if board[5, col] == player:
            labels.append(7 + col)

    return labels


def legal_aware_predict(model: ID3DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
    """
    Predict actions using the same legality fallback used by ID3Player.

    If the top class is illegal in the current state, choose the legal action
    with the highest predicted probability.
    """
    raw = model.predict(X)
    probabilities = model.predict_proba(X)
    class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    predictions: List[int] = []

    for predicted, probs, features in zip(raw, probabilities, X):
        predicted = int(predicted)
        legal = legal_action_labels(features)
        if predicted in legal:
            predictions.append(predicted)
            continue

        scored = [(probs[class_to_index[label]], label) for label in legal if label in class_to_index]
        predictions.append(max(scored)[1] if scored else predicted)

    return np.asarray(predictions, dtype=int)


def legal_aware_accuracy(model: ID3DecisionTreeClassifier, X: np.ndarray, y: Sequence[int]) -> float:
    return accuracy_score(y, legal_aware_predict(model, X))


def cross_validate_depths(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    depths: Iterable[int | None],
    min_samples_leaf: int = 1,
    scoring: str = "raw",
    folds: int = 5,
    seed: int = 42,
) -> Tuple[int | None, List[Tuple[int | None, float]]]:
    rng = random.Random(seed)
    fold_indices = [[] for _ in range(folds)]

    for cls in sorted(set(y.tolist())):
        indices = np.where(y == cls)[0].tolist()
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            fold_indices[i % folds].append(idx)

    scores: List[Tuple[int | None, float]] = []
    all_indices = set(range(len(y)))

    for depth in depths:
        fold_scores = []
        for fold in range(folds):
            test_idx = np.asarray(fold_indices[fold], dtype=int)
            train_idx = np.asarray(sorted(all_indices - set(test_idx.tolist())), dtype=int)

            model = ID3DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_samples_leaf,
                feature_names=feature_names,
                categorical_features=feature_names,
            )
            model.fit(X[train_idx], y[train_idx])
            if scoring == "legal_aware":
                fold_scores.append(legal_aware_accuracy(model, X[test_idx], y[test_idx]))
            else:
                fold_scores.append(model.score(X[test_idx], y[test_idx]))

        scores.append((depth, float(np.mean(fold_scores))))

    best_depth, _ = max(scores, key=lambda item: item[1])
    return best_depth, scores


def tune_id3_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    depths: Iterable[int | None],
    min_leaf_values: Iterable[int],
    scoring: str = "raw",
    folds: int = 5,
    seed: int = 42,
) -> Tuple[Tuple[int | None, int], List[Tuple[int | None, int, float]]]:
    results: List[Tuple[int | None, int, float]] = []
    for min_leaf in min_leaf_values:
        _, scores = cross_validate_depths(
            X,
            y,
            feature_names,
            depths,
            min_samples_leaf=min_leaf,
            scoring=scoring,
            folds=folds,
            seed=seed,
        )
        for depth, score in scores:
            results.append((depth, min_leaf, score))

    best_depth, best_min_leaf, _ = max(results, key=lambda item: item[2])
    return (best_depth, best_min_leaf), results


def main() -> None:
    X, y, feature_names = load_popout_dataset()
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
    classes = [int(cls) for cls in sorted(Counter(y).keys())]

    # Depth 0 is the majority-class baseline, not a useful decision tree.
    depths = [1, 2, 3, 4, 5, 6, 8, 12, None]
    min_leaf_values = [1, 3, 5, 10, 20, 30]
    (best_depth, best_min_leaf), cv_scores = tune_id3_hyperparameters(
        X_train,
        y_train,
        feature_names,
        depths,
        min_leaf_values,
        scoring="legal_aware",
    )

    model = ID3DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_leaf=best_min_leaf,
        feature_names=feature_names,
        categorical_features=feature_names,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    legal_test_pred = legal_aware_predict(model, X_test)

    print(f"Dataset rows : {len(y)}")
    print(f"Features     : {len(feature_names)}")
    print(f"Classes      : {classes}")
    print("\nDepth | Min leaf | CV legal-aware accuracy")
    print("-" * 34)
    for depth, min_leaf, score in sorted(cv_scores, key=lambda item: (item[1], str(item[0]))):
        label = str(depth) if depth is not None else "None"
        print(f"{label:>5} | {min_leaf:>8} | {score:.4f}")

    majority_baseline = max(Counter(y_test).values()) / len(y_test)

    print(f"\nMajority baseline test accuracy : {majority_baseline:.4f}")
    print(f"Best depth                    : {best_depth}")
    print(f"Best min leaf  : {best_min_leaf}")
    print(f"Train accuracy : {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test accuracy  : {accuracy_score(y_test, test_pred):.4f}")
    print(f"Legal-aware test accuracy : {accuracy_score(y_test, legal_test_pred):.4f}")
    print(f"Tree depth     : {model.tree_depth_}")
    print(f"Node count     : {model.node_count_}")
    print("\nClassification report:")
    print(class_report(y_test, test_pred))

    with MODEL_PATH.open("wb") as fh:
        pickle.dump(model, fh)
    TREE_TEXT_PATH.write_text(model.export_text(max_depth=5), encoding="utf-8")

    print(f"\nSaved model : {MODEL_PATH}")
    print(f"Saved tree  : {TREE_TEXT_PATH}")


if __name__ == "__main__":
    main()
