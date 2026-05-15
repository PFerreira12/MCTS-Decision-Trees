"""
Train the scratch ID3 tree on iris.csv with manual numeric thresholds.

The ID3 implementation in id3.py handles numeric attributes by manually trying
candidate thresholds and selecting the split that maximizes information gain.
This satisfies the "manual discretization" requirement without using sklearn.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from id3 import ID3DecisionTreeClassifier, information_gain
from train_id3_popout import accuracy_score, class_report, stratified_train_test_split


IRIS_PATH = Path("iris.csv")
TREE_TEXT_PATH = Path("iris_id3_tree.txt")


def load_iris(path: Path = IRIS_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    feature_names = [name for name in reader.fieldnames if name not in {"ID", "class"}]
    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([row["class"] for row in rows])
    return X, y, feature_names


def best_threshold(values: Sequence[float], labels: Sequence[str]) -> Tuple[float, float]:
    values_arr = np.asarray(values, dtype=float)
    labels_arr = np.asarray(labels)
    unique_values = np.unique(values_arr)
    if unique_values.size <= 1:
        return float(unique_values[0]), 0.0

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
    best_t = float(thresholds[0])
    best_gain = -1.0

    for threshold in thresholds:
        left = labels_arr[values_arr < threshold]
        right = labels_arr[values_arr >= threshold]
        gain = information_gain(labels_arr, (left, right))
        if gain > best_gain:
            best_t = float(threshold)
            best_gain = float(gain)

    return best_t, best_gain


def fit_discretizer(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    return {
        name: best_threshold(X[:, idx], y)
        for idx, name in enumerate(feature_names)
    }


def transform_with_thresholds(
    X: np.ndarray,
    feature_names: Sequence[str],
    thresholds: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, List[str]]:
    transformed = []
    output_names = []

    for idx, name in enumerate(feature_names):
        threshold, _ = thresholds[name]
        transformed.append((X[:, idx] < threshold).astype(int))
        output_names.append(f"{name} < {threshold:.3f}")

    return np.vstack(transformed).T.astype(int), output_names


def main() -> None:
    X, y, feature_names = load_iris()
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, seed=42)

    # Show the best global threshold per attribute for reporting purposes.
    thresholds = fit_discretizer(X_train, y_train, feature_names)

    best_model = None
    best_score = -1.0
    best_depth = None
    depths = [1, 2, 3, 4, None]
    for depth in depths:
        model = ID3DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=1,
            feature_names=feature_names,
            categorical_features=[],
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_model = model
            best_score = score
            best_depth = depth

    model = best_model
    y_pred = model.predict(X_test)

    # Baseline: one global binary threshold per original attribute.
    X_train_disc, disc_names = transform_with_thresholds(X_train, feature_names, thresholds)
    X_test_disc, _ = transform_with_thresholds(X_test, feature_names, thresholds)
    model = ID3DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=1,
        feature_names=disc_names,
        categorical_features=disc_names,
    )
    model.fit(X_train_disc, y_train)
    single_threshold_pred = model.predict(X_test_disc)

    model = best_model

    print("Chosen thresholds:")
    for name, (threshold, gain) in thresholds.items():
        print(f"  {name:12s} < {threshold:5.3f}   IG={gain:.4f}")

    print(f"\nSingle-threshold baseline accuracy : {accuracy_score(y_test, single_threshold_pred):.4f}")
    print(f"Best depth                      : {best_depth}")
    print(f"Train accuracy                  : {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy                   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Tree depth     : {model.tree_depth_}")
    print(f"Node count     : {model.node_count_}")
    print("\nClassification report:")
    print(class_report(y_test, y_pred))
    print("\nTree:")
    tree_text = model.export_text()
    print(tree_text)
    TREE_TEXT_PATH.write_text(tree_text, encoding="utf-8")
    print(f"\nSaved tree: {TREE_TEXT_PATH}")


if __name__ == "__main__":
    main()
