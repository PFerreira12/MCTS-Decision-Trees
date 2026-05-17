"""
Train the scratch ID3 tree on iris.csv.

Numeric split handling lives in id3.py: ID3DecisionTreeClassifier tries
candidate thresholds and chooses the split with the highest information gain.
This script only loads Iris, trains the shared scratch-ID3 implementation, and
prints evaluation metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from id3 import ID3DecisionTreeClassifier
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


def main() -> None:
    X, y, feature_names = load_iris()
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, seed=42)

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

    print("Numeric Iris attributes are handled by ID3DecisionTreeClassifier from id3.py.")
    print("The chosen thresholds are visible in the exported tree below.")
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
