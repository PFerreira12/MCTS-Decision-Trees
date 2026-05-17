from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def entropy(labels: Sequence[Any]) -> float:
    """Shannon entropy H(S) for a sequence of class labels."""
    y = np.asarray(labels)
    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def information_gain(parent_labels: Sequence[Any], child_label_groups: Iterable[Sequence[Any]]) -> float:
    """Information gain produced by splitting parent_labels into child groups."""
    parent = np.asarray(parent_labels)
    if parent.size == 0:
        return 0.0

    base_entropy = entropy(parent)
    remainder = 0.0

    for group in child_label_groups:
        child = np.asarray(group)
        if child.size == 0:
            continue
        remainder += (child.size / parent.size) * entropy(child)

    return float(base_entropy - remainder)


@dataclass
class ID3Node:
    is_leaf: bool
    prediction: Any
    class_counts: Dict[Any, int]
    n_samples: int
    depth: int
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    split_type: Optional[str] = None  # "categorical" or "numeric"
    threshold: Optional[float] = None
    gain: float = 0.0
    children: Dict[Any, "ID3Node"] = field(default_factory=dict)


class ID3DecisionTreeClassifier:
    """
    ID3 classifier implemented from scratch.

    Parameters
    ----------
    max_depth:
        Maximum tree depth. None means unlimited.
    min_samples_split:
        Do not split nodes with fewer samples than this.
    min_samples_leaf:
        Reject splits that would create a child leaf with fewer samples.
    min_gain:
        Minimum information gain required to split.
    max_categorical_values:
        Numeric-looking columns with at most this many unique values are treated
        as categorical by default. This is useful for PopOut board cells.
    categorical_features:
        Optional indices/names that must be treated as categorical.
    feature_names:
        Optional display names used by export_text.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-12,
        max_categorical_values: int = 10,
        categorical_features: Optional[Sequence[int | str]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_categorical_values = max_categorical_values
        self.categorical_features = set(categorical_features or [])
        self.feature_names = list(feature_names) if feature_names is not None else None

        self.root_: Optional[ID3Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.tree_depth_: int = 0
        self.node_count_: int = 0

    def fit(self, X: Sequence[Sequence[Any]], y: Sequence[Any]) -> "ID3DecisionTreeClassifier":
        X_arr = self._as_2d_array(X)
        y_arr = np.asarray(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X_arr.shape[0] == 0:
            raise ValueError("Cannot fit ID3 on an empty dataset.")

        self.n_features_in_ = X_arr.shape[1]
        if self.feature_names is None:
            self.feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        if len(self.feature_names) != self.n_features_in_:
            raise ValueError("feature_names length must match the number of columns in X.")

        self.classes_ = np.unique(y_arr)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        available_features = tuple(range(self.n_features_in_))
        self.root_ = self._build_tree(X_arr, y_arr, available_features, depth=0)

        total_gain = self.feature_importances_.sum()
        if total_gain > 0:
            self.feature_importances_ = self.feature_importances_ / total_gain

        self.tree_depth_ = self._max_depth(self.root_)
        self.node_count_ = self._count_nodes(self.root_)
        return self

    def predict(self, X: Sequence[Sequence[Any]]) -> np.ndarray:
        self._check_fitted()
        X_arr = self._as_2d_array(X)
        return np.asarray([self._predict_one(row, self.root_) for row in X_arr])

    def predict_proba(self, X: Sequence[Sequence[Any]]) -> np.ndarray:
        self._check_fitted()
        X_arr = self._as_2d_array(X)
        rows = []
        for row in X_arr:
            node = self._leaf_for_row(row, self.root_)
            total = sum(node.class_counts.values())
            if total == 0:
                rows.append(np.zeros(len(self.classes_), dtype=float))
                continue
            rows.append(np.asarray([node.class_counts.get(cls, 0) / total for cls in self.classes_]))
        return np.vstack(rows)

    def score(self, X: Sequence[Sequence[Any]], y: Sequence[Any]) -> float:
        y_arr = np.asarray(y)
        if y_arr.size == 0:
            return 0.0
        return float(np.mean(self.predict(X) == y_arr))

    def export_text(self, max_depth: Optional[int] = None) -> str:
        self._check_fitted()
        lines: List[str] = []
        self._export_node(self.root_, lines, indent="", max_depth=max_depth)
        return "\n".join(lines)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        available_features: Tuple[int, ...],
        depth: int,
    ) -> ID3Node:
        prediction = self._majority_class(y)
        counts = self._class_counts(y)
        node = ID3Node(
            is_leaf=True,
            prediction=prediction,
            class_counts=counts,
            n_samples=int(y.size),
            depth=depth,
        )

        if np.unique(y).size == 1:
            return node
        if not available_features:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if y.size < self.min_samples_split:
            return node

        split = self._best_split(X, y, available_features)
        if split is None or split["gain"] < self.min_gain:
            return node

        feature = split["feature"]
        node.is_leaf = False
        node.feature_index = feature
        node.feature_name = self.feature_names[feature]
        node.split_type = split["split_type"]
        node.threshold = split.get("threshold")
        node.gain = split["gain"]

        self.feature_importances_[feature] += split["gain"] * y.size
        next_features = tuple(f for f in available_features if f != feature)

        for branch_value, indices in split["branches"].items():
            node.children[branch_value] = self._build_tree(
                X[indices],
                y[indices],
                next_features,
                depth + 1,
            )

        return node

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        available_features: Tuple[int, ...],
    ) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None

        for feature in available_features:
            candidate = (
                self._categorical_split(X, y, feature)
                if self._is_categorical_feature(X[:, feature], feature)
                else self._numeric_split(X, y, feature)
            )
            if candidate is None:
                continue
            if best is None or candidate["gain"] > best["gain"]:
                best = candidate

        return best

    def _categorical_split(self, X: np.ndarray, y: np.ndarray, feature: int) -> Optional[Dict[str, Any]]:
        values = np.unique(X[:, feature])
        if values.size <= 1:
            return None

        branches = {value: np.where(X[:, feature] == value)[0] for value in values}
        if min(len(indices) for indices in branches.values()) < self.min_samples_leaf:
            return None
        gain = information_gain(y, (y[idx] for idx in branches.values()))
        return {
            "feature": feature,
            "split_type": "categorical",
            "gain": gain,
            "branches": branches,
        }

    def _numeric_split(self, X: np.ndarray, y: np.ndarray, feature: int) -> Optional[Dict[str, Any]]:
        values = X[:, feature].astype(float)
        unique_values = np.unique(values)
        if unique_values.size <= 1:
            return None

        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
        best: Optional[Dict[str, Any]] = None

        for threshold in thresholds:
            left = np.where(values < threshold)[0]
            right = np.where(values >= threshold)[0]
            if left.size == 0 or right.size == 0:
                continue
            if min(left.size, right.size) < self.min_samples_leaf:
                continue

            gain = information_gain(y, (y[left], y[right]))
            if best is None or gain > best["gain"]:
                best = {
                    "feature": feature,
                    "split_type": "numeric",
                    "threshold": float(threshold),
                    "gain": gain,
                    "branches": {"<": left, ">=": right},
                }

        return best

    def _is_categorical_feature(self, column: np.ndarray, feature: int) -> bool:
        name = self.feature_names[feature]
        if feature in self.categorical_features or name in self.categorical_features:
            return True
        if not self._is_numeric_column(column):
            return True
        return np.unique(column).size <= self.max_categorical_values

    def _predict_one(self, row: np.ndarray, node: ID3Node) -> Any:
        return self._leaf_for_row(row, node).prediction

    def _leaf_for_row(self, row: np.ndarray, node: ID3Node) -> ID3Node:
        while not node.is_leaf:
            if node.split_type == "numeric":
                value = float(row[node.feature_index])
                branch = "<" if value < node.threshold else ">="
            else:
                branch = row[node.feature_index]

            if branch not in node.children:
                return node
            node = node.children[branch]
        return node

    def _export_node(
        self,
        node: ID3Node,
        lines: List[str],
        indent: str,
        max_depth: Optional[int],
    ) -> None:
        counts = ", ".join(f"{cls}:{count}" for cls, count in node.class_counts.items())
        if node.is_leaf or (max_depth is not None and node.depth >= max_depth):
            lines.append(f"{indent}Predict {node.prediction}  [n={node.n_samples}; {counts}]")
            return

        if node.split_type == "numeric":
            tests = [
                ("<", f"{node.feature_name} < {node.threshold:.6g}"),
                (">=", f"{node.feature_name} >= {node.threshold:.6g}"),
            ]
        else:
            tests = [(value, f"{node.feature_name} == {value}") for value in node.children]

        for branch_value, label in tests:
            child = node.children.get(branch_value)
            if child is None:
                continue
            lines.append(f"{indent}if {label}:  [gain={node.gain:.6f}]")
            self._export_node(child, lines, indent + "  ", max_depth)

    def _check_fitted(self) -> None:
        if self.root_ is None or self.classes_ is None:
            raise ValueError("This ID3DecisionTreeClassifier instance is not fitted yet.")

    @staticmethod
    def _as_2d_array(X: Sequence[Sequence[Any]]) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array-like object.")
        return X_arr

    @staticmethod
    def _class_counts(y: np.ndarray) -> Dict[Any, int]:
        classes, counts = np.unique(y, return_counts=True)
        return {cls.item() if hasattr(cls, "item") else cls: int(count) for cls, count in zip(classes, counts)}

    @staticmethod
    def _majority_class(y: np.ndarray) -> Any:
        classes, counts = np.unique(y, return_counts=True)
        best = classes[int(np.argmax(counts))]
        return best.item() if hasattr(best, "item") else best

    @staticmethod
    def _is_numeric_column(column: np.ndarray) -> bool:
        try:
            column.astype(float)
            return True
        except (TypeError, ValueError):
            return False

    def _max_depth(self, node: ID3Node) -> int:
        if node.is_leaf:
            return node.depth
        return max(self._max_depth(child) for child in node.children.values())

    def _count_nodes(self, node: ID3Node) -> int:
        return 1 + sum(self._count_nodes(child) for child in node.children.values())


__all__ = [
    "ID3DecisionTreeClassifier",
    "ID3Node",
    "entropy",
    "information_gain",
]
