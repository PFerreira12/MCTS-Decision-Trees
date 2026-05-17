"""
PopOut agent backed by the scratch ID3 decision tree.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from logic import PopOutGame


N_ACTIONS = 14
DEFAULT_MODEL_PATH = Path("dt_id3_scratch_model.pkl")


def encode_move(move: Tuple[str, int]) -> int:
    move_type, col = move
    if move_type == "drop":
        return col
    if move_type == "pop":
        return 7 + col
    raise ValueError(f"Unsupported move type for ID3 action space: {move_type}")


def decode_move(label: int) -> Tuple[str, int]:
    label = int(label)
    if 0 <= label <= 6:
        return "drop", label
    if 7 <= label <= 13:
        return "pop", label - 7
    raise ValueError(f"Invalid encoded action: {label}")


def state_to_features(game: PopOutGame) -> np.ndarray:
    return np.asarray(list(game.board.flatten()) + [game.current_player], dtype=int)


class ID3Player:
    """Agent interface around the scratch ID3 model."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH, name: str = "ID3"):
        self.name = name
        with Path(model_path).open("rb") as fh:
            self.model = pickle.load(fh)

    def get_move(self, game: PopOutGame) -> Optional[Tuple[str, int]]:
        all_legal_moves = game.get_legal_moves()
        legal_moves = [move for move in all_legal_moves if move[0] in {"drop", "pop"}]
        if not legal_moves:
            if ("draw", -1) in all_legal_moves:
                return "draw", -1
            return None

        features = state_to_features(game).reshape(1, -1)
        predicted = decode_move(int(self.model.predict(features)[0]))
        if predicted in legal_moves:
            return predicted

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            class_to_index = {int(cls): idx for idx, cls in enumerate(self.model.classes_)}
            scored = []
            for move in legal_moves:
                label = encode_move(move)
                if label in class_to_index:
                    scored.append((probabilities[class_to_index[label]], move))
            if scored:
                return max(scored, key=lambda item: item[0])[1]

        return random.choice(legal_moves)
