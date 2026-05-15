"""
Monte Carlo Tree Search agents for PopOut.

This module contains the complete MCTS implementation that was previously
defined inside the notebook:
- Standard UCT MCTS
- RAVE / AMAF MCTS
- Top-K expansion pruning
- heuristic rollout/prior MCTS
- a small MCTSPlayer wrapper used by the GUI
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from logic import MAX_ROLLOUT_MOVES


Move = Tuple[str, int]


class MCTSNode:
    """
    A node in the MCTS search tree.

    wins are accumulated from the perspective of the player who just moved
    into the node.
    """

    __slots__ = (
        "state",
        "move",
        "parent",
        "children",
        "visits",
        "wins",
        "untried",
        "player",
        "rave_visits",
        "rave_wins",
        "prior_wins",
    )

    def __init__(
        self,
        state,
        move: Optional[Move] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state = state
        self.move = move
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.wins = 0.0
        self.player = 3 - state.current_player
        self.untried: List[Move] = state.get_legal_moves()
        random.shuffle(self.untried)

        self.rave_visits: Dict[Move, int] = defaultdict(int)
        self.rave_wins: Dict[Move, float] = defaultdict(float)
        self.prior_wins = 0.0

    def uct_score(self, c: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float("inf")

        parent_visits = self.parent.visits if self.parent else self.visits
        parent_visits = max(parent_visits, 1)
        exploitation = (self.wins + self.prior_wins) / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_terminal(self) -> bool:
        return self.state.game_over

    def best_child(self, c: float = math.sqrt(2)) -> "MCTSNode":
        return max(self.children, key=lambda child: child.uct_score(c))

    def most_visited_child(self) -> "MCTSNode":
        return max(self.children, key=lambda child: child.visits)

    def expand(self) -> "MCTSNode":
        move = self.untried.pop()
        new_state = self.state.copy()
        new_state.make_move(move[0], move[1])
        child = MCTSNode(new_state, move=move, parent=self)
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        return (
            f"MCTSNode(move={self.move}, visits={self.visits}, "
            f"wins={self.wins:.1f}, children={len(self.children)})"
        )


class MCTS:
    """Standard Monte Carlo Tree Search with UCT selection."""

    reason = "MCTS"

    def __init__(
        self,
        time_limit: float = 1.0,
        iterations: Optional[int] = None,
        c: float = math.sqrt(2),
    ):
        self.time_limit = time_limit
        self.iterations = iterations
        self.c = c
        self.root: Optional[MCTSNode] = None
        self.last_search_stats: Dict[str, object] = {}

    def choose_move(self, game) -> Optional[Move]:
        result = self.search(game, return_stats=False)
        return result

    def search(self, game, return_stats: bool = False):
        legal = game.get_legal_moves()
        if not legal:
            stats = self._empty_stats(None, 0.0)
            return (None, stats) if return_stats else None
        if len(legal) == 1:
            stats = self._empty_stats(legal[0], 0.0)
            return (legal[0], stats) if return_stats else legal[0]

        self.root = MCTSNode(game.copy())
        elapsed = self._run_search()
        best = self.root.most_visited_child() if self.root.children else None
        move = best.move if best else random.choice(legal)
        stats = self._build_stats(move, elapsed)
        self.last_search_stats = stats
        return (move, stats) if return_stats else move

    def _run_search(self) -> float:
        start = time.time()
        n = 0
        while True:
            if self.iterations is not None and n >= self.iterations:
                break
            if self.iterations is None and time.time() - start > self.time_limit:
                break

            node = self._select(self.root)
            if not node.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            n += 1
        return time.time() - start

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and not node.is_terminal():
            if not node.children:
                break
            node = node.best_child(self.c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.untried:
            return node.expand()
        return node

    def _simulate(self, node: MCTSNode) -> float:
        state = node.state.copy()
        mover = node.player
        moves_played = 0

        while not state.game_over:
            if moves_played >= MAX_ROLLOUT_MOVES:
                return 0.5
            moves = state.get_legal_moves()
            if not moves:
                break
            move = random.choice(moves)
            state.make_move(move[0], move[1])
            moves_played += 1

        return self._outcome(state, mover)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        root_player = self.root.state.current_player
        while node is not None:
            node.visits += 1
            if node.player == root_player:
                node.wins += reward
            else:
                node.wins += 1 - reward
            node = node.parent

    @staticmethod
    def _outcome(state, player: int) -> float:
        if state.winner == player:
            return 1.0
        if state.is_draw or state.winner is None:
            return 0.5
        return 0.0

    def get_move_statistics(self) -> List[Dict[str, object]]:
        if self.root is None:
            return []
        stats = []
        for child in self.root.children:
            stats.append(
                {
                    "move": child.move,
                    "visits": child.visits,
                    "win_rate": child.wins / child.visits if child.visits else 0.0,
                    "uct": child.uct_score(self.c),
                }
            )
        return sorted(stats, key=lambda item: item["visits"], reverse=True)

    def _build_stats(self, move: Optional[Move], elapsed: float) -> Dict[str, object]:
        move_stats = self.get_move_statistics()
        chosen = next((item for item in move_stats if item["move"] == move), None)
        return {
            "reason": self.reason,
            "elapsed_s": elapsed,
            "root_sims": self.root.visits if self.root else 0,
            "n_children": len(self.root.children) if self.root else 0,
            "chosen_visits": chosen["visits"] if chosen else 0,
            "chosen_wr": chosen["win_rate"] if chosen else 0.0,
            "move_stats": move_stats,
        }

    def _empty_stats(self, move: Optional[Move], elapsed: float) -> Dict[str, object]:
        return {
            "reason": self.reason,
            "elapsed_s": elapsed,
            "root_sims": 0,
            "n_children": 0,
            "chosen_visits": 0,
            "chosen_wr": 0.0,
            "move_stats": [],
            "move": move,
        }


class MCTS_RAVE(MCTS):
    """MCTS with RAVE / AMAF statistics."""

    reason = "MCTS RAVE"

    def __init__(
        self,
        time_limit: float = 1.0,
        iterations: Optional[int] = None,
        c: float = math.sqrt(2),
        k: float = 1000,
    ):
        super().__init__(time_limit=time_limit, iterations=iterations, c=c)
        self.k = k

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and not node.is_terminal():
            if not node.children:
                break
            node = self._best_rave_child(node)
        return node

    def _best_rave_child(self, node: MCTSNode) -> MCTSNode:
        def rave_value(child: MCTSNode) -> float:
            n = child.visits
            n_rave = node.rave_visits[child.move]
            beta = (
                n_rave / (n + n_rave + 4 * self.k * n * n_rave)
                if (n + n_rave) > 0
                else 1.0
            )
            uct = child.wins / child.visits if child.visits else 0.0
            rave = node.rave_wins[child.move] / n_rave if n_rave else 0.0
            explore = self.c * math.sqrt(math.log(max(node.visits, 1)) / max(child.visits, 1))
            return (1 - beta) * uct + beta * rave + explore

        return max(node.children, key=rave_value)

    def _simulate(self, node: MCTSNode):
        state = node.state.copy()
        mover = node.player
        moves_played: List[Tuple[Move, int]] = []

        while not state.game_over:
            if len(moves_played) >= MAX_ROLLOUT_MOVES:
                return 0.5, moves_played
            moves = state.get_legal_moves()
            if not moves:
                break
            move = random.choice(moves)
            moves_played.append((move, state.current_player))
            state.make_move(move[0], move[1])

        reward = self._outcome(state, mover)
        return reward, moves_played

    def _backpropagate(self, node: MCTSNode, payload) -> None:
        reward, moves_played = payload if isinstance(payload, tuple) else (payload, [])
        root_player = self.root.state.current_player
        current = node

        while current is not None:
            current.visits += 1
            flipped = reward if current.player == root_player else 1 - reward
            current.wins += flipped

            if current.parent is not None:
                parent_reward = reward if current.parent.player == root_player else 1 - reward
                parent_player = current.parent.player
                for move, player_of_move in moves_played:
                    if player_of_move == parent_player:
                        current.parent.rave_visits[move] += 1
                        current.parent.rave_wins[move] += parent_reward

            current = current.parent


class MCTSTopK(MCTS):
    """MCTS that keeps only the top K children after expansion."""

    reason = "MCTS Top-K"

    def __init__(
        self,
        time_limit: float = 1.0,
        iterations: Optional[int] = None,
        c: float = math.sqrt(2),
        k: int = 7,
    ):
        super().__init__(time_limit=time_limit, iterations=iterations, c=c)
        self.k = k

    def _expand(self, node: MCTSNode) -> MCTSNode:
        child = super()._expand(node)
        if len(node.children) > self.k:
            node.children.sort(key=lambda candidate: candidate.uct_score(self.c), reverse=True)
            node.children = node.children[: self.k]
            if child not in node.children:
                child = node.children[0]
        return child

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and not node.is_terminal():
            if not node.children:
                break
            node = node.best_child(self.c)
        return node


class MCTSWithHeuristics(MCTS):
    """MCTS with biased rollouts and heuristic priors."""

    reason = "MCTS Heuristic"

    def __init__(
        self,
        time_limit: float = 1.0,
        iterations: Optional[int] = None,
        c: float = math.sqrt(2),
        prior_weight: float = 2.0,
    ):
        super().__init__(time_limit=time_limit, iterations=iterations, c=c)
        self.prior_weight = prior_weight

    @staticmethod
    def _score_move_fast(state, move: Move, player: int) -> float:
        if move[0] == "draw":
            return 0.5

        move_type, col = move
        board = state.board
        score = 1.0

        if col == 3:
            score += 0.8
        elif col in (2, 4):
            score += 0.5
        elif col in (1, 5):
            score += 0.2

        bottom_row = board.shape[0] - 1
        opponent = 3 - player

        if move_type == "pop":
            if board[bottom_row, col] == opponent:
                score += 1.5
            elif board[bottom_row, col] == player:
                score += 0.3
        elif move_type == "drop":
            col_pieces = int((board[:, col] != 0).sum())
            if col_pieces >= board.shape[0] - 1:
                score -= 0.3

        return max(score, 0.01)

    @staticmethod
    def _score_move(state, move: Move, player: int) -> float:
        if move[0] == "draw":
            return 0.5

        opponent = 3 - player
        test = state.copy()
        test.make_move(move[0], move[1])

        if test.game_over and test.winner == player:
            return 5.0

        for opponent_move in state.get_legal_moves():
            if opponent_move[0] == "draw":
                continue
            reply = state.copy()
            reply.current_player = opponent
            reply.make_move(opponent_move[0], opponent_move[1])
            if reply.game_over and reply.winner == opponent:
                if opponent_move == move:
                    return 4.0

        score = 1.0
        if move[1] in (3, 2, 4):
            score += 0.5
        return score

    def _weighted_choice_fast(self, state, moves: List[Move], player: int) -> Move:
        weights = [max(self._score_move_fast(state, move, player), 0.01) for move in moves]
        return self._weighted_choice_from_weights(moves, weights)

    def _weighted_choice(self, state, moves: List[Move], player: int) -> Move:
        weights = [max(self._score_move(state, move, player), 0.01) for move in moves]
        return self._weighted_choice_from_weights(moves, weights)

    @staticmethod
    def _weighted_choice_from_weights(moves: List[Move], weights: List[float]) -> Move:
        total = sum(weights)
        marker = random.random() * total
        cumulative = 0.0
        for move, weight in zip(moves, weights):
            cumulative += weight
            if marker <= cumulative:
                return move
        return moves[-1]

    def _simulate(self, node: MCTSNode) -> float:
        state = node.state.copy()
        mover = node.player
        moves_played = 0

        while not state.game_over:
            if moves_played >= MAX_ROLLOUT_MOVES:
                return 0.5

            moves = state.get_legal_moves()
            if not moves:
                break

            winning_move = None
            for move in moves:
                if move[0] == "draw":
                    continue
                test = state.copy()
                test.make_move(move[0], move[1])
                if test.game_over and test.winner == state.current_player:
                    winning_move = move
                    break

            move = winning_move or self._weighted_choice_fast(state, moves, state.current_player)
            state.make_move(move[0], move[1])
            moves_played += 1

        return self._outcome(state, mover)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        child = super()._expand(node)
        if child.move[0] == "draw":
            child.prior_wins = 0.5 * self.prior_weight
            return child

        move_count = len(node.state.move_history) if hasattr(node.state, "move_history") else 99
        if move_count < 10:
            prior = self._score_move(node.state, child.move, child.player)
        else:
            prior = self._score_move_fast(node.state, child.move, child.player)
        child.prior_wins = prior * self.prior_weight
        return child


def make_mcts(strategy: str = "standard", **kwargs) -> MCTS:
    mapping = {
        "standard": MCTS,
        "rave": MCTS_RAVE,
        "topk": MCTSTopK,
        "heuristic": MCTSWithHeuristics,
    }
    try:
        cls = mapping[strategy.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(mapping)}") from exc
    return cls(**kwargs)


class MCTSPlayer:
    """Small player wrapper used by the GUI and simple game loops."""

    def __init__(
        self,
        name: str = "MCTS",
        player_num: int = 1,
        strategy: str = "topk",
        time_limit: float = 1.0,
        iterations: Optional[int] = None,
        **kwargs,
    ):
        self.name = name
        self.player_num = player_num
        self.strategy = strategy
        self.engine = make_mcts(
            strategy,
            time_limit=time_limit,
            iterations=iterations,
            **kwargs,
        )

    def get_move(self, game) -> Optional[Move]:
        return self.engine.choose_move(game)


__all__ = [
    "MCTSNode",
    "MCTS",
    "MCTS_RAVE",
    "MCTSTopK",
    "MCTSWithHeuristics",
    "MCTSPlayer",
    "make_mcts",
]
