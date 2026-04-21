"""
Monte Carlo Tree Search for PopOut.

Features
--------
- UCT-based MCTS
- Heuristic-guided expansion order
- Tactical rollout policy (not pure random)
- Detects immediate wins / immediate opponent threats
- Works directly with your PopOutGame API from logic.py
- Can be used as a drop-in "player" object in interface.py

Recommended defaults
--------------------
For human-vs-AI:
    time_limit = 1.0 to 2.0 seconds

For AI-vs-AI stronger play:
    time_limit = 2.0 to 4.0 seconds

Notes
-----
This implementation stores values from Player 1's perspective:
    P1 win  -> 1.0
    Draw    -> 0.5
    P2 win  -> 0.0

When selecting children, the node chooses the child that is best
for the player whose turn it currently is.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from logic import PopOutGame


Move = Tuple[str, int]  # ('drop' or 'pop', column)


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def opponent(player: int) -> int:
    return 2 if player == 1 else 1


def reward_from_p1_perspective(game: PopOutGame) -> float:
    """Terminal reward mapped to [0,1] from Player 1 perspective."""
    if game.winner == PopOutGame.P1:
        return 1.0
    if game.winner == PopOutGame.P2:
        return 0.0
    return 0.5


def apply_move_copy(game: PopOutGame, move: Move) -> PopOutGame:
    """Return a copied game after applying move."""
    g = game.copy()
    ok = g.make_move(move[0], move[1])
    if not ok:
        raise ValueError(f"Illegal move attempted in MCTS: {move}")
    return g


def get_immediate_winning_moves(game: PopOutGame, player: Optional[int] = None) -> List[Move]:
    """
    Return all legal moves for `player` that immediately win.
    If player is None, use game.current_player.
    """
    if player is None:
        player = game.current_player

    # If player != current player, simulate from a copied state with changed current_player.
    if player != game.current_player:
        temp = game.copy()
        temp.current_player = player
        working_game = temp
    else:
        working_game = game

    wins = []
    for move in working_game.get_legal_moves():
        g2 = working_game.copy()
        g2.make_move(move[0], move[1])
        if g2.game_over and g2.winner == player:
            wins.append(move)
    return wins


def move_ordering_score(game: PopOutGame, move: Move, root_player: Optional[int] = None) -> float:
    """
    Heuristic used to sort moves during expansion and playouts.
    Bigger is better for the current player.
    """
    if root_player is None:
        root_player = game.current_player

    cols_center = (game.cols - 1) / 2.0
    move_type, col = move
    score = 0.0

    # Prefer center columns
    score += (game.cols - abs(col - cols_center)) * 2.0

    # Slight preference for drop over pop unless tactically useful
    if move_type == 'drop':
        score += 1.5
    else:
        score -= 0.5

    g2 = game.copy()
    g2.make_move(move_type, col)

    # Immediate win is massive
    if g2.game_over and g2.winner == game.current_player:
        score += 10000.0
        return score

    # Avoid moves that immediately give opponent a winning reply
    opp_wins = get_immediate_winning_moves(g2)
    if opp_wins:
        score -= 4000.0 * len(opp_wins)

    # Prefer moves that create our own threats
    my_next_wins = get_immediate_winning_moves_for_player_after_switch(g2, game.current_player)
    score += 120.0 * len(my_next_wins)

    # Slightly punish creating draw-claimable repetition in rollout ordering
    if g2.can_declare_draw():
        score -= 5.0

    return score


def get_immediate_winning_moves_for_player_after_switch(game: PopOutGame, player: int) -> List[Move]:
    """
    Returns winning moves for `player` on the given board,
    even if `player` is not the current player.
    """
    temp = game.copy()
    temp.current_player = player
    return get_immediate_winning_moves(temp, player)


def has_forced_immediate_loss(game: PopOutGame) -> bool:
    """True if opponent (the side to move now) has an immediate winning move."""
    return len(get_immediate_winning_moves(game)) > 0


def ordered_legal_moves(game: PopOutGame, top_k: Optional[int] = None) -> List[Move]:
    moves = game.get_legal_moves()
    moves.sort(key=lambda m: move_ordering_score(game, m), reverse=True)
    if top_k is not None:
        return moves[:top_k]
    return moves


# ---------------------------------------------------------------------
# Rollout policy
# ---------------------------------------------------------------------

def choose_rollout_move(game: PopOutGame) -> Optional[Move]:
    """
    Heuristic simulation policy:
    1. Play an immediate winning move if available
    2. Block opponent immediate win if possible
    3. Prefer strong ordered moves, with some randomness
    """
    moves = game.get_legal_moves()
    if not moves:
        return None

    # 1) If we can win now, do it
    winning_moves = get_immediate_winning_moves(game)
    if winning_moves:
        return random.choice(winning_moves)

    # 2) If opponent has immediate win next turn, try to block it
    opp = opponent(game.current_player)
    opponent_wins_next_if_we_do_nothing = get_immediate_winning_moves_for_player_after_switch(game, opp)

    if opponent_wins_next_if_we_do_nothing:
        safe_moves = []
        for move in moves:
            g2 = game.copy()
            g2.make_move(move[0], move[1])

            # If we ourselves win, excellent
            if g2.game_over and g2.winner == game.current_player:
                return move

            opp_wins = get_immediate_winning_moves(g2)
            if not opp_wins:
                safe_moves.append(move)

        if safe_moves:
            safe_moves.sort(key=lambda m: move_ordering_score(game, m), reverse=True)
            # Mostly choose the best safe move, occasionally vary
            if len(safe_moves) >= 2 and random.random() < 0.15:
                return random.choice(safe_moves[:min(3, len(safe_moves))])
            return safe_moves[0]

    # 3) Heuristic-biased random choice
    ranked = ordered_legal_moves(game)
    if not ranked:
        return None

    # Soft randomness among top candidates
    band = ranked[:min(4, len(ranked))]
    weights = [max(1.0, 8.0 - i * 2.0) for i in range(len(band))]
    return random.choices(band, weights=weights, k=1)[0]


def rollout(game: PopOutGame, max_depth: int = 80) -> float:
    """
    Simulate until terminal state or max depth.
    Returns reward from Player 1 perspective.
    """
    g = game.copy()
    depth = 0

    while not g.game_over and depth < max_depth:
        # If a draw may be declared, treat it as draw during simulation.
        # This prevents endless repetition loops in rollouts.
        if g.can_declare_draw():
            return 0.5

        move = choose_rollout_move(g)
        if move is None:
            # Defensive fallback
            if g.can_declare_draw():
                return 0.5
            return reward_from_p1_perspective(g)

        g.make_move(move[0], move[1])
        depth += 1

    if g.game_over:
        return reward_from_p1_perspective(g)

    # Depth cutoff: mild static estimate
    return static_estimate_p1(g)


def static_estimate_p1(game: PopOutGame) -> float:
    """
    Lightweight non-terminal evaluation mapped to [0,1].
    Used only at rollout depth cutoff.
    """
    p1_score = count_threat_score(game, PopOutGame.P1)
    p2_score = count_threat_score(game, PopOutGame.P2)

    # Immediate tactical pressure
    p1_wins = len(get_immediate_winning_moves_for_player_after_switch(game, PopOutGame.P1))
    p2_wins = len(get_immediate_winning_moves_for_player_after_switch(game, PopOutGame.P2))

    raw = (p1_score - p2_score) + 25.0 * (p1_wins - p2_wins)

    # Squash into [0,1]
    return 1.0 / (1.0 + math.exp(-raw / 25.0))


def count_threat_score(game: PopOutGame, player: int) -> float:
    """
    Count promising windows of length 4.
    """
    b = game.board
    rows, cols = game.rows, game.cols
    score = 0.0

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                cells = []
                for i in range(4):
                    rr = r + i * dr
                    cc = c + i * dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        cells.append(b[rr, cc])
                    else:
                        cells = []
                        break

                if len(cells) != 4:
                    continue

                mine = sum(1 for x in cells if x == player)
                opp = sum(1 for x in cells if x == opponent(player))
                empties = 4 - mine - opp

                if opp == 0:
                    if mine == 4:
                        score += 1000.0
                    elif mine == 3 and empties == 1:
                        score += 30.0
                    elif mine == 2 and empties == 2:
                        score += 8.0
                    elif mine == 1 and empties == 3:
                        score += 1.5

    # Center control
    center = game.cols // 2
    for r in range(rows):
        if b[r, center] == player:
            score += 2.0

    return score


# ---------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------

@dataclass
class MCTSNode:
    game: PopOutGame
    parent: Optional["MCTSNode"] = None
    move: Optional[Move] = None
    children: Dict[Move, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    total_value_p1: float = 0.0

    # moves not expanded yet
    untried_moves: List[Move] = field(default_factory=list)

    def __post_init__(self):
        if not self.untried_moves and not self.game.game_over:
            self.untried_moves = ordered_legal_moves(self.game)

    def is_terminal(self) -> bool:
        return self.game.game_over

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def q_for_player(self, player: int) -> float:
        if self.visits == 0:
            return 0.5
        mean_p1 = self.total_value_p1 / self.visits
        return mean_p1 if player == PopOutGame.P1 else (1.0 - mean_p1)

    def uct_score(self, parent_visits: int, player_to_move: int, c: float) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.q_for_player(player_to_move)
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float) -> "MCTSNode":
        """
        Choose child using UCT from perspective of the player to move at this node.
        """
        player_to_move = self.game.current_player
        return max(
            self.children.values(),
            key=lambda child: child.uct_score(self.visits, player_to_move, c)
        )

    def expand_one(self, expansion_top_k: Optional[int] = None) -> Optional["MCTSNode"]:
        """
        Expand one untried move. If expansion_top_k is set, expansion only considers
        the top-k ordered moves initially stored in untried_moves.
        """
        if not self.untried_moves:
            return None

        move = self.untried_moves.pop(0)
        child_game = apply_move_copy(self.game, move)
        child = MCTSNode(game=child_game, parent=self, move=move)
        self.children[move] = child
        return child

    def tree_path(self) -> List["MCTSNode"]:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------
# MCTS core
# ---------------------------------------------------------------------

class MCTS:
    """
    Core search engine.
    """

    def __init__(
        self,
        time_limit: float = 1.5,
        iteration_limit: Optional[int] = None,
        exploration_constant: float = 1.35,
        rollout_depth: int = 80,
        expansion_top_k: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.expansion_top_k = expansion_top_k
        self.random = random.Random(random_seed)

        if random_seed is not None:
            random.seed(random_seed)

    def search(self, root_game: PopOutGame, return_stats: bool = False):
        """
        Run MCTS from the current state and return the selected move.

        Returns
        -------
        If return_stats is False:
            best_move

        If return_stats is True:
            (best_move, stats_dict)
        """
        root = MCTSNode(game=root_game.copy())
        current_player = root.game.current_player

        # ------------------------------------------------------------
        # 1) Immediate tactical win: play instantly
        # ------------------------------------------------------------
        immediate_wins = get_immediate_winning_moves(root.game)
        if immediate_wins:
            best = random.choice(immediate_wins)

            if return_stats:
                value_for_p1 = 1.0 if current_player == PopOutGame.P1 else 0.0
                return best, {
                    "iterations": 0,
                    "reason": "immediate win",
                    "children": {
                        best: {
                            "visits": 1,
                            "value_for_current_player": 1.0,
                            "value_for_p1": value_for_p1,
                        }
                    },
                }

            return best

        # ------------------------------------------------------------
        # 2) Look for safe moves that do not allow immediate opponent win
        # ------------------------------------------------------------
        legal_moves = root.game.get_legal_moves()
        safe_moves = []

        for move in legal_moves:
            g2 = root.game.copy()
            g2.make_move(move[0], move[1])

            # If opponent has no immediate winning reply, the move is safe
            opp_wins = get_immediate_winning_moves(g2)
            if not opp_wins:
                safe_moves.append(move)

        # If exactly one safe move exists, play it instantly
        if len(safe_moves) == 1:
            chosen = safe_moves[0]

            if return_stats:
                return chosen, {
                    "iterations": 0,
                    "reason": "forced defense",
                    "children": {
                        chosen: {
                            "visits": 1,
                            "value_for_current_player": 0.5,
                            "value_for_p1": 0.5,
                        }
                    },
                }

            return chosen

        # ------------------------------------------------------------
        # 3) Optionally restrict first-level expansion
        # ------------------------------------------------------------
        if self.expansion_top_k is not None:
            root.untried_moves = root.untried_moves[:self.expansion_top_k]

        # ------------------------------------------------------------
        # 4) Main MCTS loop
        # ------------------------------------------------------------
        start_time = time.time()
        iterations = 0

        while True:
            if self.iteration_limit is not None:
                if iterations >= self.iteration_limit:
                    break
            else:
                if (time.time() - start_time) >= self.time_limit:
                    break

            node = root

            # -------------------------
            # Selection
            # -------------------------
            while (not node.is_terminal()) and node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_constant)

            # -------------------------
            # Expansion
            # -------------------------
            if not node.is_terminal():
                if self.expansion_top_k is not None and node.parent is not None:
                    if len(node.untried_moves) > self.expansion_top_k:
                        node.untried_moves = node.untried_moves[:self.expansion_top_k]

                expanded = node.expand_one()
                if expanded is not None:
                    node = expanded

            # -------------------------
            # Simulation
            # -------------------------
            value_p1 = rollout(node.game, max_depth=self.rollout_depth)

            # -------------------------
            # Backpropagation
            # -------------------------
            while node is not None:
                node.visits += 1
                node.total_value_p1 += value_p1
                node = node.parent

            iterations += 1

        # ------------------------------------------------------------
        # 5) Fallback if no children were expanded
        # ------------------------------------------------------------
        if not root.children:
            fallback = ordered_legal_moves(root.game)[0]

            if return_stats:
                fallback_value_p1 = 0.5
                fallback_value_for_current = fallback_value_p1 if current_player == PopOutGame.P1 else (1.0 - fallback_value_p1)
                return fallback, {
                    "iterations": iterations,
                    "reason": "fallback",
                    "children": {
                        fallback: {
                            "visits": 1,
                            "value_for_current_player": fallback_value_for_current,
                            "value_for_p1": fallback_value_p1,
                        }
                    },
                }

            return fallback

        # ------------------------------------------------------------
        # 6) Final move choice:
        #    highest visit count, tie-break by value for current player
        # ------------------------------------------------------------
        best_child = max(
            root.children.values(),
            key=lambda child: (child.visits, child.q_for_player(current_player))
        )
        best_move = best_child.move

        if return_stats:
            stats = {
                "iterations": iterations,
                "reason": "search",
                "children": {
                    move: {
                        "visits": child.visits,
                        "value_for_current_player": round(child.q_for_player(current_player), 4),
                        "value_for_p1": round(child.total_value_p1 / child.visits, 4) if child.visits else 0.5,
                    }
                    for move, child in sorted(
                        root.children.items(),
                        key=lambda item: item[1].visits,
                        reverse=True
                    )
                },
            }
            return best_move, stats

        return best_move


# ---------------------------------------------------------------------
# Player wrapper for your interface
# ---------------------------------------------------------------------

class MCTSPlayer:
    """
    Drop-in player object compatible with your interface.py play_game loop.
    It does not inherit from Player to avoid circular imports.
    """

    def __init__(
        self,
        name: str,
        player_num: int,
        time_limit: float = 1.5,
        iteration_limit: Optional[int] = None,
        exploration_constant: float = 1.35,
        rollout_depth: int = 80,
        expansion_top_k: Optional[int] = None,
        verbose: bool = True,
        random_seed: Optional[int] = None,
    ):
        self.name = name
        self.player_num = player_num
        self.verbose = verbose

        self.engine = MCTS(
            time_limit=time_limit,
            iteration_limit=iteration_limit,
            exploration_constant=exploration_constant,
            rollout_depth=rollout_depth,
            expansion_top_k=expansion_top_k,
            random_seed=random_seed,
        )

    def get_move(self, game: PopOutGame) -> Optional[Move]:
        if game.game_over:
            return None

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        # Optional: if draw can be declared, you could choose to declare draw
        # in clearly bad positions. For now, we keep it simple and continue searching.

        move, stats = self.engine.search(game, return_stats=True)

        if self.verbose:
            print(f"\n{self.name} (Player {self.player_num}) chooses: {move[0].upper()} {move[1] + 1}")
            print(f"MCTS iterations: {stats['iterations']}")

            top_items = list(stats["children"].items())[:5]
            if top_items:
                print("Top candidate moves:")
                for m, info in top_items:
                    print(
                        f"  {m[0].upper()} {m[1] + 1} -> "
                        f"visits={info['visits']}, "
                        f"value={info['value_for_current_player']}"
                    )

        return move


# ---------------------------------------------------------------------
# Simple standalone quick test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    g = PopOutGame(6, 7)
    ai = MCTSPlayer(
        name="MCTS",
        player_num=1,
        time_limit=1.0,
        exploration_constant=1.35,
        rollout_depth=80,
        expansion_top_k=None,
        verbose=True,
    )

    for _ in range(10):
        if g.game_over:
            break
        move = ai.get_move(g)
        if move is None:
            break
        g.make_move(*move)
        print(g)
        print()