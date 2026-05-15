"""
mcts.py — MCTS core: shared MCTSNode used by all strategy classes.

All four strategy classes (MCTS, MCTS_RAVE, MCTSTopK, MCTSWithHeuristics)
are defined in the notebook and inherit from this node.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
 
# ---------------------------------------------------------------------------
# Shared node class
# ---------------------------------------------------------------------------
 
class MCTSNode:
    """
    A node in the MCTS search tree.
 
    Attributes
    ----------
    state      : PopOutGame — game state at this node
    move       : the move that led to this node  (None for root)
    parent     : parent MCTSNode
    children   : list of child MCTSNode
    visits     : N(s)
    wins       : W(s)  — accumulated wins for the player who just moved
    untried    : moves not yet expanded
    player     : the player who *just moved* into this state
                 (i.e. whose perspective the wins are counted for)
    """
 
    __slots__ = (
        "state", "move", "parent", "children",
        "visits", "wins", "untried", "player",
        # RAVE-specific
        "rave_visits", "rave_wins",
        # Heuristic prior (used by MCTSWithHeuristics)
        "prior_wins",
    )
 
    def __init__(
        self,
        state,
        move: Optional[Tuple[str, int]] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state = state
        self.move = move
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.player: int = 3 - state.current_player  # player who just moved
        self.untried: List[Tuple[str, int]] = state.get_legal_moves()
        random.shuffle(self.untried)
 
        # RAVE statistics (keyed by move tuple)
        self.rave_visits: Dict[Tuple[str, int], int] = defaultdict(int)
        self.rave_wins: Dict[Tuple[str, int], float] = defaultdict(float)

        # Heuristic prior win count (set by MCTSWithHeuristics._expand; 0 otherwise)
        self.prior_wins: float = 0.0

    def uct_score(self, c: float = math.sqrt(2)) -> float:
        """UCT score blending real wins with the heuristic prior.

        prior_wins is 0.0 for Standard/RAVE/TopK nodes and a small positive
        value set by MCTSWithHeuristics._expand for heuristic nodes.
        The prior is added to the exploitation term only — it does NOT
        inflate self.visits, so the exploration term is unaffected.
        """
        if self.visits == 0:
            return float("inf")

        parent_visits = self.parent.visits if self.parent else self.visits

        exploitation = (self.wins + self.prior_wins) / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)

        return exploitation + exploration
 
    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0
 
    def is_terminal(self) -> bool:
        return self.state.game_over
 
    def best_child(self, c: float = math.sqrt(2)) -> "MCTSNode":
        return max(self.children, key=lambda ch: ch.uct_score(c))
 
    def most_visited_child(self) -> "MCTSNode":
        return max(self.children, key=lambda ch: ch.visits)
 
    def expand(self) -> "MCTSNode":
        move = self.untried.pop()
        new_state = self.state.copy()
        new_state.make_move(move[0], move[1])
        child = MCTSNode(new_state, move=move, parent=self)
        self.children.append(child)
        return child
 
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"MCTSNode(move={self.move}, visits={self.visits}, "
                f"wins={self.wins:.1f}, children={len(self.children)})")