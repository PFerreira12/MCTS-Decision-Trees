"""
PopOut Game - Connect 4 variant with pop-out mechanic.

Rules:
- Players alternate turns. Player 1 = 1, Player 2 = 2.
- Each turn a player may either:
    1. DROP: add a disc to the top of any non-full column.
    2. POP:  remove one of their own discs from the bottom of any column
             that has their disc on the bottom row; every disc above falls
             down one space.
- Win condition: first to connect four discs horizontally, vertically, or
  diagonally.
- Special rules:
    1. Simultaneous four-in-rows after a pop: the player who popped wins.
    2. Full board on DROP turn: the player to move may declare a draw
       instead of making a drop move.
    3. Repetition: if the same board state (including whose turn it is)
       occurs three times, either player may declare the game drawn.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import copy


class PopOutGame:
    """
    PopOut Game implementation.

    Board convention
    ----------------
    board[0]  = top row (row 0 is the topmost visual row).
    board[rows-1] = bottom row (where discs rest and pops happen).

    Cell values: 0 = empty, 1 = player 1's disc, 2 = player 2's disc.
    """

    EMPTY = 0
    P1 = 1
    P2 = 2
    CONNECT = 4  # number in a row to win

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.board: np.ndarray = np.zeros((rows, cols), dtype=int)
        self.current_player: int = self.P1
        self.move_history: List[Tuple[str, int]] = []
        # Maps hashable board-state+player to occurrence count for repetition rule.
        self._state_counts: Dict[tuple, int] = defaultdict(int)
        self._record_state()
        self.winner: Optional[int] = None   # set once game is decided
        self.is_draw: bool = False
        self.game_over: bool = False

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _state_key(self) -> tuple:
        """Hashable key that includes board + whose turn it is."""
        return (tuple(self.board.flatten()), self.current_player)

    def _record_state(self):
        self._state_counts[self._state_key()] += 1

    def get_repetition_count(self) -> int:
        """Return how many times the current position has occurred."""
        return self._state_counts[self._state_key()]

    def reset(self):
        """Reset everything to the initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = self.P1
        self.move_history = []
        self._state_counts = defaultdict(int)
        self.winner = None
        self.is_draw = False
        self.game_over = False
        self._record_state()

    def copy(self) -> "PopOutGame":
        """Deep copy of the full game state (including history/counts)."""
        new = PopOutGame.__new__(PopOutGame)
        new.rows = self.rows
        new.cols = self.cols
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.move_history = self.move_history.copy()
        new._state_counts = defaultdict(int, self._state_counts)
        new.winner = self.winner
        new.is_draw = self.is_draw
        new.game_over = self.game_over
        return new

    # ------------------------------------------------------------------
    # Board queries
    # ------------------------------------------------------------------

    def _col_height(self, col: int) -> int:
        """Number of discs in a column (discs sit at the bottom)."""
        return int(np.sum(self.board[:, col] != self.EMPTY))

    def _is_col_full(self, col: int) -> bool:
        return self._col_height(col) == self.rows

    def _is_board_full(self) -> bool:
        return all(self._is_col_full(c) for c in range(self.cols))

    def count_pieces(self) -> int:
        """Total discs on board."""
        return int(np.sum(self.board != self.EMPTY))

    def get_board_state(self) -> tuple:
        return tuple(self.board.flatten())

    # ------------------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------------------

    def get_legal_drops(self) -> List[int]:
        """Columns where a disc can be dropped from the top."""
        return [c for c in range(self.cols) if not self._is_col_full(c)]

    def get_legal_pops(self, player: Optional[int] = None) -> List[int]:
        """
        Columns where the current player (or given player) can pop.
        A pop is legal when the bottom cell of that column belongs to the player.
        """
        if player is None:
            player = self.current_player
        bottom = self.rows - 1
        return [c for c in range(self.cols) if self.board[bottom, c] == player]

    def get_legal_moves(self) -> List[Tuple[str, int]]:
        """
        All legal moves for the current player.
        Returns list of ('drop', col) or ('pop', col).
        """
        moves: List[Tuple[str, int]] = []
        for c in self.get_legal_drops():
            moves.append(('drop', c))
        for c in self.get_legal_pops():
            moves.append(('pop', c))
        return moves

    # ------------------------------------------------------------------
    # Win detection
    # ------------------------------------------------------------------

    def _check_win_for(self, player: int) -> bool:
        """Return True if *player* has four in a row on the current board."""
        b = self.board
        n = self.CONNECT

        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - n + 1):
                if all(b[r, c + i] == player for i in range(n)):
                    return True

        # Vertical
        for r in range(self.rows - n + 1):
            for c in range(self.cols):
                if all(b[r + i, c] == player for i in range(n)):
                    return True

        # Diagonal down-right
        for r in range(self.rows - n + 1):
            for c in range(self.cols - n + 1):
                if all(b[r + i, c + i] == player for i in range(n)):
                    return True

        # Diagonal down-left
        for r in range(self.rows - n + 1):
            for c in range(n - 1, self.cols):
                if all(b[r + i, c - i] == player for i in range(n)):
                    return True

        return False

    def _evaluate_wins_after_move(self, move_type: str) -> Optional[int]:
        """
        After a move has been applied to self.board, determine the outcome.

        Returns:
            player number if that player wins,
            0             if it's a draw (both four-in-rows after a pop, which
                          cannot happen — rule 1 says popper wins; returned 0
                          is only used internally),
            None          if no win yet.

        Rule 1: if a pop creates four-in-rows for both players simultaneously,
                the player who popped wins.
        """
        p1_wins = self._check_win_for(self.P1)
        p2_wins = self._check_win_for(self.P2)

        if p1_wins and p2_wins:
            # Simultaneous: only possible after a pop — popper wins.
            # The *previous* player just moved (we haven't switched yet).
            return self.current_player  # still the mover at call time

        if p1_wins:
            return self.P1
        if p2_wins:
            return self.P2
        return None

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def _drop_disc(self, col: int) -> bool:
        """
        Drop current player's disc into *col*.
        Disc falls to the lowest empty row in that column.
        Returns False if column is full.
        """
        if self._is_col_full(col):
            return False
        # Find lowest empty row (gravity: bottom = rows-1)
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == self.EMPTY:
                self.board[row, col] = self.current_player
                return True
        return False

    def _pop_disc(self, col: int) -> bool:
        """
        Remove current player's disc from the bottom of *col*.
        Every disc above falls down one space.
        Returns False if the pop is illegal.
        """
        bottom = self.rows - 1
        if self.board[bottom, col] != self.current_player:
            return False
        # Shift everything above the bottom down by one row.
        for row in range(bottom, 0, -1):
            self.board[row, col] = self.board[row - 1, col]
        self.board[0, col] = self.EMPTY  # top cell becomes empty
        return True

    def make_move(self, move_type: str, index: int) -> bool:
        """
        Apply a move for the current player.

        move_type : 'drop' | 'pop'
        index     : column number

        Returns True if the move was legal and applied, False otherwise.
        Updates self.winner / self.is_draw / self.game_over as appropriate.
        """
        if self.game_over:
            return False

        if move_type == 'drop':
            if index not in self.get_legal_drops():
                return False
            self._drop_disc(index)

        elif move_type == 'pop':
            if index not in self.get_legal_pops():
                return False
            self._pop_disc(index)

        else:
            return False

        self.move_history.append((move_type, index))

        # --- Win check ---
        outcome = self._evaluate_wins_after_move(move_type)
        if outcome is not None:
            self.winner = outcome
            self.game_over = True
            return True

        # --- Switch player ---
        self.current_player = self.P2 if self.current_player == self.P1 else self.P1

        # --- Record state for repetition rule ---
        self._record_state()

        # --- Check for no legal moves (shouldn't normally happen in PopOut
        #     because a player can always pop if they have a disc anywhere,
        #     but handle defensively) ---
        if not self.get_legal_moves():
            self.is_draw = True
            self.game_over = True

        return True

    # ------------------------------------------------------------------
    # Special rule invocations (called explicitly by game controller/UI)
    # ------------------------------------------------------------------

    def declare_draw(self) -> bool:
        """
        Rule 2 (full board) or Rule 3 (repetition):
        Either player may call this when conditions are met.
        Returns True if the declaration is valid, False otherwise.
        """
        if self.game_over:
            return False
        if self._is_board_full() or self.get_repetition_count() >= 3:
            self.is_draw = True
            self.game_over = True
            return True
        return False

    def can_declare_draw(self) -> bool:
        """
        True if a draw can currently be declared (rules 2 or 3).
        """
        if self.game_over:
            return False
        return self._is_board_full() or self.get_repetition_count() >= 3

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        symbols = {self.EMPTY: '.', self.P1: 'X', self.P2: 'O'}
        rows_str = []
        for r in range(self.rows):
            rows_str.append(' '.join(symbols[v] for v in self.board[r]))
        col_numbers = ' '.join(str(c+1) for c in range(self.cols))
        header = f"  {col_numbers}"
        separator = '  ' + '-' * (self.cols * 2 - 1)
        board_str = '\n'.join(f"{self.rows - i} {rows_str[i]}"
                              for i in range(self.rows))
        # Flip row labels so bottom row = 1 (more intuitive for display).
        lines = [header, separator]
        for i in range(self.rows):
            row_label = self.rows - i  # 1 at bottom
            lines.append(f"{row_label} {rows_str[i]}")
        lines.append(separator)
        turn = f"Player {self.current_player}'s turn"
        status = ""
        if self.game_over:
            if self.winner:
                status = f"  Player {self.winner} WINS!"
            elif self.is_draw:
                status = "  DRAW"
        return '\n'.join(lines) + f"\n{turn}{status}"

    def get_status(self) -> str:
        """Human-readable game status."""
        if not self.game_over:
            rep = self.get_repetition_count()
            rep_warning = f" (position repeated {rep}x)" if rep >= 2 else ""
            full_warning = " [board full — draw available]" if self._is_board_full() else ""
            return (f"Player {self.current_player}'s turn"
                    f"{rep_warning}{full_warning}")
        if self.winner:
            return f"Game over — Player {self.winner} wins!"
        if self.is_draw:
            return "Game over — Draw!"
        return "Game over."


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np  # ensure numpy available for standalone run

    print("=== PopOut Game – smoke test ===\n")
    g = PopOutGame(rows=6, cols=7)
    print(g)
    print()

    # Simulate a sequence of drops for both players
    moves = [
        ('drop', 0), ('drop', 1),
        ('drop', 0), ('drop', 1),
        ('drop', 0), ('drop', 1),
        ('drop', 0),  # Player 1 should win (4 in col 0)
    ]
    for mt, idx in moves:
        ok = g.make_move(mt, idx)
        print(f"Player {'1' if len(g.move_history) % 2 == 1 else '2'} "
              f"{mt}s col {idx}: {'ok' if ok else 'ILLEGAL'}")

    print()
    print(g)
    print(g.get_status())
    print()

    # Test pop mechanic
    print("--- Pop mechanic test ---")
    g2 = PopOutGame(rows=4, cols=5)
    g2.make_move('drop', 0)  # P1
    g2.make_move('drop', 0)  # P2
    g2.make_move('drop', 0)  # P1
    print("Before pop:\n", g2)
    print("Legal pops for P2:", g2.get_legal_pops())
    g2.make_move('pop', 0)   # P2 pops bottom of col 0 (P2's disc)
    print("\nAfter P2 pops col 0:\n", g2)

    # Test repetition detection
    print("\n--- Repetition rule test ---")
    g3 = PopOutGame(rows=4, cols=4)
    g3.make_move('drop', 0)  # P1
    g3.make_move('drop', 1)  # P2
    # Pop back to re-create the same position (simplified)
    print("Repetition count:", g3.get_repetition_count())
    print("Can declare draw:", g3.can_declare_draw())