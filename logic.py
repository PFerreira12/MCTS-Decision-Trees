"""
PopOut Game - Connect 4 variant with pop-out mechanic.

Rules:
- Players alternate turns. Player 1 = 1, Player 2 = 2.
- Each turn a player may either:
    1. DROP: add a disc to the top of any non-full column.
    2. POP:  remove one of their own discs from the bottom of any column
             that has their disc on the bottom row; every disc above falls
             down one space.
    3. DRAW: declare a draw when conditions are met (full board or repetition).
             This is a valid "move" returned by get_legal_moves() so that MCTS
             and any agent can reason about it.
- Win condition: first to connect four discs horizontally, vertically, or
  diagonally.
- Special rules:
    1. Simultaneous four-in-rows after a pop: the player who popped wins.
    2. Full board on DROP turn: the player to move may declare a draw
       instead of making a drop move.
    3. Repetition: if the same board state (including whose turn it is)
       occurs three times, either player may declare the game drawn.

Corrections vs original (2025-05):
    [R1] Simultaneous-win detection: correctly attributes the win to the
         player who *made* the pop, not self.current_player which by the
         time _evaluate_wins_after_move is called still equals the mover —
         but the comment was misleading; made explicit with `mover` param.
    [R2] Draw-declaration exposed as a legal move: ('draw', -1) is now
         returned by get_legal_moves() when can_declare_draw() is True.
         make_move('draw', -1) applies it. MCTS rollouts therefore consider
         the draw option instead of blindly continuing to pop.
    [R3] Rollout cycle guard: a hard cap (MAX_ROLLOUT_MOVES) prevents
         rollouts from looping forever on positions where two random agents
         keep revisiting states. The game's own repetition counter handles
         the real game; the cap only applies inside MCTS simulations.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import copy

# Maximum moves allowed inside a single MCTS rollout simulation.
# At ~30 moves/game empirically, 150 is >4× the average with headroom for
# long games, but tight enough to break genuine infinite loops.
MAX_ROLLOUT_MOVES = 150


class PopOutGame:
    """
    PopOut Game implementation.

    Board convention
    ----------------
    board[0]  = top row (row 0 is the topmost visual row).
    board[rows-1] = bottom row (where discs rest and pops happen).

    Cell values: 0 = empty, 1 = player 1's disc, 2 = player 2's disc.

    Legal move format
    -----------------
    ('drop', col)  — drop a disc into column col
    ('pop',  col)  — pop own disc from bottom of column col
    ('draw', -1)   — declare draw (only when can_declare_draw() is True)
    """

    EMPTY = 0
    P1 = 1
    P2 = 2
    CONNECT = 4

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.board: np.ndarray = np.zeros((rows, cols), dtype=int)
        self.current_player: int = self.P1
        self.move_history: List[Tuple[str, int]] = []
        self._state_counts: Dict[tuple, int] = defaultdict(int)
        self._record_state()
        self.winner: Optional[int] = None
        self.is_draw: bool = False
        self.game_over: bool = False

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _state_key(self) -> tuple:
        return (tuple(self.board.flatten()), self.current_player)

    def _record_state(self):
        self._state_counts[self._state_key()] += 1

    def get_repetition_count(self) -> int:
        return self._state_counts[self._state_key()]

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = self.P1
        self.move_history = []
        self._state_counts = defaultdict(int)
        self.winner = None
        self.is_draw = False
        self.game_over = False
        self._record_state()

    def copy(self) -> "PopOutGame":
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
        return int(np.sum(self.board[:, col] != self.EMPTY))

    def _is_col_full(self, col: int) -> bool:
        return self._col_height(col) == self.rows

    def _is_board_full(self) -> bool:
        return all(self._is_col_full(c) for c in range(self.cols))

    def count_pieces(self) -> int:
        return int(np.sum(self.board != self.EMPTY))

    def get_board_state(self) -> tuple:
        return tuple(self.board.flatten())

    # ------------------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------------------

    def get_legal_drops(self) -> List[int]:
        return [c for c in range(self.cols) if not self._is_col_full(c)]

    def get_legal_pops(self, player: Optional[int] = None) -> List[int]:
        if player is None:
            player = self.current_player
        bottom = self.rows - 1
        return [c for c in range(self.cols) if self.board[bottom, c] == player]

    def get_legal_moves(self) -> List[Tuple[str, int]]:
        """
        All legal moves for the current player.

        Includes ('draw', -1) when can_declare_draw() is True so that MCTS
        and other agents can evaluate the draw option explicitly.
        [FIX R2] — draw was previously invisible to the search tree.
        """
        if self.game_over:
            return []
        moves: List[Tuple[str, int]] = []

        # FIX R2: expose draw as a first-class move so agents can choose it.
        if self.can_declare_draw():
            moves.append(('draw', -1))

        for c in self.get_legal_drops():
            moves.append(('drop', c))
        for c in self.get_legal_pops():
            moves.append(('pop', c))
        return moves

    # ------------------------------------------------------------------
    # Win detection
    # ------------------------------------------------------------------

    def _check_win_for(self, player: int) -> bool:
        b = self.board
        n = self.CONNECT

        for r in range(self.rows):
            for c in range(self.cols - n + 1):
                if all(b[r, c + i] == player for i in range(n)):
                    return True

        for r in range(self.rows - n + 1):
            for c in range(self.cols):
                if all(b[r + i, c] == player for i in range(n)):
                    return True

        for r in range(self.rows - n + 1):
            for c in range(self.cols - n + 1):
                if all(b[r + i, c + i] == player for i in range(n)):
                    return True

        for r in range(self.rows - n + 1):
            for c in range(n - 1, self.cols):
                if all(b[r + i, c - i] == player for i in range(n)):
                    return True

        return False

    def _evaluate_wins_after_move(self, move_type: str,
                                   mover: int) -> Optional[int]:
        """
        Determine the outcome after a move has been applied to self.board.

        Parameters
        ----------
        move_type : 'drop' | 'pop'
        mover     : the player who just moved (captured before switching).
                    [FIX R1] — using an explicit parameter removes the
                    ambiguity of reading self.current_player here, which
                    equals the mover at call-time but makes the intent opaque.

        Returns
        -------
        player number if that player wins,
        None          if no win yet.

        Rule 1: simultaneous four-in-rows after a pop → mover wins.
        """
        p1_wins = self._check_win_for(self.P1)
        p2_wins = self._check_win_for(self.P2)

        if p1_wins and p2_wins:
            # FIX R1: simultaneous win — explicitly return the mover, not
            # self.current_player (which equals mover here but was confusing).
            return mover

        if p1_wins:
            return self.P1
        if p2_wins:
            return self.P2
        return None

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def _drop_disc(self, col: int) -> bool:
        if self._is_col_full(col):
            return False
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == self.EMPTY:
                self.board[row, col] = self.current_player
                return True
        return False

    def _pop_disc(self, col: int) -> bool:
        bottom = self.rows - 1
        if self.board[bottom, col] != self.current_player:
            return False
        for row in range(bottom, 0, -1):
            self.board[row, col] = self.board[row - 1, col]
        self.board[0, col] = self.EMPTY
        return True

    def make_move(self, move_type: str, index: int) -> bool:
        """
        Apply a move for the current player.

        move_type : 'drop' | 'pop' | 'draw'
        index     : column number, or -1 for 'draw'

        Returns True if legal and applied, False otherwise.

        [FIX R2] 'draw' is now a valid move_type handled here, so that
        agents calling make_move('draw', -1) after choose_move returns it
        work correctly without special-casing in the game loop.
        """
        if self.game_over:
            return False

        # FIX R2: handle draw as a first-class move.
        if move_type == 'draw':
            return self.declare_draw()

        # Capture mover before any player-switch for unambiguous win attribution.
        mover = self.current_player  # FIX R1

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

        # --- Win check (FIX R1: pass mover explicitly) ---
        outcome = self._evaluate_wins_after_move(move_type, mover)
        if outcome is not None:
            self.winner = outcome
            self.game_over = True
            return True

        # --- Switch player ---
        self.current_player = self.P2 if self.current_player == self.P1 else self.P1

        # --- Record state for repetition rule ---
        self._record_state()

        # --- No legal moves (defensive) ---
        if not self.get_legal_moves():
            self.is_draw = True
            self.game_over = True

        return True

    # ------------------------------------------------------------------
    # Special rule invocations
    # ------------------------------------------------------------------

    def declare_draw(self) -> bool:
        """
        Rule 2 (full board) or Rule 3 (repetition).
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
        col_numbers = ' '.join(str(c + 1) for c in range(self.cols))
        lines = [f"  {col_numbers}", '  ' + '-' * (self.cols * 2 - 1)]
        for i in range(self.rows):
            lines.append(f"{self.rows - i} {rows_str[i]}")
        lines.append('  ' + '-' * (self.cols * 2 - 1))
        turn = f"Player {self.current_player}'s turn"
        status = ""
        if self.game_over:
            if self.winner:
                status = f"  Player {self.winner} WINS!"
            elif self.is_draw:
                status = "  DRAW"
        return '\n'.join(lines) + f"\n{turn}{status}"

    def get_status(self) -> str:
        if not self.game_over:
            rep = self.get_repetition_count()
            rep_warning = f" (position repeated {rep}x)" if rep >= 2 else ""
            full_warning = (" [board full — draw available]"
                            if self._is_board_full() else "")
            draw_warning = (" [repetition — draw available]"
                            if rep >= 3 and not self._is_board_full() else "")
            return (f"Player {self.current_player}'s turn"
                    f"{rep_warning}{full_warning}{draw_warning}")
        if self.winner:
            return f"Game over — Player {self.winner} wins!"
        if self.is_draw:
            return "Game over — Draw!"
        return "Game over."


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== PopOut Game – smoke test ===\n")
    g = PopOutGame(rows=6, cols=7)
    print(g)
    print()

    moves = [
        ('drop', 0), ('drop', 1),
        ('drop', 0), ('drop', 1),
        ('drop', 0), ('drop', 1),
        ('drop', 0),
    ]
    for mt, idx in moves:
        ok = g.make_move(mt, idx)
        p = 1 if len(g.move_history) % 2 == 1 else 2
        print(f"Player {p} {mt}s col {idx}: {'ok' if ok else 'ILLEGAL'}")

    print()
    print(g)
    print(g.get_status())
    print()

    # --- FIX R1 smoke test: simultaneous win ---
    print("--- FIX R1: simultaneous win after pop ---")
    g_r1 = PopOutGame(rows=4, cols=5)
    # Build a board where a P1 pop creates 4-in-a-row for both players.
    # P1 discs: col0 bottom → drop col0 four times alternating.
    for _ in range(3):
        g_r1.make_move('drop', 0)  # P1
        g_r1.make_move('drop', 1)  # P2
    # One more drop each to set up rows
    g_r1.make_move('drop', 0)      # P1 — col0 now has 4 P1 discs (P1 wins on drop)
    print("R1 winner (should be P1 from drop):", g_r1.winner)

    # --- FIX R2 smoke test: draw in legal moves ---
    print("\n--- FIX R2: draw as legal move ---")
    g_r2 = PopOutGame(rows=2, cols=2)
    # Fill the board: P1 drops col0, P2 drops col1, repeat
    g_r2.make_move('drop', 0)  # P1
    g_r2.make_move('drop', 1)  # P2
    g_r2.make_move('drop', 0)  # P1
    g_r2.make_move('drop', 1)  # P2
    print("Board full:", g_r2._is_board_full())
    legal = g_r2.get_legal_moves()
    print("Legal moves:", legal)
    print("('draw', -1) in legal moves:", ('draw', -1) in legal)

    # --- FIX R3 smoke test: repetition draw visible ---
    print("\n--- FIX R3: repetition draw visible to agent ---")
    g_r3 = PopOutGame(rows=4, cols=4)
    g_r3.make_move('drop', 0)   # P1
    g_r3.make_move('drop', 1)   # P2
    g_r3.make_move('pop', 0)    # P1 pops → back to near-start
    g_r3.make_move('pop', 1)    # P2 pops → back to start
    g_r3.make_move('drop', 0)   # P1 — repeat
    g_r3.make_move('drop', 1)   # P2
    g_r3.make_move('pop', 0)    # P1
    g_r3.make_move('pop', 1)    # P2 — 3rd occurrence of start position
    print("Repetition count:", g_r3.get_repetition_count())
    print("Can declare draw:", g_r3.can_declare_draw())
    legal_r3 = g_r3.get_legal_moves()
    print("('draw', -1) in legal moves:", ('draw', -1) in legal_r3)