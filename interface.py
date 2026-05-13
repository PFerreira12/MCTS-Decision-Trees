import numpy as np
import random
import time
from typing import Tuple, Callable, Optional
from logic import PopOutGame


class Player:
    """Base class for players."""
    
    def __init__(self, name: str, player_num: int):
        self.name = name
        self.player_num = player_num
    
    def get_move(self, game: PopOutGame) -> Optional[Tuple[str, int]]:
        raise NotImplementedError


class HumanPlayer(Player):
    """Human player - gets input from terminal."""
    
    def get_move(self, game: PopOutGame) -> Optional[Tuple[str, int]]:
        while True:
            print(f"\n{self.name}'s turn (Player {self.player_num})")
            print(f"Pieces on board: {game.count_pieces()}")
            print(f"Repetition count: {game.get_repetition_count()}")
            
            moves = game.get_legal_moves()
            
            # Organize moves by type
            drop_moves = [m[1] + 1 for m in moves if m[0] == 'drop']  # Convert to 1-based display
            pop_moves = [m[1] + 1 for m in moves if m[0] == 'pop']    # Convert to 1-based display
            
            print("\nLegal moves:")
            if drop_moves:
                print(f"  DROP: {drop_moves}")
            if pop_moves:
                print(f"  POP:  {pop_moves}")
            
            if not moves:
                print("No legal moves available!")
                return None
            
            # Check if draw can be declared
            if game.can_declare_draw():
                reason = "(board full)" if game._is_board_full() else "(position repeated 3x)"
                draw_choice = input(f"\nDraw available {reason}. Declare draw? (y/n): ").strip().lower()
                if draw_choice == 'y':
                    game.declare_draw()
                    return None
            
            move_input = input("\nEnter move (e.g., 'drop 3' or 'pop 2'): ").strip().lower().split()
            
            if len(move_input) != 2:
                print("Invalid format. Use 'drop X' or 'pop X'")
                continue
            
            move_type = move_input[0]
            try:
                col_input = int(move_input[1])
                col = col_input - 1  # Convert from 1-based to 0-based index
            except ValueError:
                print("Column must be a number!")
                continue
            
            # Validate move
            if (move_type, col) in moves:
                return (move_type, col)
            else:
                print(f"Invalid move: {move_type} {col_input}")
                print(f"Please introduce a valid move!")


class AIPlayer(Player):
    """AI player - uses minimax or random strategy."""
    
    def __init__(self, name: str, player_num: int, difficulty: str = "hard"):
        super().__init__(name, player_num)
        self.difficulty = difficulty  # "easy" = random, "hard" = minimax
    
    def get_move(self, game: PopOutGame) -> Optional[Tuple[str, int]]:
        moves = game.get_legal_moves()
        
        if not moves:
            return None
        
        if self.difficulty == "easy":
            move = random.choice(moves)
        else:  # hard
            move = self._best_move(game)
        
        print(f"\n{self.name} (Player {self.player_num}) chooses: {move[0].upper()} {move[1]}")
        time.sleep(1)  # Pause for readability
        
        return move
    
    def _best_move(self, game: PopOutGame) -> Tuple[str, int]:
        """Minimax algorithm with alpha-beta pruning."""
        moves = game.get_legal_moves()
        best_move = moves[0]
        best_score = float('-inf')
        
        for move in moves:
            test_game = game.copy()
            test_game.make_move(move[0], move[1])
            score = -self._minimax(test_game, depth=4, alpha=float('-inf'), beta=float('inf'))
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(self, game: PopOutGame, depth: int, alpha: float, beta: float) -> float:
        """Minimax with alpha-beta pruning for Connect 4 evaluation."""
        # Terminal states
        if game.game_over:
            if game.winner == self.player_num:
                return 100000  # Win
            elif game.winner is not None:
                return -100000  # Loss
            else:
                return 0  # Draw
        
        if depth == 0:
            return self._evaluate_board(game)
        
        max_eval = float('-inf')
        for move in game.get_legal_moves():
            child_game = game.copy()
            child_game.make_move(move[0], move[1])
            eval_score = -self._minimax(child_game, depth - 1, -beta, -alpha)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        
        return max_eval
    
    def _evaluate_board(self, game: PopOutGame) -> float:
        """Heuristic evaluation of board position."""
        score = 0
        
        # Check for threats and opportunities
        for row in range(game.rows):
            for col in range(game.cols):
                if game.board[row, col] == self.player_num:
                    score += self._count_threats(game, row, col, self.player_num)
                elif game.board[row, col] == (3 - self.player_num):  # opponent
                    score -= self._count_threats(game, row, col, 3 - self.player_num) * 1.2
        
        return score
    
    def _count_threats(self, game: PopOutGame, row: int, col: int, player: int) -> float:
        """Count consecutive pieces and potentials from a position."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # H, V, Diag-right, Diag-left
        score = 0
        
        for dr, dc in directions:
            count = 1
            # Check forward
            r, c = row + dr, col + dc
            while 0 <= r < game.rows and 0 <= c < game.cols and game.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check backward
            r, c = row - dr, col - dc
            while 0 <= r < game.rows and 0 <= c < game.cols and game.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            # Score based on consecutive count
            if count >= 3:
                score += count ** 2
        
        return score


def clear_screen():
    """Clear terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def display_board(game: PopOutGame):
    """Display the game board nicely."""
    symbols = {0: '.', 1: 'X', 2: 'O'}
    print("\n" + "=" * 50)
    print("POPOUT GAME BOARD")
    print("=" * 50)
    
    # Column numbers
    print("    ", end="")
    for col in range(game.cols):
        print(f" {col} ", end="")
    print()
    print("    " + "-" * (game.cols * 2 - 1))
    
    # Board with row numbers (top to bottom visual)
    for row in range(game.rows):
        print(f" {row}  ", end="")
        for col in range(game.cols):
            print(f" {symbols[game.board[row, col]]} ", end="")
        print()
    print("=" * 50)


def show_main_menu() -> int:
    """Show main menu and get player choice."""
    clear_screen()
    print("\n" + "=" * 50)
    print("WELCOME TO POPOUT GAME".center(50))
    print("=" * 50)
    print("\nSelect game mode:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    print("4. Exit")
    print("-" * 50)
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def show_difficulty_menu() -> str:
    """Show difficulty selection."""
    print("\nSelect AI difficulty:")
    print("1. Easy (Random moves)")
    print("2. Hard (Smart moves)")
    
    while True:
        choice = input("\nEnter difficulty (1-2): ").strip()
        if choice == '1':
            return "easy"
        elif choice == '2':
            return "hard"
        print("Invalid choice.")


def play_game(player1: Player, player2: Player, board_size: Tuple[int, int]):
    """Main game loop."""
    rows, cols = board_size
    game = PopOutGame(rows, cols)
    players = [None, player1, player2]  # Index by player_num (1 or 2)
    move_count = 0
    
    print("\nStarting game...")
    time.sleep(1)
    
    while True:
        clear_screen()
        print(game)  # Use the game's __str__ method
        print()
        
        # Get move from current player
        current_player = players[game.current_player]
        move = current_player.get_move(game)
        
        if move is None:
            # Draw was declared or no legal moves
            if game.is_draw:
                print(f"\n{'=' * 50}")
                print("DRAW DECLARED!")
                print(f"{'=' * 50}")
            else:
                print(f"\n{'=' * 50}")
                print("No legal moves available.")
                print(f"{'=' * 50}")
            break
        
        # Make the move
        game.make_move(move[0], move[1])
        move_count += 1
        
        # Check if game is over
        if game.game_over:
            clear_screen()
            print(game)
            print()
            print(f"{'=' * 50}")
            if game.winner:
                winner = players[game.winner]
                print(f"GAME OVER! {winner.name} (Player {game.winner}) WINS!")
            elif game.is_draw:
                print("GAME OVER! DRAW!")
            print(f"{'=' * 50}")
            break
    
    # Ask to play again
    while True:
        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again in ['y', 'n']:
            return play_again == 'y'
        print("Please enter 'y' or 'n'")


def main():
    """Main program loop."""
    while True:
        choice = show_main_menu()
        
        if choice == 4:
            print("\nThanks for playing PopOut! Goodbye!")
            break
        
        # Get board size
        board_size = (6,7)
        
        # Setup players based on mode
        if choice == 1:  # Human vs Human
            clear_screen()
            name1 = input("Enter name for Player 1: ").strip() or "Player 1"
            name2 = input("Enter name for Player 2: ").strip() or "Player 2"
            player1 = HumanPlayer(name1, 1)
            player2 = HumanPlayer(name2, 2)
        
        elif choice == 2:  # Human vs AI
            clear_screen()
            name1 = input("Enter your name: ").strip() or "Player"
            difficulty = show_difficulty_menu()
            player1 = HumanPlayer(name1, 1)
            player2 = AIPlayer("AI", 2, difficulty)
        
        elif choice == 3:  # AI vs AI
            clear_screen()
            diff1 = show_difficulty_menu()
            print("\nSelect AI 2 difficulty:")
            diff2 = show_difficulty_menu()
            player1 = AIPlayer("AI 1", 1, diff1)
            player2 = AIPlayer("AI 2", 2, diff2)
        
        # Play the game
        play_again = True
        while play_again:
            play_again = play_game(player1, player2, board_size)
        
        # If user chose not to play again, exit the program
        if not play_again:
            print("\nThanks for playing PopOut! Goodbye!")
            break


if __name__ == "__main__":
    main()