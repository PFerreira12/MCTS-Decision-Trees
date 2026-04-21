import random
import time
from typing import Tuple, Optional

from logic import PopOutGame
from mcts import MCTSPlayer


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

            drop_moves = [m[1] + 1 for m in moves if m[0] == 'drop']
            pop_moves = [m[1] + 1 for m in moves if m[0] == 'pop']

            print("\nLegal moves:")
            if drop_moves:
                print(f"  DROP: {drop_moves}")
            if pop_moves:
                print(f"  POP:  {pop_moves}")

            if not moves:
                print("No legal moves available!")
                return None

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
            if move_type not in ("drop", "pop"):
                print("Move type must be 'drop' or 'pop'")
                continue

            try:
                col_input = int(move_input[1])
                col = col_input - 1
            except ValueError:
                print("Column must be a number!")
                continue

            if (move_type, col) in moves:
                return (move_type, col)

            print(f"Invalid move: {move_type} {col_input}")
            print("Please introduce a valid move!")


class RandomAIPlayer(Player):
    """Simple random AI for easy mode."""

    def get_move(self, game: PopOutGame) -> Optional[Tuple[str, int]]:
        moves = game.get_legal_moves()
        if not moves:
            return None

        move = random.choice(moves)
        print(f"\n{self.name} (Player {self.player_num}) chooses: {move[0].upper()} {move[1] + 1}")
        time.sleep(1)
        return move


def clear_screen():
    """Clear terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


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


def show_ai_menu(player_label: str = "AI") -> str:
    """Show AI type selection."""
    print(f"\nSelect {player_label} type:")
    print("1. Easy (Random)")
    print("2. Hard (MCTS)")

    while True:
        choice = input("\nEnter choice (1-2): ").strip()
        if choice == '1':
            return "easy"
        if choice == '2':
            return "hard"
        print("Invalid choice.")


def show_mcts_strength_menu() -> float:
    """
    Return a time limit per move for the MCTS AI.
    Higher = stronger but slower.
    """
    print("\nSelect MCTS strength:")
    print("1. Fast   (0.5 sec/move)")
    print("2. Medium (1.5 sec/move)")
    print("3. Strong (3.0 sec/move)")

    while True:
        choice = input("\nEnter strength (1-3): ").strip()
        if choice == '1':
            return 0.5
        if choice == '2':
            return 1.5
        if choice == '3':
            return 3.0
        print("Invalid choice.")


def create_ai_player(name: str, player_num: int, ai_type: str) -> Player:
    """Factory for AI players."""
    if ai_type == "easy":
        return RandomAIPlayer(name, player_num)

    time_limit = show_mcts_strength_menu()
    return MCTSPlayer(
        name=name,
        player_num=player_num,
        time_limit=time_limit,
        exploration_constant=1.35,
        rollout_depth=80,
        expansion_top_k=None,
        verbose=True,
    )


def play_game(player1: Player, player2: Player, board_size: Tuple[int, int]):
    """Main game loop."""
    rows, cols = board_size
    game = PopOutGame(rows, cols)
    players = [None, player1, player2]

    print("\nStarting game...")
    time.sleep(1)

    while True:
        clear_screen()
        print(game)
        print()

        current_player = players[game.current_player]
        move = current_player.get_move(game)

        if move is None:
            if game.is_draw:
                print(f"\n{'=' * 50}")
                print("DRAW DECLARED!")
                print(f"{'=' * 50}")
            else:
                print(f"\n{'=' * 50}")
                print("No legal moves available.")
                print(f"{'=' * 50}")
            break

        game.make_move(move[0], move[1])

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

        board_size = (6, 7)

        if choice == 1:  # Human vs Human
            clear_screen()
            name1 = input("Enter name for Player 1: ").strip() or "Player 1"
            name2 = input("Enter name for Player 2: ").strip() or "Player 2"
            player1 = HumanPlayer(name1, 1)
            player2 = HumanPlayer(name2, 2)

        elif choice == 2:  # Human vs AI
            clear_screen()
            name1 = input("Enter your name: ").strip() or "Player"
            ai_type = show_ai_menu("AI")

            player1 = HumanPlayer(name1, 1)
            player2 = create_ai_player("AI", 2, ai_type)

        elif choice == 3:  # AI vs AI
            clear_screen()
            ai1_type = show_ai_menu("AI 1")
            print("\nSelect AI 2 settings:")
            ai2_type = show_ai_menu("AI 2")

            player1 = create_ai_player("AI 1", 1, ai1_type)
            player2 = create_ai_player("AI 2", 2, ai2_type)

        play_again = True
        while play_again:
            play_again = play_game(player1, player2, board_size)

        if not play_again:
            print("\nThanks for playing PopOut! Goodbye!")
            break


if __name__ == "__main__":
    main()
