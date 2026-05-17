import sys
import math
import pickle
import copy
from pathlib import Path

import pygame

from logic import PopOutGame
from mcts import MCTSPlayer

try:
    from id3_player import DEFAULT_MODEL_PATH, ID3Player
    ID3_AVAILABLE = Path(DEFAULT_MODEL_PATH).exists()
except ImportError:
    ID3_AVAILABLE = False


ROWS, COLS = 6, 7
CELL_SIZE = 84

BOARD_WIDTH = COLS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE

BOARD_PADDING_X = 60
TOP_PANEL_HEIGHT = 200
BOARD_TOP_GAP = 40
BOTTOM_PANEL_HEIGHT = 80

WIDTH = BOARD_WIDTH + (BOARD_PADDING_X * 2)
HEIGHT = TOP_PANEL_HEIGHT + BOARD_TOP_GAP + BOARD_HEIGHT + BOTTOM_PANEL_HEIGHT

FPS = 60
AI_THINK_DELAY_MS = 400

BEST_MCTS_PATH = Path(__file__).with_name("best_mcts_agent.pkl")

BG_COLOR = (12, 18, 32)
PANEL_COLOR = (24, 38, 68)
INNER_PANEL = (35, 55, 95)
PANEL_BORDER = (50, 90, 160)
BUTTON_HOVER = (45, 100, 210)

BOARD_COLOR = (35, 85, 190)
BOARD_BORDER = (90, 150, 255)

P1_COLORS = {"main": (235, 65, 80), "gloss": (255, 140, 150), "ring": (140, 20, 35)}
P2_COLORS = {"main": (255, 200, 50), "gloss": (255, 230, 150), "ring": (160, 110, 10)}

TEXT_PRIMARY = (245, 245, 250)
TEXT_SECONDARY = (160, 180, 210)
SUCCESS_GREEN = (80, 230, 140)
INFO_BLUE = (100, 200, 255)
PURPLE_ACCENT = (170, 130, 255)
RED_POP = (255, 100, 100)
GOLD_RESTART = (255, 210, 100)


class AnimationManager:
    def __init__(self):
        self.active_pieces = []
        self.on_complete_callback = None

    def trigger_drop(self, col, row, player, callback):
        bx, by = board_origin()
        target_y = by + row * CELL_SIZE + CELL_SIZE // 2
        start_y = by - CELL_SIZE

        self.on_complete_callback = callback
        self.active_pieces.append({
            "curr_y": start_y,
            "target_y": target_y,
            "col": col,
            "color": P1_COLORS if player == 1 else P2_COLORS,
            "vel": 0,
        })

    def trigger_pop(self, col, old_column_data, callback):
        bx, by = board_origin()
        self.on_complete_callback = callback

        for r in range(len(old_column_data) - 1):
            val = old_column_data[r]
            if val != 0:
                self.active_pieces.append({
                    "curr_y": by + r * CELL_SIZE + CELL_SIZE // 2,
                    "target_y": by + (r + 1) * CELL_SIZE + CELL_SIZE // 2,
                    "col": col,
                    "color": P1_COLORS if val == 1 else P2_COLORS,
                    "vel": 5,
                })

        if not self.active_pieces:
            callback = self.on_complete_callback
            self.on_complete_callback = None
            callback()

    def update(self):
        if not self.active_pieces:
            return

        finished = True

        for piece in self.active_pieces:
            piece["vel"] += 1.8
            piece["curr_y"] += piece["vel"]

            if piece["curr_y"] < piece["target_y"]:
                finished = False
            else:
                piece["curr_y"] = piece["target_y"]

        if finished:
            self.active_pieces = []
            if self.on_complete_callback:
                callback = self.on_complete_callback
                self.on_complete_callback = None
                callback()

    def is_animating(self):
        return len(self.active_pieces) > 0


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
pygame.display.set_caption("PopOut Pro")
clock = pygame.time.Clock()

def make_id3_player():
    try:
        agent = ID3Player(name="ID3", player_num=2)
    except TypeError:
        try:
            agent = ID3Player(name="ID3")
        except TypeError:
            agent = ID3Player()

    try:
        agent.name = "ID3"
    except Exception:
        pass

    try:
        agent.player_num = 2
    except Exception:
        pass

    return agent


def get_font(size, bold=False):
    return pygame.font.SysFont("Avenir Next", size, bold=bold)


title_f = get_font(52, True)
header_f = get_font(40, True)
button_f = get_font(24, True)
body_f = get_font(18)
label_f = get_font(13, True)
val_f = get_font(20, True)
small_bold = get_font(15, True)
coord_f = get_font(18, True)


def draw_panel(rect, fill, border, radius=12, width=2):
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=width, border_radius=radius)


def draw_text(text, x, y, font, color=TEXT_PRIMARY):
    surf = font.render(str(text), True, color)
    screen.blit(surf, (x, y))


def board_origin():
    return (WIDTH - BOARD_WIDTH) // 2, TOP_PANEL_HEIGHT + BOARD_TOP_GAP


def move_to_string(move):
    if move is None:
        return "—"
    return f"{move[0].upper()} Col {move[1] + 1}"


def fallback_mcts_player(name, player_num):
    return MCTSPlayer(
        name=name,
        player_num=player_num,
        time_limit=2.0,
        exploration_constant=1.2,
        rollout_depth=100,
        expansion_top_k=8,
        verbose=False,
    )


def load_best_mcts_agent(name="Best MCTS", player_num=1):
    if not BEST_MCTS_PATH.exists():
        print(f"[WARNING] {BEST_MCTS_PATH.name} not found. Using fallback MCTS.")
        return fallback_mcts_player(name, player_num)

    try:
        with open(BEST_MCTS_PATH, "rb") as file:
            agent = pickle.load(file)

        try:
            agent.name = name
        except Exception:
            pass

        try:
            agent.player_num = player_num
        except Exception:
            pass

        return agent

    except Exception as exc:
        print(f"[WARNING] Could not load {BEST_MCTS_PATH.name}: {exc}")
        print("[WARNING] Using fallback MCTS.")
        return fallback_mcts_player(name, player_num)


def clone_agent(agent, name, player_num):
    try:
        cloned = copy.deepcopy(agent)
    except Exception:
        cloned = load_best_mcts_agent(name, player_num)

    try:
        cloned.name = name
    except Exception:
        pass

    try:
        cloned.player_num = player_num
    except Exception:
        pass

    return cloned


def choose_ai_move(agent, game):
    name = getattr(agent, "name", "AI")

    if hasattr(agent, "engine") and hasattr(agent.engine, "search"):
        result = agent.engine.search(game, return_stats=True)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {"reason": name}

    if hasattr(agent, "search"):
        try:
            result = agent.search(game, return_stats=True)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return result, {"reason": name}
        except TypeError:
            move = agent.search(game)
            return move, {"reason": name}

    if hasattr(agent, "choose_move"):
        move = agent.choose_move(game)
        return move, {"reason": name}

    if hasattr(agent, "get_move"):
        move = agent.get_move(game)
        return move, {"reason": name}

    raise TypeError(f"Unsupported AI agent type: {type(agent)}")


def build_agents(mode):
    if mode == "hvh":
        return {1: "H", 2: "H"}

    if mode == "hvai":
        return {
            1: "H",
            2: load_best_mcts_agent("Best MCTS", 2),
        }

    if mode == "aivai":
        mcts_agent = load_best_mcts_agent("Best MCTS", 1)
        return {
            1: clone_agent(mcts_agent, "Best MCTS", 1),
            2: make_id3_player(),
        }

    raise ValueError(f"Unknown mode: {mode}")


class MenuButton:
    def __init__(self, x, y, w, h, text, action, p1_type, p2_type, enabled=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.enabled = enabled
        self.p1_type = p1_type
        self.p2_type = p2_type

    def draw_icon(self, cx, cy, player_type, color):
        if player_type == "H":
            pygame.draw.circle(screen, color, (cx, cy), 6)
        else:
            pygame.draw.circle(screen, color, (cx, cy), 5)
            for i in range(8):
                angle = i * (math.pi / 4)
                pygame.draw.line(
                    screen,
                    color,
                    (cx, cy),
                    (cx + math.cos(angle) * 7, cy + math.sin(angle) * 7),
                    2,
                )

    def draw(self, mouse_pos):
        hovered = self.rect.collidepoint(mouse_pos) and self.enabled

        fill = BUTTON_HOVER if hovered else PANEL_COLOR
        border = BOARD_BORDER if hovered else PANEL_BORDER
        text_color = TEXT_PRIMARY if self.enabled else TEXT_SECONDARY

        draw_panel(self.rect.move(0, 4), (10, 10, 20), (10, 10, 20), radius=15)
        draw_panel(self.rect, fill, border, radius=15)

        text_w = button_f.size(self.text)[0]
        draw_text(self.text, self.rect.centerx - text_w // 2, self.rect.y + 10, button_f, text_color)

        sub_y = self.rect.y + 48

        self.draw_icon(self.rect.centerx - 45, sub_y, self.p1_type, P1_COLORS["main"])
        draw_text("P1 vs", self.rect.centerx - 32, sub_y - 9, label_f, TEXT_SECONDARY)

        self.draw_icon(self.rect.centerx + 12, sub_y, self.p2_type, P2_COLORS["main"])
        draw_text("P2", self.rect.centerx + 24, sub_y - 9, label_f, TEXT_SECONDARY)


def draw_info_card(rect, label, value, color):
    draw_panel(rect, PANEL_COLOR, PANEL_BORDER, radius=12)
    pygame.draw.rect(screen, INNER_PANEL, rect.inflate(-10, -10), border_radius=8)

    label_text = label.upper()
    draw_text(label_text, rect.centerx - label_f.size(label_text)[0] // 2, rect.y + 12, label_f, TEXT_SECONDARY)
    draw_text(str(value), rect.centerx - val_f.size(str(value))[0] // 2, rect.y + 30, val_f, color)


def draw_controls_panel(rect):
    draw_panel(rect, PANEL_COLOR, PANEL_BORDER, radius=15)
    draw_text("COMMAND CENTER", rect.x + 18, rect.y + 12, small_bold, INFO_BLUE)

    controls = [
        ("LEFT CLICK", "DROP", SUCCESS_GREEN),
        ("RIGHT CLICK", "POP", RED_POP),
        ("R KEY", "RESET", GOLD_RESTART),
        ("M KEY", "MENU", INFO_BLUE),
        ("ESC", "QUIT", PURPLE_ACCENT),
    ]

    for i, (key, action, color) in enumerate(controls):
        row = pygame.Rect(rect.x + 12, rect.y + 40 + (i * 26), rect.width - 24, 22)
        pygame.draw.rect(screen, INNER_PANEL, row, border_radius=6)

        draw_text(key, row.x + 8, row.y + 3, label_f, TEXT_SECONDARY)
        draw_text(action, row.x + 115, row.y + 1, small_bold, color)


def draw_piece(cx, cy, color_dict):
    pygame.draw.circle(screen, color_dict["ring"], (cx, cy), 32)
    pygame.draw.circle(screen, color_dict["main"], (cx, cy), 28)
    pygame.draw.circle(screen, color_dict["gloss"], (cx - 8, cy - 8), 8)


def draw_board_and_pieces(game, anim_mgr):
    bx, by = board_origin()

    board_rect = pygame.Rect(bx - 15, by - 15, BOARD_WIDTH + 30, BOARD_HEIGHT + 30)
    draw_panel(board_rect, BOARD_COLOR, BOARD_BORDER, radius=20, width=4)

    anim_cols = {piece["col"] for piece in anim_mgr.active_pieces}

    for row in range(ROWS):
        for col in range(COLS):
            cx = bx + col * CELL_SIZE + CELL_SIZE // 2
            cy = by + row * CELL_SIZE + CELL_SIZE // 2
            val = game.board[row, col]

            if val != 0 and col not in anim_cols:
                draw_piece(cx, cy, P1_COLORS if val == 1 else P2_COLORS)
                pygame.draw.circle(screen, (45, 65, 110), (cx, cy), 30, 3)
            else:
                pygame.draw.circle(screen, (15, 25, 45), (cx, cy), 30)
                pygame.draw.circle(screen, (45, 65, 110), (cx, cy), 30, 3)

            if row == 0:
                draw_text(str(col + 1), cx - 5, by - 40, coord_f)

        draw_text(str(ROWS - row), bx - 40, by + row * CELL_SIZE + 32, coord_f)

    for piece in anim_mgr.active_pieces:
        cx = bx + piece["col"] * CELL_SIZE + CELL_SIZE // 2
        draw_piece(cx, int(piece["curr_y"]), piece["color"])


def animate_or_apply_move(game, anim_mgr, move):
    move_type, col = move

    if move_type == "draw":
        game.make_move(move_type, col)
        return "DRAW"

    callback = lambda t=move_type, c=col: game.make_move(t, c)

    if move_type == "drop":
        row = next(r for r in range(ROWS - 1, -1, -1) if game.board[r, col] == 0)
        anim_mgr.trigger_drop(col, row, game.current_player, callback)

    elif move_type == "pop":
        anim_mgr.trigger_pop(col, game.board[:, col].copy(), callback)

    return move_to_string(move)


def main():
    state = "menu"
    anim_mgr = AnimationManager()

    game = None
    agents = None
    last_move = "-"
    last_reason = "-"

    ai_pending = False
    ai_start_time = 0

    buttons = [
        MenuButton(WIDTH // 2 - 210, 260, 420, 72, "Human vs Human", "hvh", "H", "H"),
        MenuButton(WIDTH // 2 - 210, 345, 420, 72, "Human vs Best MCTS", "hvai", "H", "AI"),
        MenuButton(
            WIDTH // 2 - 210,
            430,
            420,
            72,
            "Best MCTS vs ID3 AI",
            "aivai",
            "AI",
            "AI",
            enabled=ID3_AVAILABLE,
        ),
    ]

    while True:
        mouse_pos = pygame.mouse.get_pos()
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_m:
                    state = "menu"
                    ai_pending = False

                if event.key == pygame.K_r and state == "game":
                    game = PopOutGame(ROWS, COLS)
                    anim_mgr = AnimationManager()
                    last_move = "-"
                    last_reason = "Reset"
                    ai_pending = False

            if event.type == pygame.MOUSEBUTTONDOWN and not anim_mgr.is_animating():
                if state == "menu":
                    for button in buttons:
                        if button.rect.collidepoint(event.pos) and button.enabled:
                            game = PopOutGame(ROWS, COLS)
                            agents = build_agents(button.action)
                            anim_mgr = AnimationManager()
                            state = "game"
                            last_move = "-"
                            last_reason = "Start"
                            ai_pending = False

                elif state == "game":
                    if game.game_over:
                        continue

                    if not isinstance(agents[game.current_player], str):
                        continue

                    bx, by = board_origin()
                    x, y = event.pos

                    if bx <= x <= bx + BOARD_WIDTH and by <= y <= by + BOARD_HEIGHT:
                        col = (x - bx) // CELL_SIZE

                        if event.button == 1:
                            move = ("drop", col)
                        elif event.button == 3:
                            move = ("pop", col)
                        else:
                            move = None

                        if move and move in game.get_legal_moves():
                            last_move = animate_or_apply_move(game, anim_mgr, move)
                            last_reason = "Human User"
                            ai_pending = False

        anim_mgr.update()

        if state == "game" and game is not None:
            if not game.game_over and not anim_mgr.is_animating():
                agent = agents[game.current_player]

                if not isinstance(agent, str):
                    if not ai_pending:
                        ai_pending = True
                        ai_start_time = now
                        last_reason = "AI thinking..."

                    elif now - ai_start_time >= AI_THINK_DELAY_MS:
                        move, stats = choose_ai_move(agent, game)

                        if move:
                            last_move = animate_or_apply_move(game, anim_mgr, move)
                            last_reason = stats.get("reason", getattr(agent, "name", "AI"))

                        ai_pending = False
                else:
                    ai_pending = False

        screen.fill(BG_COLOR)

        if state == "menu":
            draw_text("PopOut Pro", WIDTH // 2 - 140, 80, title_f)
            draw_text("Select Assignment Mode", WIDTH // 2 - 100, 160, body_f, TEXT_SECONDARY)

            for button in buttons:
                button.draw(mouse_pos)

            if not BEST_MCTS_PATH.exists():
                draw_text(
                    "best_mcts_agent.pkl not found — fallback MCTS will be used.",
                    WIDTH // 2 - 230,
                    530,
                    body_f,
                    GOLD_RESTART,
                )

            if not ID3_AVAILABLE:
                draw_text(
                    "ID3 model/player not found — MCTS vs ID3 disabled.",
                    WIDTH // 2 - 210,
                    560,
                    body_f,
                    RED_POP,
                )

        elif state == "game":
            draw_text("PopOut", 40, 30, header_f)

            if game.game_over:
                status_text = f"Winner: P{game.winner}" if game.winner else "Game Over: Draw"
                status_color = SUCCESS_GREEN
            else:
                status_text = f"Player {game.current_player}'s Turn"
                status_color = TEXT_PRIMARY

            draw_text(status_text, 40, 85, button_f, status_color)

            draw_info_card(pygame.Rect(40, 130, 135, 65), "Pieces", game.count_pieces(), INFO_BLUE)
            draw_info_card(pygame.Rect(190, 130, 135, 65), "Reps", game.get_repetition_count(), PURPLE_ACCENT)

            draw_controls_panel(pygame.Rect(WIDTH - 315, 15, 280, 185))
            draw_board_and_pieces(game, anim_mgr)

            footer = pygame.Rect(40, HEIGHT - 65, WIDTH - 80, 45)
            draw_panel(footer, PANEL_COLOR, PANEL_BORDER, radius=10)

            draw_text(f"Move: {last_move}", 60, HEIGHT - 52, body_f, TEXT_SECONDARY)
            draw_text(f"Source: {last_reason}", WIDTH // 2 + 20, HEIGHT - 52, body_f, PURPLE_ACCENT)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
