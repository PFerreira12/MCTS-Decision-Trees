import sys
import pygame
import math
from logic import PopOutGame
from mcts import MCTSPlayer

# Try to import the future ID3 agent
try:
    from id3_player import ID3Player
    ID3_AVAILABLE = True
except ImportError:
    ID3_AVAILABLE = False

# --------------------------------------------------
# Configuration & Layout
# --------------------------------------------------
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

# --- Premium Palette ---
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

# --------------------------------------------------
# Animation Engine
# --------------------------------------------------
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
            "curr_y": start_y, "target_y": target_y, 
            "col": col, "color": P1_COLORS if player == 1 else P2_COLORS, "vel": 0
        })

    def trigger_pop(self, col, old_column_data, callback):
        bx, by = board_origin()
        self.on_complete_callback = callback
        for r in range(len(old_column_data)-1):
            val = old_column_data[r]
            if val != 0:
                self.active_pieces.append({
                    "curr_y": by + r * CELL_SIZE + CELL_SIZE // 2,
                    "target_y": by + (r + 1) * CELL_SIZE + CELL_SIZE // 2,
                    "col": col, "color": P1_COLORS if val == 1 else P2_COLORS, "vel": 5
                })

    def update(self):
        if not self.active_pieces: return
        
        finished = True
        for p in self.active_pieces:
            p["vel"] += 1.8 
            p["curr_y"] += p["vel"]
            if p["curr_y"] < p["target_y"]:
                finished = False
            else:
                p["curr_y"] = p["target_y"]

        if finished:
            self.active_pieces = []
            if self.on_complete_callback:
                self.on_complete_callback()
                self.on_complete_callback = None

    def is_animating(self):
        return len(self.active_pieces) > 0

# --------------------------------------------------
# UI Components
# --------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PopOut Pro")
clock = pygame.time.Clock()

def get_font(size, bold=False):
    return pygame.font.SysFont("Avenir Next", size, bold=bold)

title_f, header_f, button_f = get_font(52, True), get_font(40, True), get_font(24, True)
body_f, label_f, val_f = get_font(18), get_font(13, True), get_font(20, True)
small_bold, coord_f = get_font(15, True), get_font(18, True)

def draw_panel(rect, fill, border, radius=12, width=2):
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=width, border_radius=radius)

def board_origin():
    return (WIDTH - BOARD_WIDTH) // 2, TOP_PANEL_HEIGHT + BOARD_TOP_GAP

def draw_text(t, x, y, f, c=TEXT_PRIMARY):
    s = f.render(t, True, c)
    screen.blit(s, (x, y))

class MenuButton:
    def __init__(self, x, y, w, h, text, action, p1_type, p2_type, enabled=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.text, self.action, self.enabled = text, action, enabled
        self.p1_type, self.p2_type = p1_type, p2_type

    def draw_icon(self, cx, cy, p_type, color):
        if p_type == "H": pygame.draw.circle(screen, color, (cx, cy), 6)
        else:
            pygame.draw.circle(screen, color, (cx, cy), 5)
            for i in range(8):
                ang = i * (math.pi / 4)
                pygame.draw.line(screen, color, (cx, cy), (cx+math.cos(ang)*7, cy+math.sin(ang)*7), 2)

    def draw(self, m_pos):
        hover = self.rect.collidepoint(m_pos) and self.enabled
        fill = BUTTON_HOVER if hover else PANEL_COLOR
        draw_panel(self.rect.move(0, 4), (10, 10, 20), (10, 10, 20), radius=15)
        draw_panel(self.rect, fill, BOARD_BORDER if hover else PANEL_BORDER, radius=15)
        draw_text(self.text, self.rect.centerx - button_f.size(self.text)[0]//2, self.rect.y + 10, button_f, TEXT_PRIMARY if self.enabled else TEXT_SECONDARY)
        sub_y = self.rect.y + 48 
        self.draw_icon(self.rect.centerx - 45, sub_y, self.p1_type, P1_COLORS["main"])
        draw_text("P1 vs", self.rect.centerx - 32, sub_y - 9, label_f, TEXT_SECONDARY)
        self.draw_icon(self.rect.centerx + 12, sub_y, self.p2_type, P2_COLORS["main"])
        draw_text("P2", self.rect.centerx + 24, sub_y - 9, label_f, TEXT_SECONDARY)

def draw_info_card(rect, label, value, color):
    draw_panel(rect, PANEL_COLOR, PANEL_BORDER, radius=12)
    pygame.draw.rect(screen, INNER_PANEL, rect.inflate(-10, -10), border_radius=8)
    draw_text(label.upper(), rect.centerx - label_f.size(label.upper())[0]//2, rect.y + 12, label_f, TEXT_SECONDARY)
    draw_text(str(value), rect.centerx - val_f.size(str(value))[0]//2, rect.y + 30, val_f, color)

def draw_controls_panel(rect):
    draw_panel(rect, PANEL_COLOR, PANEL_BORDER, radius=15)
    draw_text("COMMAND CENTER", rect.x + 18, rect.y + 12, small_bold, INFO_BLUE)
    ctrls = [("LEFT CLICK", "DROP", SUCCESS_GREEN), ("RIGHT CLICK", "POP", RED_POP), ("R KEY", "RESET", GOLD_RESTART), ("M KEY", "MENU", INFO_BLUE), ("ESC", "QUIT", PURPLE_ACCENT)]
    for i, (key, action, color) in enumerate(ctrls):
        row = pygame.Rect(rect.x + 12, rect.y + 40 + (i * 26), rect.width - 24, 22)
        pygame.draw.rect(screen, INNER_PANEL, row, border_radius=6)
        draw_text(key, row.x + 8, row.y + 3, label_f, TEXT_SECONDARY)
        draw_text(action, row.x + 115, row.y + 1, small_bold, color)

def draw_piece(cx, cy, color_dict):
    pygame.draw.circle(screen, color_dict["ring"], (cx, cy), 32)
    pygame.draw.circle(screen, color_dict["main"], (cx, cy), 28)
    pygame.draw.circle(screen, color_dict["gloss"], (cx-8, cy-8), 8)

def draw_board_and_pieces(game, anim_mgr):
    bx, by = board_origin()
    draw_panel(pygame.Rect(bx-15, by-15, BOARD_WIDTH+30, BOARD_HEIGHT+30), BOARD_COLOR, BOARD_BORDER, radius=20, width=4)
    
    # Static board slots
    for r in range(ROWS):
        for c in range(COLS):
            cx, cy = bx + c*CELL_SIZE + CELL_SIZE//2, by + r*CELL_SIZE + CELL_SIZE//2
            pygame.draw.circle(screen, (15, 25, 45), (cx, cy), 30)
            pygame.draw.circle(screen, (45, 65, 110), (cx, cy), 30, 2)
            
            # Anti-flicker: Only draw static piece if that column isn't animating
            anim_cols = [p["col"] for p in anim_mgr.active_pieces]
            val = game.board[r, c]
            if val != 0 and c not in anim_cols:
                draw_piece(cx, cy, P1_COLORS if val == 1 else P2_COLORS)
            
            if r == 0: draw_text(str(c+1), cx-5, by-40, coord_f)
        draw_text(str(ROWS-r), bx-40, by + r*CELL_SIZE + 32, coord_f)

    # Animated pieces on top
    for p in anim_mgr.active_pieces:
        cx = bx + p["col"] * CELL_SIZE + CELL_SIZE // 2
        draw_piece(cx, int(p["curr_y"]), p["color"])

# --------------------------------------------------
# Main Loop
# --------------------------------------------------
def main():
    state, anim_mgr = "menu", AnimationManager()
    game, agents, last_move, last_reason = None, None, "—", "—"
    
    buttons = [
        MenuButton(WIDTH//2-210, 260, 420, 72, "Human vs Human", "hvh", "H", "H"),
        MenuButton(WIDTH//2-210, 345, 420, 72, "Human vs MCTS AI", "hvai", "H", "AI"),
        MenuButton(WIDTH//2-210, 430, 420, 72, "MCTS vs ID3 AI", "aivai", "AI", "AI", enabled=ID3_AVAILABLE)
    ]

    while True:
        m_pos = pygame.mouse.get_pos()
        screen.fill(BG_COLOR)
        anim_mgr.update()

        if state == "menu":
            draw_text("PopOut Pro", WIDTH//2-140, 80, title_f)
            draw_text("Select Assignment Mode", WIDTH//2-100, 160, body_f, TEXT_SECONDARY)
            for b in buttons: b.draw(m_pos)

        elif state == "game":
            draw_text("PopOut", 40, 30, header_f)
            status_t = (f"Winner: P{game.winner}" if game.winner else "Game Over: Draw") if game.game_over else f"Player {game.current_player}'s Turn"
            draw_text(status_t, 40, 85, button_f, SUCCESS_GREEN if game.game_over else TEXT_PRIMARY)
            
            draw_info_card(pygame.Rect(40, 130, 135, 65), "Pieces", game.count_pieces(), INFO_BLUE)
            draw_info_card(pygame.Rect(190, 130, 135, 65), "Reps", game.get_repetition_count(), PURPLE_ACCENT)
            
            draw_controls_panel(pygame.Rect(WIDTH-315, 15, 280, 185))
            draw_board_and_pieces(game, anim_mgr)

            # Footer bar
            draw_panel(pygame.Rect(40, HEIGHT-65, WIDTH-80, 45), PANEL_COLOR, PANEL_BORDER, radius=10)
            draw_text(f"Move: {last_move}", 60, HEIGHT-52, body_f, TEXT_SECONDARY)
            draw_text(f"Source: {last_reason}", WIDTH//2 + 20, HEIGHT-52, body_f, PURPLE_ACCENT)

            # AI Logic (Only if no animation)
            if not game.game_over and not anim_mgr.is_animating():
                agent = agents[game.current_player]
                if not isinstance(agent, str):
                    pygame.time.wait(AI_THINK_DELAY_MS)
                    res = agent.engine.search(game, return_stats=True) if isinstance(agent, MCTSPlayer) else (agent.get_move(game), {"reason":"ID3"})
                    move, stats = res
                    if move:
                        m_type, col = move
                        callback = lambda t=m_type, c=col: game.make_move(t, c)
                        if m_type == "drop":
                            row = next(r for r in range(ROWS-1, -1, -1) if game.board[r, col] == 0)
                            anim_mgr.trigger_drop(col, row, game.current_player, callback)
                        else:
                            anim_mgr.trigger_pop(col, game.board[:, col].copy(), callback)
                        last_move, last_reason = f"{m_type.upper()} Col {col+1}", stats.get("reason", "AI")

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m: state = "menu"
                if event.key == pygame.K_r and state == "game":
                    game = PopOutGame(ROWS, COLS); last_move, last_reason = "—", "Reset"
                if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not anim_mgr.is_animating():
                if state == "menu":
                    for b in buttons:
                        if b.rect.collidepoint(event.pos) and b.enabled:
                            game = PopOutGame(ROWS, COLS)
                            p1 = "H" if b.p1_type == "H" else MCTSPlayer(name="MCTS", player_num=1, time_limit=1.0)
                            p2 = "H" if b.p2_type == "H" else MCTSPlayer(name="MCTS", player_num=2, time_limit=1.0)
                            agents, state, last_move, last_reason = {1: p1, 2: p2}, "game", "—", "Start"
                elif state == "game" and not game.game_over and isinstance(agents[game.current_player], str):
                    bx, by = board_origin()
                    if bx <= event.pos[0] <= bx + BOARD_WIDTH:
                        col = (event.pos[0] - bx) // CELL_SIZE
                        m_type = "drop" if event.button == 1 else "pop"
                        if (m_type, col) in game.get_legal_moves():
                            callback = lambda t=m_type, c=col: game.make_move(t, c)
                            if m_type == "drop":
                                row = next(r for r in range(ROWS-1, -1, -1) if game.board[r, col] == 0)
                                anim_mgr.trigger_drop(col, row, game.current_player, callback)
                            else:
                                anim_mgr.trigger_pop(col, game.board[:, col].copy(), callback)
                            last_move, last_reason = f"{m_type.upper()} Col {col+1}", "Human User"

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()