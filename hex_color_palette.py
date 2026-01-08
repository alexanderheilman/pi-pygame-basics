import math
import pygame
import colorsys

# ----------------------------
# Config
# ----------------------------
WIDTH, HEIGHT = 800, 480
FPS = 60

BG = (245, 245, 245)
PANEL_BG = (235, 235, 235)
TEXT = (20, 20, 20)

MARGIN = 12
RIGHT_PANEL_W = 200
LEFT_PANEL_W = WIDTH - RIGHT_PANEL_W

# Pointy-top hex grid
HEX_SIZE = 18
GRID_RADIUS = 8  # pointy-top fits landscape nicely; tweak as desired

# Buttons (smaller)
BTN_W, BTN_H = 52, 44
BTN_GAP = 10

REPEAT_DELAY_MS = 320
REPEAT_RATE_MS = 65

SQRT3 = math.sqrt(3.0)

# ----------------------------
# Helpers
# ----------------------------
def clamp255(x: int) -> int:
    return 0 if x < 0 else 255 if x > 255 else x

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def hsv_to_rgb255(h_deg: float, s: float, v: float):
    h = (h_deg % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

# ----------------------------
# Hex math (POINTY-TOP axial)
# ----------------------------
def axial_to_pixel_pointy(q, r, size, origin):
    """Pointy-top axial -> pixel center."""
    ox, oy = origin
    x = size * SQRT3 * (q + r / 2.0) + ox
    y = size * 1.5 * r + oy
    return (x, y)

def pixel_to_axial_pointy(x, y, size, origin):
    """Pixel -> fractional axial (pointy-top)."""
    ox, oy = origin
    px = (x - ox) / size
    py = (y - oy) / size
    q = (SQRT3 / 3.0) * px - (1.0 / 3.0) * py
    r = (2.0 / 3.0) * py
    return (q, r)

def cube_round(x, y, z):
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return rx, ry, rz

def axial_round(q, r):
    x = q
    z = r
    y = -x - z
    rx, ry, rz = cube_round(x, y, z)
    return (rx, rz)

def hex_distance(q, r):
    return int((abs(q) + abs(r) + abs(q + r)) / 2)

def hex_polygon_pointy(cx, cy, size):
    """Vertices for POINTY-top hex."""
    pts = []
    for i in range(6):
        ang = math.radians(60 * i - 30)  # -30° makes a vertex point straight up
        pts.append((cx + size * math.cos(ang), cy + size * math.sin(ang)))
    return pts

# ----------------------------
# Button widget
# ----------------------------
class BigButton:
    def __init__(self, rect, label, on_click):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.is_down = False
        self.down_since = 0
        self.last_repeat = 0

    def draw(self, screen, font):
        fill = (255, 255, 255) if not self.is_down else (220, 220, 220)
        pygame.draw.rect(screen, fill, self.rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, width=2, border_radius=10)

        txt = font.render(self.label, True, TEXT)
        screen.blit(txt, (self.rect.centerx - txt.get_width() // 2,
                          self.rect.centery - txt.get_height() // 2))

    def handle_event(self, event, now_ms):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_down = True
                self.down_since = now_ms
                self.last_repeat = now_ms
                self.on_click()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_down = False
        elif event.type == pygame.MOUSEMOTION:
            if self.is_down and not self.rect.collidepoint(event.pos):
                self.is_down = False

    def tick_repeat(self, now_ms):
        if not self.is_down:
            return
        if now_ms - self.down_since < REPEAT_DELAY_MS:
            return
        if now_ms - self.last_repeat >= REPEAT_RATE_MS:
            self.last_repeat = now_ms
            self.on_click()

# ----------------------------
# Palette pre-render (pointy-top)
# ----------------------------
def build_hex_palette_surface_pointy(size, grid_radius):
    centers = []
    for q in range(-grid_radius, grid_radius + 1):
        for r in range(-grid_radius, grid_radius + 1):
            if hex_distance(q, r) <= grid_radius:
                x, y = axial_to_pixel_pointy(q, r, size, (0, 0))
                centers.append((x, y))

    minx = min(x for x, _ in centers) - size - 3
    maxx = max(x for x, _ in centers) + size + 3
    miny = min(y for _, y in centers) - size - 3
    maxy = max(y for _, y in centers) + size + 3

    w = int(math.ceil(maxx - minx))
    h = int(math.ceil(maxy - miny))

    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    ox = -minx
    oy = -miny

    # saturation scaling based on max dist from center
    max_dist_px = 1.0
    for q in range(-grid_radius, grid_radius + 1):
        for r in range(-grid_radius, grid_radius + 1):
            if hex_distance(q, r) <= grid_radius:
                cx, cy = axial_to_pixel_pointy(q, r, size, (ox, oy))
                max_dist_px = max(max_dist_px, math.hypot(cx - ox, cy - oy))

    hex_map = {}

    for q in range(-grid_radius, grid_radius + 1):
        for r in range(-grid_radius, grid_radius + 1):
            if hex_distance(q, r) > grid_radius:
                continue

            cx, cy = axial_to_pixel_pointy(q, r, size, (ox, oy))
            dx = cx - ox
            dy = cy - oy

            hue = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
            sat = min(1.0, math.hypot(dx, dy) / max_dist_px)
            val = 1.0

            rgb = hsv_to_rgb255(hue, sat, val)

            poly = hex_polygon_pointy(cx, cy, size - 1)
            pygame.draw.polygon(surf, rgb, poly)
            pygame.draw.polygon(surf, (0, 0, 0, 40), poly, width=1)

            hex_map[(q, r)] = {"center_surf": (cx, cy), "rgb": rgb}

    return surf, hex_map, (ox, oy)

def place_palette_on_left(palette_surf):
    avail_w = LEFT_PANEL_W - 2 * MARGIN
    avail_h = HEIGHT - 2 * MARGIN
    x = MARGIN + (avail_w - palette_surf.get_width()) // 2
    y = MARGIN + (avail_h - palette_surf.get_height()) // 2
    return (x, y)

# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 26)
    font_small = pygame.font.SysFont(None, 22)
    font_hex = pygame.font.SysFont(None, 34)
    font_btn = pygame.font.SysFont(None, 34)

    # Build palette once
    palette_surf, hex_map, (ox, oy) = build_hex_palette_surface_pointy(HEX_SIZE, GRID_RADIUS)
    palette_pos = place_palette_on_left(palette_surf)

    # Axial (0,0) on screen
    grid_center = (palette_pos[0] + ox, palette_pos[1] + oy)

    # Screen-space centers for outlines
    for k, v in hex_map.items():
        cx, cy = v["center_surf"]
        v["center"] = (palette_pos[0] + cx, palette_pos[1] + cy)

    selected_hex = (0, 0)
    rgb = hex_map[selected_hex]["rgb"]
    custom_mode = False

    # Buttons
    def make_channel_buttons(label, y, idx):
        x0 = LEFT_PANEL_W + 20

        def set_rgb(new_rgb):
            nonlocal rgb, custom_mode
            rgb = new_rgb
            custom_mode = True

        def dec():
            vals = list(rgb)
            vals[idx] = clamp255(vals[idx] - 1)
            set_rgb(tuple(vals))

        def inc():
            vals = list(rgb)
            vals[idx] = clamp255(vals[idx] + 1)
            set_rgb(tuple(vals))

        label_surf = font.render(label, True, TEXT)
        label_pos = (x0, y - 24)
        minus_rect = (x0, y, BTN_W, BTN_H)
        plus_rect  = (x0 + BTN_W + BTN_GAP, y, BTN_W, BTN_H)
        return label_surf, label_pos, BigButton(minus_rect, "−", dec), BigButton(plus_rect, "+", inc)

    r_label, r_label_pos, r_minus, r_plus = make_channel_buttons("Red",   180, 0)
    g_label, g_label_pos, g_minus, g_plus = make_channel_buttons("Green", 275, 1)
    b_label, b_label_pos, b_minus, b_plus = make_channel_buttons("Blue",  370, 2)

    buttons = [r_minus, r_plus, g_minus, g_plus, b_minus, b_plus]

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            for btn in buttons:
                btn.handle_event(event, now)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < LEFT_PANEL_W:
                    qf, rf = pixel_to_axial_pointy(mx, my, HEX_SIZE, grid_center)
                    q, r = axial_round(qf, rf)
                    if (q, r) in hex_map:
                        selected_hex = (q, r)
                        rgb = hex_map[selected_hex]["rgb"]
                        custom_mode = False

        for btn in buttons:
            btn.tick_repeat(now)

        # ---- Draw ----
        screen.fill(BG)

        left_panel = pygame.Rect(0, 0, LEFT_PANEL_W, HEIGHT)
        pygame.draw.rect(screen, (255, 255, 255), left_panel)
        screen.blit(palette_surf, palette_pos)

        if not custom_mode and selected_hex in hex_map:
            cx, cy = hex_map[selected_hex]["center"]
            poly = hex_polygon_pointy(cx, cy, HEX_SIZE + 2)
            pygame.draw.polygon(screen, (0, 0, 0), poly, width=3)

        right_panel = pygame.Rect(LEFT_PANEL_W, 0, RIGHT_PANEL_W, HEIGHT)
        pygame.draw.rect(screen, PANEL_BG, right_panel)

        preview = pygame.Rect(LEFT_PANEL_W + 20, 20, RIGHT_PANEL_W - 40, 44)
        pygame.draw.rect(screen, rgb, preview, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), preview, width=2, border_radius=10)

        screen.blit(font.render("Selected:", True, TEXT), (LEFT_PANEL_W + 20, 72))
        screen.blit(font_hex.render(rgb_to_hex(rgb), True, TEXT), (LEFT_PANEL_W + 20, 96))

        rr, gg, bb = rgb
        screen.blit(r_label, r_label_pos)
        screen.blit(g_label, g_label_pos)
        screen.blit(b_label, b_label_pos)

        x_val = LEFT_PANEL_W + 20 + (BTN_W * 2 + BTN_GAP) + 14
        screen.blit(font_small.render(f"{rr:3d}", True, TEXT), (x_val, 195))
        screen.blit(font_small.render(f"{gg:3d}", True, TEXT), (x_val, 290))
        screen.blit(font_small.render(f"{bb:3d}", True, TEXT), (x_val, 385))

        for btn in buttons:
            btn.draw(screen, font_btn)

        # hint = "Tap palette • Hold +/- to repeat • Esc to quit"
        # screen.blit(pygame.font.SysFont(None, 20).render(hint, True, (70, 70, 70)),
        #             (LEFT_PANEL_W + 20, HEIGHT - 26))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
