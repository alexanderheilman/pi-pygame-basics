import math
import pygame
import colorsys

# ----------------------------
# Config / constants
# ----------------------------
WIDTH, HEIGHT = 800, 480
FPS = 60

PANEL_BG = (235, 235, 235)
TEXT = (20, 20, 20)

PICKER_CENTER = (150, 140)
OUTER_R = 110
INNER_R = 80
TRI_R = 72  # triangle "radius" from center to vertex

WHEEL_SURF_SIZE = OUTER_R * 2 + 2
TRI_SURF_SIZE = OUTER_R * 2 + 2

RIGHT_PANEL_X = 320

# ----------------------------
# Color math helpers
# ----------------------------
def hsv_to_rgb255(h_deg: float, s: float, v: float):
    """h_deg in [0,360), s,v in [0,1] -> (r,g,b) 0..255"""
    h = (h_deg % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

def rgb255_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

# ----------------------------
# Geometry helpers
# ----------------------------
def angle_deg_from_center(px, py, cx, cy):
    # pygame y+ is down, so invert dy for conventional angle if desired
    dx = px - cx
    dy = py - cy
    ang = math.degrees(math.atan2(-dy, dx))  # -dy to make up positive
    return (ang + 360.0) % 360.0

def dist(px, py, cx, cy):
    return math.hypot(px - cx, py - cy)

def equilateral_triangle_vertices(cx, cy, radius, hue_deg):
    """
    Make an equilateral triangle centered at (cx,cy), rotated so one vertex points along hue angle.
    Vertex A: pure hue (S=1,V=1) points at hue angle.
    Vertex B: black (V=0)
    Vertex C: white (S=0,V=1)
    """
    theta = math.radians(hue_deg)
    # A points to hue direction
    Ax = cx + radius * math.cos(theta)
    Ay = cy - radius * math.sin(theta)

    # other vertices are 120° apart
    theta_b = theta + 2.0 * math.pi / 3.0
    theta_c = theta - 2.0 * math.pi / 3.0

    Bx = cx + radius * math.cos(theta_b)
    By = cy - radius * math.sin(theta_b)

    Cx = cx + radius * math.cos(theta_c)
    Cy = cy - radius * math.sin(theta_c)

    return (Ax, Ay), (Bx, By), (Cx, Cy)

def barycentric(p, a, b, c):
    """Return barycentric coordinates (u,v,w) for p with triangle (a,b,c)."""
    px, py = p
    ax, ay = a
    bx, by = b
    cx, cy = c

    v0x, v0y = (bx - ax), (by - ay)
    v1x, v1y = (cx - ax), (cy - ay)
    v2x, v2y = (px - ax), (py - ay)

    denom = v0x * v1y - v1x * v0y
    if abs(denom) < 1e-9:
        return None

    v = (v2x * v1y - v1x * v2y) / denom
    w = (v0x * v2y - v2x * v0y) / denom
    u = 1.0 - v - w
    return (u, v, w)

def point_in_triangle(bary):
    if bary is None:
        return False
    u, v, w = bary
    eps = -1e-6
    return (u >= eps) and (v >= eps) and (w >= eps)

# ----------------------------
# Rendering: Hue ring (static)
# ----------------------------
def render_hue_wheel(size, outer_r, inner_r):
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = size // 2
    cy = size // 2
    for y in range(size):
        for x in range(size):
            r = dist(x, y, cx, cy)
            if inner_r <= r <= outer_r:
                ang = angle_deg_from_center(x, y, cx, cy)
                col = hsv_to_rgb255(ang, 1.0, 1.0)
                surf.set_at((x, y), (*col, 255))
            else:
                surf.set_at((x, y), (0, 0, 0, 0))
    return surf

# ----------------------------
# Rendering: SV triangle (changes with hue)
# ----------------------------
def render_sv_triangle(size, hue_deg, tri_r):
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = size // 2
    cy = size // 2

    A, B, C = equilateral_triangle_vertices(cx, cy, tri_r, hue_deg)

    # bounding box for speed
    minx = max(0, int(min(A[0], B[0], C[0])) - 1)
    maxx = min(size - 1, int(max(A[0], B[0], C[0])) + 1)
    miny = max(0, int(min(A[1], B[1], C[1])) - 1)
    maxy = min(size - 1, int(max(A[1], B[1], C[1])) + 1)

    # Fill transparent
    surf.fill((0, 0, 0, 0))

    # For each pixel inside triangle:
    # A = (S=1, V=1), B = (V=0), C = (S=0, V=1)
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            bc = barycentric((x + 0.5, y + 0.5), A, B, C)
            if not point_in_triangle(bc):
                continue
            u, v, w = bc  # u for A, v for B, w for C

            V = clamp01(u + w)       # A and C have V=1
            if V <= 1e-6:
                S = 0.0
            else:
                S = clamp01(u / V)   # fraction of "colored" vs "white" at that V

            col = hsv_to_rgb255(hue_deg, S, V)
            surf.set_at((x, y), (*col, 255))

    # draw triangle outline (optional)
    pygame.draw.polygon(surf, (0, 0, 0, 60), [A, B, C], width=2)

    return surf, (A, B, C)

def sv_to_point_in_triangle(S, V, A, B, C):
    # Inverse of mapping above:
    # wB = 1 - V
    # wA = S * V
    # wC = (1 - S) * V
    wB = clamp01(1.0 - V)
    wA = clamp01(S * V)
    wC = clamp01((1.0 - S) * V)
    # Renormalize (just in case of rounding)
    total = wA + wB + wC
    if total <= 1e-9:
        wA, wB, wC = 0.0, 1.0, 0.0
        total = 1.0
    wA /= total; wB /= total; wC /= total

    x = wA * A[0] + wB * B[0] + wC * C[0]
    y = wA * A[1] + wB * B[1] + wC * C[1]
    return (x, y)

# ----------------------------
# UI drawing helpers
# ----------------------------
def draw_label_value(screen, font, x, y, label, value):
    label_s = font.render(label, True, TEXT)
    value_s = font.render(str(value), True, TEXT)
    screen.blit(label_s, (x, y))
    screen.blit(value_s, (x + 110, y))

# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Color Picker (GTK-style)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 26)
    font_small = pygame.font.SysFont(None, 22)

    # Start color
    hue = 53.0
    sat = 0.69
    val = 0.99

    # Pre-render wheel once
    wheel = render_hue_wheel(WHEEL_SURF_SIZE, OUTER_R, INNER_R)

    # Triangle depends on hue
    tri, (A, B, C) = render_sv_triangle(TRI_SURF_SIZE, hue, TRI_R)

    dragging = None  # None | "hue" | "sv"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Touchscreens often appear as mouse events; handle those first.
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                cx, cy = PICKER_CENTER
                # translate mouse into wheel-local coords
                local_x = mx - (cx - WHEEL_SURF_SIZE // 2)
                local_y = my - (cy - WHEEL_SURF_SIZE // 2)

                # Are we in the ring?
                r = dist(local_x, local_y, WHEEL_SURF_SIZE // 2, WHEEL_SURF_SIZE // 2)
                if INNER_R <= r <= OUTER_R:
                    dragging = "hue"
                    hue = angle_deg_from_center(local_x, local_y, WHEEL_SURF_SIZE // 2, WHEEL_SURF_SIZE // 2)
                    tri, (A, B, C) = render_sv_triangle(TRI_SURF_SIZE, hue, TRI_R)

                else:
                    # Are we in the SV triangle?
                    # compute in triangle-local coords (same surface size & center)
                    bc = barycentric(
                        (local_x + 0.5, local_y + 0.5),
                        A, B, C
                    )
                    if point_in_triangle(bc):
                        dragging = "sv"
                        u, v, w = bc
                        val = clamp01(u + w)
                        sat = 0.0 if val <= 1e-6 else clamp01(u / val)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = None

            if event.type == pygame.MOUSEMOTION and dragging is not None:
                mx, my = event.pos
                cx, cy = PICKER_CENTER
                local_x = mx - (cx - WHEEL_SURF_SIZE // 2)
                local_y = my - (cy - WHEEL_SURF_SIZE // 2)

                if dragging == "hue":
                    hue = angle_deg_from_center(local_x, local_y, WHEEL_SURF_SIZE // 2, WHEEL_SURF_SIZE // 2)
                    tri, (A, B, C) = render_sv_triangle(TRI_SURF_SIZE, hue, TRI_R)

                elif dragging == "sv":
                    bc = barycentric((local_x + 0.5, local_y + 0.5), A, B, C)
                    if point_in_triangle(bc):
                        u, v, w = bc
                        val = clamp01(u + w)
                        sat = 0.0 if val <= 1e-6 else clamp01(u / val)

        # ---- Draw ----
        screen.fill(PANEL_BG)

        # Blit wheel and triangle
        cx, cy = PICKER_CENTER
        wheel_pos = (cx - WHEEL_SURF_SIZE // 2, cy - WHEEL_SURF_SIZE // 2)
        screen.blit(wheel, wheel_pos)
        screen.blit(tri, wheel_pos)

        # Hue handle marker on ring
        ang = math.radians(hue)
        ring_r = (OUTER_R + INNER_R) / 2.0
        hx = cx + ring_r * math.cos(ang)
        hy = cy - ring_r * math.sin(ang)
        pygame.draw.circle(screen, (0, 0, 0), (int(hx), int(hy)), 8, width=2)

        # SV marker in triangle
        mx, my = sv_to_point_in_triangle(sat, val, A, B, C)
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (int(wheel_pos[0] + mx), int(wheel_pos[1] + my)),
            6,
            width=2
        )

        # Preview swatch
        rgb = hsv_to_rgb255(hue, sat, val)
        preview_rect = pygame.Rect(40, 285, 220, 55)
        pygame.draw.rect(screen, rgb, preview_rect)
        pygame.draw.rect(screen, (0, 0, 0), preview_rect, width=2)

        # Right-side text
        h_int = int(round(hue))
        s_int = int(round(sat * 100))
        v_int = int(round(val * 100))
        draw_label_value(screen, font, RIGHT_PANEL_X, 40, "Hue:", h_int)
        draw_label_value(screen, font, RIGHT_PANEL_X, 75, "Saturation:", s_int)
        draw_label_value(screen, font, RIGHT_PANEL_X, 110, "Value:", v_int)

        draw_label_value(screen, font, RIGHT_PANEL_X + 230, 40, "Red:", rgb[0])
        draw_label_value(screen, font, RIGHT_PANEL_X + 230, 75, "Green:", rgb[1])
        draw_label_value(screen, font, RIGHT_PANEL_X + 230, 110, "Blue:", rgb[2])

        hex_str = rgb255_to_hex(rgb)
        hex_label = font.render("Color name:", True, TEXT)
        screen.blit(hex_label, (RIGHT_PANEL_X, 160))
        hex_box = pygame.Rect(RIGHT_PANEL_X + 120, 155, 140, 30)
        pygame.draw.rect(screen, (255, 255, 255), hex_box)
        pygame.draw.rect(screen, (0, 0, 0), hex_box, 2)
        screen.blit(font_small.render(hex_str, True, TEXT), (hex_box.x + 8, hex_box.y + 6))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
