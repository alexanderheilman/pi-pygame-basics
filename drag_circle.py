import pygame
import math
import colorsys

WIDTH, HEIGHT = 800, 480
# CIRCLE_RADIUS = 40
# CIRCLE_COLOR_OFF = (255, 100, 100)
# CIRCLE_COLOR_ON = (255, 200, 100)

# Background
# BG_COLOR = (20, 20, 20)
BACKGROUND_IMAGE = '/usr/share/rpd-wallpaper/clouds.jpg'

# Fading trail
FADE_PER_FRAME = 5  # higher = faster fade (try 8–25)
STEP_SIZE_FACTOR = 0.25  # smaller = smoother

# Dynamic radius scaling
MIN_RADIUS = 10
MAX_RADIUS = 40
SPEED_FOR_MAX = 100.0  # pixels per event

# Time-weighted / exponentially-smoothed moving average
# vema ​← vema​ + α(vinst​ − vema​); α=1−e^(−Δt/τ)
# Velocity tuning (pixels/second)
V_FOR_MAX = 2400.0  # speed that maps to MAX_RADIUS; tune this

# Smoothing time constants (seconds)
VEL_TAU = 0.10      # velocity smoothing (~100ms)
RAD_TAU = 0.08      # optional radius smoothing (~80ms)

# Rainbow colors
HUE_SPEED = 0.25  # cycles per second (0.25 = 4 seconds per full rainbow)
IDLE_COLOR = (255, 255, 255)

def draw_thick_segment(surface, color, p0, p1, radius):
    """Draw a smooth thick segment (capsule): line + round end caps."""
    pygame.draw.line(surface, color, p0, p1, radius * 2)
    pygame.draw.circle(surface, color, p0, radius)
    pygame.draw.circle(surface, color, p1, radius)


def draw_smooth_segment(surface, color, p0, p1, radius):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)

    if dist == 0:
        pygame.draw.circle(surface, color, (int(x0), int(y0)), radius)
        return

    # Smaller step => smoother, more expensive. radius/2 is a good start.
    step = max(1.0, radius * STEP_SIZE_FACTOR)
    steps = int(dist / step)

    for i in range(steps + 1):
        t = i / steps if steps else 1.0
        x = x0 + dx * t
        y = y0 + dy * t
        pygame.draw.circle(surface, color, (int(x), int(y)), radius)


def ema_alpha(dt, tau):
    # Time-correct EMA coefficient
    return 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Drag the Circle (with Trail)")
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()

    # Circle state
    circle_pos = [WIDTH // 2, HEIGHT // 2]
    dragging = False
    drag_offset = (0, 0)

    # Fading surface (same size as screen, with alpha)
    # fade_surface = pygame.Surface((WIDTH, HEIGHT))
    # fade_surface.set_alpha(FADE_ALPHA)
    # fade_surface.fill(BG_COLOR)
    background = pygame.image.load(BACKGROUND_IMAGE).convert()

    # Trail layer with per-pixel alpha
    trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    
    prev_pos = None
    prev_t = None

    v_ema = 0.0
    radius_ema = float(MIN_RADIUS)

    hue = 0.0
    current_color = IDLE_COLOR
    last_time = pygame.time.get_ticks() / 1000.0

    running = True
    while running:
        now_time = pygame.time.get_ticks() / 1000.0
        dt_frame = now_time - last_time
        last_time = now_time

        if dragging:
            hue = (hue + HUE_SPEED * dt_frame) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            current_color = (int(r * 255), int(g * 255), int(b * 255))
        else:
            current_color = IDLE_COLOR

        for event in pygame.event.get():
            # --- quit handling ---
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # --- touch / mouse down ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # Check if touch is inside the circle
                dx = mx - circle_pos[0]
                dy = my - circle_pos[1]
                # hit-test using MAX_RADIUS so it's easy to grab at any size
                if dx * dx + dy * dy <= MAX_RADIUS * MAX_RADIUS:
                    dragging = True
                    drag_offset = (circle_pos[0] - mx, circle_pos[1] - my)
                    prev_pos = [circle_pos[0], circle_pos[1]]  # start segment from current pos
                    prev_t = pygame.time.get_ticks() / 1000.0  # seconds
                    v_ema = 0.0
                    radius_ema = float(MIN_RADIUS)

            # --- touch / mouse up ---
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
                prev_pos = None
                prev_t = None
                v_ema = 0.0
                radius_ema = float(MIN_RADIUS)

            # --- dragging ---
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                new_pos = [mx + drag_offset[0], my + drag_offset[1]]

                # Clamp circle to the screen
                new_pos[0] = max(MIN_RADIUS, min(WIDTH - MIN_RADIUS, new_pos[0]))
                new_pos[1] = max(MIN_RADIUS, min(HEIGHT - MIN_RADIUS, new_pos[1]))

                now = pygame.time.get_ticks() / 1000.0
                if prev_pos is not None:
                    dt = max(1e-4, now - prev_t)  # guard against 0
                    dx = new_pos[0] - prev_pos[0]
                    dy = new_pos[1] - prev_pos[1]
                    dist = math.hypot(dx, dy)

                    # Instantaneous velocity in px/s
                    v_inst = dist / dt

                    # Time-weighted EMA velocity
                    a_v = ema_alpha(dt, VEL_TAU)
                    v_ema = v_ema + a_v * (v_inst - v_ema)

                    # Map velocity → target radius
                    t = clamp(v_ema / V_FOR_MAX, 0.0, 1.0)
                    target_radius = MIN_RADIUS + t * (MAX_RADIUS - MIN_RADIUS)

                    # Optional: smooth the radius too (helps even more)
                    a_r = ema_alpha(dt, RAD_TAU)
                    radius_ema = radius_ema + a_r * (target_radius - radius_ema)

                    draw_smooth_segment(
                        trail,
                        (*current_color, 255),
                        prev_pos,
                        new_pos,
                        radius_ema,
                    )

                circle_pos[0], circle_pos[1] = new_pos
                prev_pos = new_pos
                prev_t = now

        # Fade the trail layer by reducing its alpha a little each frame
        trail.fill((0, 0, 0, FADE_PER_FRAME), special_flags=pygame.BLEND_RGBA_SUB)

        # Draw the circle *onto the trail layer*
        pygame.draw.circle(
            trail,
            (*current_color, 255),
            circle_pos,
            int(radius_ema),
        )

        # Composite
        screen.blit(background, (0, 0))
        screen.blit(trail, (0, 0))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
