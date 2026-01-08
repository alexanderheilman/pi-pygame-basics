import pygame

WIDTH, HEIGHT = 800, 480

# Grid
CELL_SIZE = 24                 # pixels per cell
GRID_LINE = True
GRID_LINE_COLOR = (210, 210, 210)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def cell_from_pos(pos, grid_w, grid_h):
    mx, my = pos
    cx = mx // CELL_SIZE
    cy = my // CELL_SIZE
    if 0 <= cx < grid_w and 0 <= cy < grid_h:
        return int(cx), int(cy)
    return None

def draw_grid(surface, grid, grid_w, grid_h):
    # Draw cells
    for y in range(grid_h):
        row = grid[y]
        py = y * CELL_SIZE
        for x in range(grid_w):
            color = BLACK if row[x] else WHITE
            pygame.draw.rect(
                surface,
                color,
                (x * CELL_SIZE, py, CELL_SIZE, CELL_SIZE),
            )

    # Optional grid lines
    if GRID_LINE:
        for x in range(grid_w + 1):
            px = x * CELL_SIZE
            pygame.draw.line(surface, GRID_LINE_COLOR, (px, 0), (px, grid_h * CELL_SIZE), 1)
        for y in range(grid_h + 1):
            py = y * CELL_SIZE
            pygame.draw.line(surface, GRID_LINE_COLOR, (0, py), (grid_w * CELL_SIZE, py), 1)

def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Pixel Drawer")
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()

    # Grid dims derived from screen + CELL_SIZE
    grid_w = WIDTH // CELL_SIZE
    grid_h = HEIGHT // CELL_SIZE

    # grid[y][x] = 0 (white) or 1 (black)
    grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

    # Input state (kept consistent with your other code)
    dragging = False
    prev_pos = None

    start_cell = None
    drag_mode = None         # None or "paint"
    last_painted_cell = None

    running = True
    while running:
        for event in pygame.event.get():
            # --- quit handling ---
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # --- touch / mouse down ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True
                prev_pos = event.pos

                start_cell = cell_from_pos(event.pos, grid_w, grid_h)
                drag_mode = None
                last_painted_cell = None

                # Tap toggles the pressed cell immediately
                if start_cell is not None:
                    cx, cy = start_cell
                    grid[cy][cx] ^= 1

            # --- touch / mouse up ---
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
                prev_pos = None
                start_cell = None
                drag_mode = None
                last_painted_cell = None

            # --- dragging / motion ---
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                prev_pos = event.pos

                cell = cell_from_pos((mx, my), grid_w, grid_h)
                if cell is None:
                    continue

                # If we moved off the initial cell, we are in drag-paint mode
                if start_cell is not None and cell != start_cell:
                    drag_mode = "paint"

                if drag_mode == "paint":
                    cx, cy = cell
                    if last_painted_cell != cell:
                        grid[cy][cx] = 1  # drag paints black
                        last_painted_cell = cell

        # Clear to pure white
        screen.fill(WHITE)

        # Draw grid
        draw_grid(screen, grid, grid_w, grid_h)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
