import sys
import time
import struct
import argparse
from pathlib import Path

import numpy as np
import pygame

try:
    import serial  # pyserial
except ImportError:
    serial = None


# ----------------------------
# Framing protocol (AA 55 ...)
# ----------------------------
MAGIC0 = 0xAA
MAGIC1 = 0x55
VERSION = 0x01
TYPE_FRAME = 0x01

# CRC16-CCITT-FALSE: poly=0x1021, init=0xFFFF, xorout=0x0000, refin=false, refout=false
def crc16_ccitt_false(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def pack_frame(payload_rgb: bytes, seq: int) -> bytes:
    # Header for CRC starts at VERSION (per your design)
    fixed = struct.pack("<BBHI", VERSION, TYPE_FRAME, len(payload_rgb), seq)
    crc_input = fixed + payload_rgb
    crc = crc16_ccitt_false(crc_input)

    packet = bytearray()
    packet.append(MAGIC0)
    packet.append(MAGIC1)
    packet += fixed
    packet += payload_rgb
    packet += struct.pack("<H", crc)
    return bytes(packet)


# ----------------------------
# RGBA -> RGB helpers
# ----------------------------
def rgba32_to_rgb24(rgba: np.ndarray, bg=(0, 0, 0)) -> np.ndarray:
    """
    rgba: (32,32,4) uint8
    Returns: (32,32,3) uint8 composited over bg.
    """
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba, 0, 255).astype(np.uint8)

    rgb = rgba[..., :3].astype(np.uint16)
    a = rgba[..., 3].astype(np.uint16)  # 0..255

    bg_rgb = np.array(bg, dtype=np.uint16).reshape((1, 1, 3))

    # out = (src*alpha + bg*(255-alpha)) / 255
    out = (rgb * a[..., None] + bg_rgb * (255 - a[..., None]) + 127) // 255
    return out.astype(np.uint8)


# ----------------------------
# UI
# ----------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H,W,4) RGBA array, got shape {arr.shape}")
    return arr


def make_surface_from_rgba(arr_rgba: np.ndarray) -> pygame.Surface:
    """
    arr_rgba: (H,W,4) uint8
    Returns pygame Surface with per-pixel alpha.
    """
    h, w, _ = arr_rgba.shape
    # pygame expects (W,H) layout in frombuffer
    raw = arr_rgba.tobytes(order="C")
    surf = pygame.image.frombuffer(raw, (w, h), "RGBA")
    # Make a copy so backing buffer isn't tied to numpy memory lifetime
    return surf.convert_alpha()


def draw_selection_overlay(screen, img_rect, sel_x, sel_y, zoom, img_w, img_h,
                           show_grid=True):
    """
    img_rect: where the scaled image is drawn on screen
    sel_x, sel_y: top-left selection in image pixel coords
    zoom: scale factor (screen pixels per image pixel)
    """
    # Selection rectangle in screen coords
    sx = img_rect.x + int(sel_x * zoom)
    sy = img_rect.y + int(sel_y * zoom)
    sw = int(32 * zoom)
    sh = int(32 * zoom)

    # Outline
    pygame.draw.rect(screen, (255, 255, 0), (sx, sy, sw, sh), width=2)

    if show_grid and zoom >= 6:
        # Draw grid lines only when visible enough
        for i in range(1, 32):
            x = sx + int(i * zoom)
            y = sy + int(i * zoom)
            pygame.draw.line(screen, (255, 255, 0), (x, sy), (x, sy + sh), 1)
            pygame.draw.line(screen, (255, 255, 0), (sx, y), (sx + sw, y), 1)

    # Corner handles (visual affordance)
    handle = max(3, int(zoom * 0.5))
    pygame.draw.rect(screen, (255, 255, 0), (sx - handle, sy - handle, handle*2, handle*2))
    pygame.draw.rect(screen, (255, 255, 0), (sx + sw - handle, sy - handle, handle*2, handle*2))
    pygame.draw.rect(screen, (255, 255, 0), (sx - handle, sy + sh - handle, handle*2, handle*2))
    pygame.draw.rect(screen, (255, 255, 0), (sx + sw - handle, sy + sh - handle, handle*2, handle*2))


def main():
    parser = argparse.ArgumentParser(description="Crop 32x32 from RGBA .npy and send over Teensy framing protocol.")
    parser.add_argument("npy", type=str, help="Path to .npy containing (H,W,4) uint8 RGBA")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port (e.g. /dev/ttyACM0 or COM5). If empty, no sending.")
    parser.add_argument("--baud", type=int, default=2000000, help="Baud rate (ignored for USB CDC on many Teensy setups, but keep consistent).")
    parser.add_argument("--bg", type=str, default="0,0,0", help="Background for alpha composite as R,G,B (default 0,0,0).")
    parser.add_argument("--zoom", type=int, default=16, help="Initial zoom (screen pixels per image pixel).")
    parser.add_argument("--win", type=str, default="900,700", help="Window size W,H (default 900,700).")
    parser.add_argument("--no-grid", action="store_true", help="Do not draw 32x32 internal grid lines.")
    args = parser.parse_args()

    npy_path = Path(args.npy).expanduser()
    arr = load_npy(npy_path)
    img_h, img_w, _ = arr.shape

    bg_parts = [int(x) for x in args.bg.split(",")]
    if len(bg_parts) != 3:
        raise ValueError("--bg must be R,G,B")
    bg = tuple(clamp(x, 0, 255) for x in bg_parts)

    win_w, win_h = [int(x) for x in args.win.split(",")]

    # Serial
    ser = None
    if args.port:
        if serial is None:
            raise RuntimeError("pyserial not installed. Install with: pip install pyserial")
        ser = serial.Serial(args.port, args.baud, timeout=0)
        # Many Teensy CDC serial ports reset on open; tiny pause helps some setups
        time.sleep(0.25)

    pygame.init()
    pygame.display.set_caption(f"NPY crop + send: {npy_path.name}")
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    img_surf = make_surface_from_rgba(arr)

    zoom = float(max(1, args.zoom))
    show_grid = not args.no_grid

    # Selection top-left in image coords
    sel_x = 0
    sel_y = 0

    # Center image in window (we draw scaled image)
    def compute_img_rect():
        scaled_w = int(img_w * zoom)
        scaled_h = int(img_h * zoom)
        x = (win_w - scaled_w) // 2
        y = (win_h - scaled_h) // 2
        return pygame.Rect(x, y, scaled_w, scaled_h)

    img_rect = compute_img_rect()

    dragging = False
    drag_offset_x = 0
    drag_offset_y = 0

    seq = 0
    last_send_ms = 0

    def clamp_selection():
        nonlocal sel_x, sel_y
        sel_x = int(clamp(sel_x, 0, max(0, img_w - 32)))
        sel_y = int(clamp(sel_y, 0, max(0, img_h - 32)))

    clamp_selection()

    def send_current_crop():
        nonlocal seq, last_send_ms
        crop = arr[sel_y:sel_y + 32, sel_x:sel_x + 32, :]  # (32,32,4)
        if crop.shape[0] != 32 or crop.shape[1] != 32:
            return  # out of bounds (shouldn't happen due to clamp)

        rgb = rgba32_to_rgb24(crop, bg=bg)  # (32,32,3)
        payload = rgb.tobytes(order="C")    # RGBRGB...

        packet = pack_frame(payload, seq)
        seq = (seq + 1) & 0xFFFFFFFF

        if ser:
            ser.write(packet)
            # optional: ser.flush()  # usually unnecessary; may slow down
            last_send_ms = pygame.time.get_ticks()

    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Move selection
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    step = 1
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        step = 8
                    if event.key == pygame.K_LEFT:
                        sel_x -= step
                    elif event.key == pygame.K_RIGHT:
                        sel_x += step
                    elif event.key == pygame.K_UP:
                        sel_y -= step
                    elif event.key == pygame.K_DOWN:
                        sel_y += step
                    clamp_selection()

                # Zoom
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    zoom = min(64.0, zoom * 1.1)
                    img_rect = compute_img_rect()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom = max(1.0, zoom / 1.1)
                    img_rect = compute_img_rect()

                # Send
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                    send_current_crop()

                # Toggle grid
                elif event.key == pygame.K_g:
                    show_grid = not show_grid

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    # If clicked inside selection rect, start dragging
                    sx = img_rect.x + int(sel_x * zoom)
                    sy = img_rect.y + int(sel_y * zoom)
                    sw = int(32 * zoom)
                    sh = int(32 * zoom)
                    if pygame.Rect(sx, sy, sw, sh).collidepoint(mx, my):
                        dragging = True
                        drag_offset_x = mx - sx
                        drag_offset_y = my - sy

                # Mouse wheel zoom (pygame 2 often uses button 4/5 on some systems)
                elif event.button == 4:
                    zoom = min(64.0, zoom * 1.1)
                    img_rect = compute_img_rect()
                elif event.button == 5:
                    zoom = max(1.0, zoom / 1.1)
                    img_rect = compute_img_rect()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                # Convert mouse back to image coords for selection top-left
                sx = mx - drag_offset_x
                sy = my - drag_offset_y
                sel_x = int(round((sx - img_rect.x) / zoom))
                sel_y = int(round((sy - img_rect.y) / zoom))
                clamp_selection()

        # Draw background
        screen.fill((25, 25, 25))

        # Draw scaled image
        scaled = pygame.transform.scale(img_surf, (img_rect.w, img_rect.h))
        screen.blit(scaled, img_rect.topleft)

        # Draw selection overlay
        draw_selection_overlay(
            screen, img_rect, sel_x, sel_y, zoom, img_w, img_h,
            show_grid=show_grid
        )

        # HUD text
        status = []
        status.append(f"Image: {img_w}x{img_h}  sel=({sel_x},{sel_y})  zoom={zoom:.1f}x")
        status.append("Drag selection with mouse. Arrows move (SHIFT = faster).")
        status.append("SPACE/ENTER send. +/- or wheel zoom. G toggle grid. ESC quit.")
        if ser:
            status.append(f"Serial: {args.port}  seq={seq}  bg={bg}")
        else:
            status.append(f"Serial: (disabled)  bg={bg}  --port to enable")

        # flash indicator after send
        if pygame.time.get_ticks() - last_send_ms < 150:
            pygame.draw.circle(screen, (0, 255, 0), (18, 18), 8)

        y = 8
        for line in status:
            txt = font.render(line, True, (230, 230, 230))
            screen.blit(txt, (8, y))
            y += 22

        pygame.display.flip()

    if ser:
        ser.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
