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


# =========================
# Framing protocol config
# =========================
MAGIC1 = 0xAA
MAGIC2 = 0x55
VERSION = 0x01
TYPE_FRAME = 0x01

W = 32
H = 32


# CRC-16/CCITT-FALSE: init=0xFFFF, poly=0x1021, no xorout, no reflect
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


def build_packet(payload: bytes, seq: int) -> bytes:
    """
    Packet:
      MAGIC1, MAGIC2,
      VERSION, TYPE, LEN_L, LEN_H, SEQ0..SEQ3,
      PAYLOAD,
      CRC_L, CRC_H

    CRC over: VERSION..PAYLOAD (i.e., header_wo_magic + payload)
    """
    length = len(payload)
    header_wo_magic = struct.pack(
        "<BBHI",
        VERSION,
        TYPE_FRAME,
        length,
        seq & 0xFFFFFFFF,
    )
    crc_input = header_wo_magic + payload
    crc = crc16_ccitt_false(crc_input)
    return bytes([MAGIC1, MAGIC2]) + crc_input + struct.pack("<H", crc)


# =========================
# Brightness + RGBA helpers
# =========================
def apply_brightness_linear(rgb_bytes: bytes, brightness: int) -> bytes:
    """
    brightness: 0..255 (0=off, 255=unchanged)
    """
    if brightness >= 255:
        return rgb_bytes
    if brightness <= 0:
        return bytes(len(rgb_bytes))

    b = int(brightness)
    out = bytearray(len(rgb_bytes))
    for i, v in enumerate(rgb_bytes):
        out[i] = (v * b + 127) // 255
    return bytes(out)


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

    out = (rgb * a[..., None] + bg_rgb * (255 - a[..., None]) + 127) // 255
    return out.astype(np.uint8)


# =========================
# Mapping: from (x,y) to physical wiring/order on 4-panel display
# =========================
def xy_to_index_1panel_serpentine(x, y, panel_w=32, panel_h=8):
    # idx = 0 in lower right, serpentine to idx = 255 in lower left
    n_cols_to_right = panel_w - x - 1
    n_pixels_to_right = n_cols_to_right * panel_h
    if x % 2 == 0:
        # Even columns, idx increases top to bottom
        n_pixels_in_col = y
    else:
        # Odd columns, idx increases bottom to top
        n_pixels_in_col = panel_h - y - 1
    return n_pixels_to_right + n_pixels_in_col


def xy_to_index_4panels_serpentine(x, y, panel_w=32, panel_h=8):
    panel = y // panel_h
    y_in_panel = y % panel_h
    idx_in_panel = xy_to_index_1panel_serpentine(x, y_in_panel, panel_w, panel_h)
    return panel * (panel_w * panel_h) + idx_in_panel


def build_index_map(W=32, H=32):
    """
    index_map[p] = (y, x) for physical pixel index p.
    So physical-order pixels = rgb[index_map[:,0], index_map[:,1]]
    """
    coords = np.zeros((W * H, 2), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            p = xy_to_index_4panels_serpentine(x, y, panel_w=W, panel_h=8)
            coords[p] = (y, x)
    return coords


def frame_rgb_rowmajor_to_physical_payload(rgb_rowmajor: bytes, index_map: np.ndarray, W=32, H=32) -> bytes:
    """
    rgb_rowmajor: RGBRGB... row-major for a (H,W) frame
    returns payload bytes in physical LED order RGBRGB...
    """
    rgb = np.frombuffer(rgb_rowmajor, dtype=np.uint8).reshape((H, W, 3))
    ordered = rgb[index_map[:, 0], index_map[:, 1]]  # (N,3)
    return ordered.astype(np.uint8, copy=False).tobytes()



# =========================
# NPY loading + pygame UI
# =========================
def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected (H,W,4) RGBA array, got shape {arr.shape}")
    # normalize dtype if needed
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def make_surface_from_rgba(arr_rgba: np.ndarray) -> pygame.Surface:
    """
    arr_rgba: (H,W,4) uint8
    Returns pygame Surface with per-pixel alpha.
    """
    h, w, _ = arr_rgba.shape
    raw = arr_rgba.tobytes(order="C")
    surf = pygame.image.frombuffer(raw, (w, h), "RGBA")
    return surf.convert_alpha()


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def draw_selection_overlay(screen, img_rect, sel_x, sel_y, zoom, show_grid=True):
    sx = img_rect.x + int(sel_x * zoom)
    sy = img_rect.y + int(sel_y * zoom)
    sw = int(32 * zoom)
    sh = int(32 * zoom)

    pygame.draw.rect(screen, (255, 255, 0), (sx, sy, sw, sh), width=2)

    if show_grid and zoom >= 6:
        for i in range(1, 32):
            x = sx + int(i * zoom)
            y = sy + int(i * zoom)
            pygame.draw.line(screen, (255, 255, 0), (x, sy), (x, sy + sh), 1)
            pygame.draw.line(screen, (255, 255, 0), (sx, y), (sx + sw, y), 1)


def main():
    ap = argparse.ArgumentParser(description="Load RGBA .npy, select 32x32, send via framing protocol with physical mapping.")
    ap.add_argument("npy", type=str, help="Path to .npy containing (H,W,4) uint8 RGBA")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port (e.g. /dev/ttyACM0 or COM5). Empty = no sending.")
    ap.add_argument("--baud", type=int, default=2000000, help="Baud rate (keep consistent with your setup).")
    ap.add_argument("--send_fps", type=float, default=20.0, help="Continuous send FPS when streaming is enabled.")
    ap.add_argument("--zoom", type=int, default=16, help="Initial zoom (screen pixels per image pixel).")
    ap.add_argument("--win", type=str, default="900,700", help="Window size W,H.")
    ap.add_argument("--bg", type=str, default="0,0,0", help="Alpha composite background for RGBA -> RGB as R,G,B.")
    ap.add_argument("--no-grid", action="store_true", help="Disable internal 32x32 grid lines.")
    args = ap.parse_args()

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
        # Teensy often resets on open; short pause helps
        time.sleep(0.8)
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass

    # Precompute mapping once
    index_map = build_index_map()

    pygame.init()
    pygame.display.set_caption(f"NPY crop + mapped send: {npy_path.name}")
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    img_surf = make_surface_from_rgba(arr)

    zoom = float(max(1, args.zoom))
    show_grid = not args.no_grid

    # Selection top-left in image coords
    sel_x = 0
    sel_y = 0

    def clamp_selection():
        nonlocal sel_x, sel_y
        sel_x = int(clamp(sel_x, 0, max(0, img_w - 32)))
        sel_y = int(clamp(sel_y, 0, max(0, img_h - 32)))

    clamp_selection()

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

    # Send state
    seq = 0
    streaming = False
    send_period = 1.0 / max(0.1, float(args.send_fps))
    last_send_t = 0.0

    brightness = 64  # start here; tweak as you like

    # Edge-trigger for Enter
    enter_was_down = False

    # Flash indicator
    last_send_ms = 0
    last_blank_ms = 0

    def build_current_frame_rowmajor_rgb() -> bytes:
        """
        Returns 32x32 RGB row-major bytes (RGBRGB...) after compositing alpha over bg.
        """
        crop = arr[sel_y:sel_y + 32, sel_x:sel_x + 32, :]
        rgb = rgba32_to_rgb24(crop, bg=bg)
        return rgb.tobytes(order="C")

    def send_rowmajor_rgb(rgb_rowmajor: bytes):
        nonlocal seq, last_send_t, last_send_ms
        if len(rgb_rowmajor) != W * H * 3:
            return

        dimmed = apply_brightness_linear(rgb_rowmajor, brightness)
        
        # Convert row-major -> physical LED order
        payload = frame_rgb_rowmajor_to_physical_payload(dimmed, index_map)
        pkt = build_packet(payload, seq)
        seq = (seq + 1) & 0xFFFFFFFF

        if ser:
            ser.write(pkt)
        last_send_t = time.time()
        last_send_ms = pygame.time.get_ticks()

    def blank_display():
        nonlocal last_blank_ms
        rgb0 = bytes(W * H * 3)  # already "row-major black"
        # brightness doesn't matter if it's all zeros, but leave it consistent:
        payload = frame_rgb_rowmajor_to_physical_payload(rgb0, index_map)
        pkt = build_packet(payload, seq=0)  # seq doesn't really matter for blank
        if ser:
            ser.write(pkt)
        last_blank_ms = pygame.time.get_ticks()

    running = True
    try:
        while running:
            clock.tick(60)

            # --- events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    # Space toggles streaming
                    elif event.key == pygame.K_SPACE:
                        streaming = not streaming

                    # Toggle grid
                    elif event.key == pygame.K_g:
                        show_grid = not show_grid

                    # Blank now
                    elif event.key == pygame.K_b:
                        blank_display()

                    # Brightness controls
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        brightness = max(0, brightness - 8)
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        brightness = min(255, brightness + 8)
                    elif event.key == pygame.K_LEFTBRACKET:
                        brightness = max(0, brightness - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        brightness = min(255, brightness + 1)
                    elif event.key == pygame.K_0:
                        brightness = 0
                    elif event.key == pygame.K_9:
                        brightness = 255

                    # Move selection with arrows (SHIFT = faster)
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

                    # Zoom keys
                    elif event.key in (pygame.K_PAGEUP,):
                        zoom = min(64.0, zoom * 1.1)
                        img_rect = compute_img_rect()
                    elif event.key in (pygame.K_PAGEDOWN,):
                        zoom = max(1.0, zoom / 1.1)
                        img_rect = compute_img_rect()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mx, my = event.pos
                        sx = img_rect.x + int(sel_x * zoom)
                        sy = img_rect.y + int(sel_y * zoom)
                        sw = int(32 * zoom)
                        sh = int(32 * zoom)
                        if pygame.Rect(sx, sy, sw, sh).collidepoint(mx, my):
                            dragging = True
                            drag_offset_x = mx - sx
                            drag_offset_y = my - sy

                    # Wheel zoom (some systems use 4/5)
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
                    sx = mx - drag_offset_x
                    sy = my - drag_offset_y
                    sel_x = int(round((sx - img_rect.x) / zoom))
                    sel_y = int(round((sy - img_rect.y) / zoom))
                    clamp_selection()

                elif event.type == pygame.MOUSEWHEEL:
                    # pygame 2 wheel event
                    zoom_factor = 1.0 + (0.10 * event.y)
                    zoom = clamp(zoom * zoom_factor, 1.0, 64.0)
                    img_rect = compute_img_rect()

            # --- draw ---
            screen.fill((25, 25, 25))

            scaled = pygame.transform.scale(img_surf, (img_rect.w, img_rect.h))
            screen.blit(scaled, img_rect.topleft)

            draw_selection_overlay(screen, img_rect, sel_x, sel_y, zoom, show_grid=show_grid)

            # HUD
            hud = [
                f"Image: {img_w}x{img_h}  sel=({sel_x},{sel_y})  zoom={zoom:.1f}x",
                f"Streaming: {'ON' if streaming else 'OFF'}   send_fps={args.send_fps:g}",
                f"Brightness: {brightness}/255    [-/+]=8  [[]=1  0=off  9=full",
                "Enter: send once   Space: toggle stream   B: blank now   G: grid   Esc/close: quit+blank",
            ]
            y = 8
            for line in hud:
                txt = font.render(line, True, (230, 230, 230))
                screen.blit(txt, (8, y))
                y += 22

            # Send/blank flash indicators
            now_ms = pygame.time.get_ticks()
            if now_ms - last_send_ms < 150:
                pygame.draw.circle(screen, (0, 255, 0), (18, 18), 8)
            if now_ms - last_blank_ms < 150:
                pygame.draw.circle(screen, (255, 0, 0), (40, 18), 8)

            pygame.display.flip()

            # --- sending logic ---
            keys = pygame.key.get_pressed()
            enter_down = keys[pygame.K_RETURN] or keys[pygame.K_KP_ENTER]
            enter_pressed = enter_down and not enter_was_down
            enter_was_down = enter_down

            if enter_pressed:
                rgb_rowmajor = build_current_frame_rowmajor_rgb()
                send_rowmajor_rgb(rgb_rowmajor)
            elif streaming:
                t = time.time()
                if t - last_send_t >= send_period:
                    rgb_rowmajor = build_current_frame_rowmajor_rgb()
                    send_rowmajor_rgb(rgb_rowmajor)

    finally:
        # Always blank on exit if we can
        if ser:
            try:
                blank_display()
                time.sleep(0.05)
            except Exception:
                pass
            try:
                ser.close()
            except Exception:
                pass
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
