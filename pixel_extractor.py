import sys
import os
import math
import time
from collections import deque

import numpy as np
import pygame
from PIL import Image

# ============================================================
# UI CONFIG
# ============================================================
WINDOW_W, WINDOW_H = 1200, 700
LEFT_W = WINDOW_W // 2
RIGHT_W = WINDOW_W - LEFT_W

BG = (18, 18, 22)
PANEL = (28, 28, 34)
TEXT = (230, 230, 240)
ACCENT = (80, 170, 255)
ROI_COLOR = (255, 220, 90)
GRID_COLOR = (90, 90, 110)

DEFAULT_PREVIEW_SCALE = 12

# Slider defaults / range (pixels)
PITCH_MIN_DEFAULT = 2.0
PITCH_MAX_DEFAULT = 80.0
PITCH_SLIDER_MIN = 1.0
PITCH_SLIDER_MAX = 256.0

# Background removal defaults
BG_TOL_DEFAULT = 12
BG_TOL_MIN = 0
BG_TOL_MAX = 80

# Toggles
PERIM_BG_REMOVE_DEFAULT = True
MANUAL_SEED_REMOVE_DEFAULT = True

# ============================================================
# SMALL HELPERS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rect_from_points(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    left = min(x0, x1)
    top = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    return pygame.Rect(left, top, w, h)

def draw_text(surf, font, text, x, y, color=TEXT):
    img = font.render(text, True, color)
    surf.blit(img, (x, y))

def draw_checker(surf, rect, s=14):
    x0, y0, w, h = rect
    for y in range(y0, y0 + h, s):
        for x in range(x0, x0 + w, s):
            c = (36, 36, 42) if ((x // s + y // s) % 2 == 0) else (30, 30, 36)
            pygame.draw.rect(surf, c, pygame.Rect(x, y, s, s))

def pil_image_to_pygame_surface(img_rgba: Image.Image) -> pygame.Surface:
    w, h = img_rgba.size
    data = img_rgba.tobytes()
    return pygame.image.fromstring(data, (w, h), "RGBA")

def rgb_array_to_surface(arr_hwc: np.ndarray) -> pygame.Surface:
    h, w, _ = arr_hwc.shape
    surf = pygame.Surface((w, h))
    arr_whc = np.transpose(arr_hwc, (1, 0, 2))  # (W,H,3)
    pygame.surfarray.blit_array(surf, arr_whc)
    return surf

def rgba_array_to_surface(arr_hw4: np.ndarray) -> pygame.Surface:
    h, w, _ = arr_hw4.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    rgb = arr_hw4[:, :, :3]
    a = arr_hw4[:, :, 3]
    rgb_whc = np.transpose(rgb, (1, 0, 2))  # (W,H,3)
    pygame.surfarray.blit_array(surf, rgb_whc)
    alpha_wh = np.transpose(a, (1, 0))      # (W,H)
    pygame.surfarray.pixels_alpha(surf)[:, :] = alpha_wh
    return surf

def clamp_roi_rect(r: pygame.Rect, img_w: int, img_h: int) -> pygame.Rect:
    if r is None:
        return None

    x0 = float(r.x)
    y0 = float(r.y)
    x1 = float(r.x + r.w)
    y1 = float(r.y + r.h)

    x0 = float(np.clip(x0, 0.0, float(img_w)))
    y0 = float(np.clip(y0, 0.0, float(img_h)))
    x1 = float(np.clip(x1, 0.0, float(img_w)))
    y1 = float(np.clip(y1, 0.0, float(img_h)))

    left = int(math.floor(min(x0, x1)))
    top = int(math.floor(min(y0, y1)))
    right = int(math.ceil(max(x0, x1)))
    bottom = int(math.ceil(max(y0, y1)))

    w = max(0, right - left)
    h = max(0, bottom - top)
    return pygame.Rect(left, top, w, h)

def sanitize_filename(name: str) -> str:
    name = name.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ "
    name = "".join(ch for ch in name if ch in allowed)
    name = name.strip().replace(" ", "_")
    return name

def unique_path(base_path_no_ext: str, ext: str) -> str:
    path = f"{base_path_no_ext}{ext}"
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        p = f"{base_path_no_ext}_{i}{ext}"
        if not os.path.exists(p):
            return p
        i += 1

# ============================================================
# SIMPLE SLIDER
# ============================================================
class Slider:
    def __init__(self, rect: pygame.Rect, vmin: float, vmax: float, value: float, label: str):
        self.rect = rect
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.value = float(value)
        self.label = label
        self.dragging = False

    def _value_from_mouse(self, mx: int) -> float:
        t = (mx - self.rect.x) / max(1, self.rect.w)
        t = float(np.clip(t, 0.0, 1.0))
        return self.vmin + t * (self.vmax - self.vmin)

    def handle_event(self, event) -> bool:
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.value = self._value_from_mouse(event.pos[0])
                changed = True
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.value = self._value_from_mouse(event.pos[0])
                changed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
        return changed

    def draw(self, surf: pygame.Surface, font_small: pygame.font.Font):
        pygame.draw.rect(surf, (45, 45, 55), self.rect, border_radius=6)
        inner = self.rect.inflate(-6, -10)
        inner.y += 5
        pygame.draw.rect(surf, (70, 70, 86), inner, border_radius=6)

        t = (self.value - self.vmin) / (self.vmax - self.vmin)
        t = float(np.clip(t, 0.0, 1.0))
        knob_x = int(self.rect.x + t * self.rect.w)
        knob = pygame.Rect(knob_x - 6, self.rect.y + 2, 12, self.rect.h - 4)
        pygame.draw.rect(surf, (120, 200, 255), knob, border_radius=6)

        draw_text(surf, font_small, f"{self.label}: {self.value:.2f}px", self.rect.x, self.rect.y - 18, TEXT)

# ============================================================
# FFT PEAK REFINEMENT + HARMONIC-AWARE PITCH
# ============================================================
def refined_peak_bin_logparabola(P: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(P) - 1:
        return float(k)
    y0 = np.log(P[k - 1] + 1e-12)
    y1 = np.log(P[k]     + 1e-12)
    y2 = np.log(P[k + 1] + 1e-12)
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(k)
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(k) + delta

def estimate_pitch_harmonic(sum_arr: np.ndarray,
                            min_pitch: float,
                            max_pitch: float,
                            max_harm: int = 6,
                            min_skip_bins: int = 1) -> tuple[float, float]:
    n = int(len(sum_arr))
    if n < 8:
        return 1.0, 0.0

    x = sum_arr.astype(np.float32)
    x -= float(x.mean())

    w = np.hanning(n).astype(np.float32)
    xw = x * w

    F = np.fft.rfft(xw)
    P = (np.abs(F) ** 2).astype(np.float64)
    P[0] = 0.0

    min_pitch = float(max(1e-6, min_pitch))
    max_pitch = float(max(min_pitch + 1e-6, max_pitch))

    k_min = int(math.ceil(n / max_pitch))
    k_max = int(math.floor(n / min_pitch))
    k_min = max(k_min, int(min_skip_bins), 1)
    k_max = min(k_max, len(P) - 2)

    if k_min >= k_max:
        k0 = int(np.argmax(P[1:]) + 1)
        k_ref = refined_peak_bin_logparabola(P, k0)
        f_ref = k_ref / float(n)
        pitch = (1.0 / f_ref) if f_ref > 0 else 1.0
        return float(pitch), float(P[k0])

    alpha = 1.5
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        score = 0.0
        for m in range(1, max_harm + 1):
            km = m * k
            if km >= len(P):
                break
            score += P[km] / (m ** alpha)
        if score > best_score:
            best_score = score
            best_k = k

    k_ref = refined_peak_bin_logparabola(P, best_k)
    f_ref = k_ref / float(n)
    if f_ref <= 1e-12:
        return 1.0, best_score

    pitch = 1.0 / f_ref
    pitch = float(np.clip(pitch, min_pitch, max_pitch))
    return pitch, float(best_score)

# ============================================================
# OFFSET ESTIMATION
# ============================================================
def estimate_offset_float(sum_arr: np.ndarray, pitch: float, search_steps: int | None = None) -> float:
    n = int(len(sum_arr))
    if n <= 1 or pitch <= 0:
        return 0.0

    if search_steps is None:
        search_steps = int(np.clip(pitch * 20.0, 64, 2000))

    offsets = np.linspace(0.0, float(pitch), num=search_steps, endpoint=False).astype(np.float32)
    xs = np.arange(n, dtype=np.float32)
    vals_src = sum_arr.astype(np.float32)

    max_k = int(math.floor((n - 1) / pitch)) + 1

    best_off = 0.0
    best_score = -1.0

    for off in offsets:
        t = off + float(pitch) * np.arange(max_k, dtype=np.float32)
        t = t[t <= (n - 1)]
        if t.size == 0:
            continue
        vals = np.interp(t, xs, vals_src)
        score = float(vals.sum())
        if score > best_score:
            best_score = score
            best_off = float(off)

    return float(best_off)

# ============================================================
# GRID BOUNDARIES + EXTRACTION
# ============================================================
def build_boundaries_float(length_px: int, pitch: float, offset_diff: float) -> list[int]:
    L = int(length_px)
    if L <= 1 or pitch <= 0:
        return [0, max(0, L)]

    base = float(offset_diff) + 1.0
    bounds_f = [0.0]

    if base < 1.0:
        base += math.ceil((1.0 - base) / float(pitch)) * float(pitch)

    b = base
    while b < float(L):
        bounds_f.append(b)
        b += float(pitch)
    bounds_f.append(float(L))

    bounds_i = np.rint(bounds_f).astype(int)
    bounds_i = np.clip(bounds_i, 0, L)

    cleaned = [int(bounds_i[0])]
    for v in bounds_i[1:]:
        v = int(v)
        if v > cleaned[-1]:
            cleaned.append(v)

    if cleaned[0] != 0:
        cleaned = [0] + cleaned
    if cleaned[-1] != L:
        cleaned.append(L)

    return cleaned

def extract_cells_median(arr_rgb: np.ndarray,
                         x_bounds: list[int],
                         y_bounds: list[int],
                         guard_frac: float = 0.12,
                         guard_min: int = 1) -> np.ndarray:
    H, W, _ = arr_rgb.shape
    grid_w = max(0, len(x_bounds) - 1)
    grid_h = max(0, len(y_bounds) - 1)
    if grid_w == 0 or grid_h == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    out = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for gy in range(grid_h):
        y0 = y_bounds[gy]
        y1 = y_bounds[gy + 1]
        ch = y1 - y0
        gy_guard = max(int(guard_min), int(round(ch * float(guard_frac))))
        sy = clamp(y0 + gy_guard, 0, H)
        ey = clamp(y1 - gy_guard, 0, H)

        for gx in range(grid_w):
            x0 = x_bounds[gx]
            x1 = x_bounds[gx + 1]
            cw = x1 - x0
            gx_guard = max(int(guard_min), int(round(cw * float(guard_frac))))
            sx = clamp(x0 + gx_guard, 0, W)
            ex = clamp(x1 - gx_guard, 0, W)

            if sx >= ex or sy >= ey:
                sx, ex = x0, x1
                sy, ey = y0, y1
                if sx >= ex or sy >= ey:
                    continue

            block = arr_rgb[sy:ey, sx:ex, :].reshape(-1, 3)
            out[gy, gx, :] = np.median(block, axis=0).astype(np.uint8)

    return out

# ============================================================
# BACKGROUND COLOR (perimeter mode) - FIXED (uint32 packing)
# ============================================================
def estimate_bg_from_perimeter(ex_rgb: np.ndarray) -> tuple[int, int, int]:
    H, W, _ = ex_rgb.shape
    if H == 0 or W == 0:
        return (255, 255, 255)

    top = ex_rgb[0, :, :]
    bottom = ex_rgb[-1, :, :] if H > 1 else top
    left = ex_rgb[1:-1, 0, :] if W > 1 and H > 2 else np.empty((0, 3), dtype=np.uint8)
    right = ex_rgb[1:-1, -1, :] if W > 1 and H > 2 else np.empty((0, 3), dtype=np.uint8)

    perim = np.concatenate([top, bottom, left, right], axis=0)

    p = perim.astype(np.uint32)
    packed = (p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]
    uniq, counts = np.unique(packed, return_counts=True)
    mode = int(uniq[np.argmax(counts)])

    return ((mode >> 16) & 255, (mode >> 8) & 255, mode & 255)

# ============================================================
# CONTIGUOUS BG MASKS
# ============================================================
def bg_like_mask(ex_rgb: np.ndarray, bg_rgb: tuple[int, int, int], tol: int) -> np.ndarray:
    bg = np.array(bg_rgb, dtype=np.int16).reshape(1, 1, 3)
    diff = np.abs(ex_rgb.astype(np.int16) - bg)
    return (diff[:, :, 0] <= tol) & (diff[:, :, 1] <= tol) & (diff[:, :, 2] <= tol)

def flood_from_seeds(is_bg_like: np.ndarray, seeds: list[tuple[int,int]], connectivity: int = 4) -> np.ndarray:
    H, W = is_bg_like.shape
    visited = np.zeros((H, W), dtype=bool)
    q = deque()

    def try_seed(y, x):
        if 0 <= y < H and 0 <= x < W and is_bg_like[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))

    for (y, x) in seeds:
        try_seed(y, x)

    if connectivity == 8:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        y, x = q.popleft()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if is_bg_like[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    return visited

def perimeter_seeds(H: int, W: int, is_bg_like: np.ndarray) -> list[tuple[int,int]]:
    seeds = []
    for x in range(W):
        if is_bg_like[0, x]:
            seeds.append((0, x))
        if is_bg_like[H - 1, x]:
            seeds.append((H - 1, x))
    for y in range(1, H - 1):
        if is_bg_like[y, 0]:
            seeds.append((y, 0))
        if is_bg_like[y, W - 1]:
            seeds.append((y, W - 1))
    return seeds

def apply_transparency_with_masks(ex_rgb: np.ndarray,
                                  bg_rgb: tuple[int,int,int],
                                  tol: int,
                                  do_perimeter: bool,
                                  manual_seeds: list[tuple[int,int]],
                                  do_manual: bool,
                                  connectivity: int = 4) -> tuple[np.ndarray, int]:
    """
    Returns (rgba, num_transparent)
    """
    H, W, _ = ex_rgb.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = ex_rgb
    rgba[:, :, 3] = 255

    if H == 0 or W == 0:
        return rgba, 0

    is_bg = bg_like_mask(ex_rgb, bg_rgb, tol)

    transparent = np.zeros((H, W), dtype=bool)

    if do_perimeter:
        seeds = perimeter_seeds(H, W, is_bg)
        if seeds:
            transparent |= flood_from_seeds(is_bg, seeds, connectivity=connectivity)

    if do_manual and manual_seeds:
        transparent |= flood_from_seeds(is_bg, manual_seeds, connectivity=connectivity)

    rgba[transparent, 3] = 0
    return rgba, int(transparent.sum())

# ============================================================
# MAIN APP
# ============================================================
def main():
    pygame.init()
    pygame.display.set_caption("Pixel Art Extractor (manual bg seed clicks)")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Menlo", 16)
    font_small = pygame.font.SysFont("Menlo", 14)
    font_big = pygame.font.SysFont("Menlo", 20)

    if len(sys.argv) < 2:
        print("Usage: python pixel_art_extractor.py path/to/image.png")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    pil_rgba = Image.open(img_path).convert("RGBA")
    pil_rgb = pil_rgba.convert("RGB")
    pil_gray = pil_rgba.convert("L")

    arr_rgb_full = np.asarray(pil_rgb, dtype=np.uint8)
    arr_gray_full = np.asarray(pil_gray, dtype=np.uint8)

    loaded_surf = pil_image_to_pygame_surface(pil_rgba)
    img_w, img_h = pil_rgba.size

    # Pan/zoom for left panel
    zoom = 1.0
    pan_x, pan_y = 0.0, 0.0

    def image_to_screen(ix, iy):
        return ix * zoom + pan_x, iy * zoom + pan_y

    def screen_to_image(sx, sy):
        return (sx - pan_x) / zoom, (sy - pan_y) / zoom

    if img_w > 0 and img_h > 0:
        zoom = min((LEFT_W - 40) / img_w, (WINDOW_H - 40) / img_h)
        zoom = max(0.05, min(20.0, zoom))
        pan_x = (LEFT_W - img_w * zoom) / 2
        pan_y = (WINDOW_H - img_h * zoom) / 2

    selecting = False
    roi_img_rect = None
    p0_img = None
    p1_img = None

    panning = False
    last_mouse = (0, 0)

    extracted_rgb = np.zeros((0, 0, 3), dtype=np.uint8)
    extracted_rgba = np.zeros((0, 0, 4), dtype=np.uint8)

    status = "Left-drag ROI. Shift+drag to pan. Click preview to add bg seeds. S to save."
    x_pitch = None
    y_pitch = None
    x_off_diff = 0.0
    y_off_diff = 0.0

    guard_frac = 0.12
    guard_min = 1
    preview_scale = DEFAULT_PREVIEW_SCALE

    # Pitch bounds
    slider_w = RIGHT_W - 40
    slider_min = Slider(pygame.Rect(LEFT_W + 20, 90, slider_w, 26), PITCH_SLIDER_MIN, PITCH_SLIDER_MAX, PITCH_MIN_DEFAULT, "Min pitch")
    slider_max = Slider(pygame.Rect(LEFT_W + 20, 150, slider_w, 26), PITCH_SLIDER_MIN, PITCH_SLIDER_MAX, PITCH_MAX_DEFAULT, "Max pitch")

    # BG removal state
    bg_tol = BG_TOL_DEFAULT
    bg_color = (255, 255, 255)
    do_perimeter_bg = PERIM_BG_REMOVE_DEFAULT
    do_manual_bg = MANUAL_SEED_REMOVE_DEFAULT
    manual_seeds = []  # list of (y,x) in extracted grid coords
    last_transparent_count = 0

    # Grid bounds for overlay
    last_x_bounds = None
    last_y_bounds = None

    # Save prompt state
    saving = False
    save_name = ""
    save_hint = "Type filename, Enter to save, Esc to cancel"

    # Preview placement (computed each frame; stored so clicks can map into grid)
    preview_blit_rect = None
    preview_scale_used = 1  # actual integer scale used
    preview_grid_size = (0, 0)  # (W,H)

    def rebuild_rgba():
        nonlocal extracted_rgba, bg_color, last_transparent_count
        if extracted_rgb.size == 0:
            extracted_rgba = np.zeros((0, 0, 4), dtype=np.uint8)
            last_transparent_count = 0
            return
        bg_color = estimate_bg_from_perimeter(extracted_rgb)
        extracted_rgba, last_transparent_count = apply_transparency_with_masks(
            extracted_rgb,
            bg_rgb=bg_color,
            tol=bg_tol,
            do_perimeter=do_perimeter_bg,
            manual_seeds=manual_seeds,
            do_manual=do_manual_bg,
            connectivity=4
        )

    def recompute_from_roi():
        nonlocal extracted_rgb, status
        nonlocal x_pitch, y_pitch, x_off_diff, y_off_diff
        nonlocal last_x_bounds, last_y_bounds
        nonlocal manual_seeds

        last_x_bounds = None
        last_y_bounds = None
        manual_seeds = []  # clear seeds when re-extracting (seed coords would no longer match)

        if roi_img_rect is None or roi_img_rect.w < 2 or roi_img_rect.h < 2:
            extracted_rgb[:] = 0
            return

        x0 = clamp(roi_img_rect.x, 0, img_w - 1)
        y0 = clamp(roi_img_rect.y, 0, img_h - 1)
        w = clamp(roi_img_rect.w, 1, img_w - x0)
        h = clamp(roi_img_rect.h, 1, img_h - y0)

        roi_rgb = arr_rgb_full[y0:y0+h, x0:x0+w, :]
        roi_gray = arr_gray_full[y0:y0+h, x0:x0+w]
        if roi_gray.size == 0:
            extracted_rgb[:] = 0
            return

        g16 = roi_gray.astype(np.int16)
        x_diff = np.abs(g16[:, 1:] - g16[:, :-1])
        y_diff = np.abs(g16[1:, :] - g16[:-1, :])

        x_sum = np.sum(x_diff, axis=0)
        y_sum = np.sum(y_diff, axis=1)

        pitch_min = float(np.clip(slider_min.value, PITCH_SLIDER_MIN, PITCH_SLIDER_MAX))
        pitch_max = float(np.clip(slider_max.value, PITCH_SLIDER_MIN, PITCH_SLIDER_MAX))
        if pitch_max < pitch_min + 0.5:
            pitch_max = pitch_min + 0.5
            slider_max.value = pitch_max

        x_pitch, _ = estimate_pitch_harmonic(x_sum, min_pitch=pitch_min, max_pitch=pitch_max, max_harm=6)
        y_pitch, _ = estimate_pitch_harmonic(y_sum, min_pitch=pitch_min, max_pitch=pitch_max, max_harm=6)

        x_off_diff = estimate_offset_float(x_sum, x_pitch)
        y_off_diff = estimate_offset_float(y_sum, y_pitch)

        last_x_bounds = build_boundaries_float(w, x_pitch, x_off_diff)
        last_y_bounds = build_boundaries_float(h, y_pitch, y_off_diff)

        extracted_rgb = extract_cells_median(
            roi_rgb,
            x_bounds=last_x_bounds,
            y_bounds=last_y_bounds,
            guard_frac=guard_frac,
            guard_min=guard_min
        )

        rebuild_rgba()

        status = (
            f"ROI {w}x{h} | pitch=(x≈{x_pitch:.3f}, y≈{y_pitch:.3f}) "
            f"| bg={bg_color} tol={bg_tol} | seeds={len(manual_seeds)} | transparent={last_transparent_count} "
            f"| out={extracted_rgb.shape[1]}x{extracted_rgb.shape[0]}"
        )

    def begin_save_prompt():
        nonlocal saving, save_name
        saving = True
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_name = sanitize_filename(base) or "extract"

    def commit_save():
        nonlocal status, saving
        if extracted_rgba.size == 0:
            status = "Nothing to save."
            saving = False
            return

        out_dir = "extracts"
        os.makedirs(out_dir, exist_ok=True)

        name = sanitize_filename(save_name)
        if not name:
            status = "Invalid filename."
            return

        base_path = os.path.join(out_dir, name)
        npy_path = unique_path(base_path, ".npy")
        png_path = unique_path(base_path, ".png")

        np.save(npy_path, extracted_rgba)

        prev_surf = rgba_array_to_surface(extracted_rgba)
        png_scale = max(1, preview_scale)
        prev_big = pygame.transform.scale(
            prev_surf,
            (prev_surf.get_width() * png_scale, prev_surf.get_height() * png_scale)
        )
        pygame.image.save(prev_big, png_path)

        status = f"Saved: {npy_path} (+ {png_path})"
        saving = False

    def try_add_seed_from_click(mx: int, my: int):
        """
        If click occurs inside preview blit area, map to extracted grid coords and add a seed.
        """
        nonlocal manual_seeds, status
        if extracted_rgb.size == 0 or preview_blit_rect is None:
            return
        if not preview_blit_rect.collidepoint((mx, my)):
            return

        # Map screen -> local in preview surface (scaled)
        lx = mx - preview_blit_rect.x
        ly = my - preview_blit_rect.y
        if preview_scale_used <= 0:
            return

        gx = int(lx // preview_scale_used)
        gy = int(ly // preview_scale_used)

        H, W, _ = extracted_rgb.shape
        if 0 <= gx < W and 0 <= gy < H:
            manual_seeds.append((gy, gx))
            rebuild_rgba()
            status = f"Added seed at (x={gx}, y={gy}). seeds={len(manual_seeds)} transparent={last_transparent_count}"

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # sliders
            changed = slider_min.handle_event(event) or slider_max.handle_event(event)
            if changed and roi_img_rect is not None:
                recompute_from_roi()

            # save prompt typing
            if saving:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        saving = False
                        status = "Save cancelled."
                    elif event.key == pygame.K_RETURN:
                        commit_save()
                    elif event.key == pygame.K_BACKSPACE:
                        save_name = save_name[:-1]
                    else:
                        ch = event.unicode
                        if ch and ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ":
                            save_name += ch
                continue

            # Mouse: ROI selection/pan on left; seed clicks on right
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                if event.button == 1:
                    if mx < LEFT_W:
                        mods = pygame.key.get_mods()
                        shift = (mods & pygame.KMOD_SHIFT) != 0
                        if shift:
                            panning = True
                            selecting = False
                            p0_img = None
                            p1_img = None
                            last_mouse = (mx, my)
                            status = "Panning (Shift+drag)…"
                        else:
                            selecting = True
                            panning = False
                            p0_img = screen_to_image(mx, my)
                            p1_img = p0_img
                    else:
                        # Right side click -> add bg seed
                        if do_manual_bg:
                            try_add_seed_from_click(mx, my)

                # zoom only on left side
                if mx < LEFT_W and event.button in (4, 5):
                    if event.button == 4:
                        before = screen_to_image(mx, my)
                        zoom = min(30.0, zoom * 1.1)
                        after = screen_to_image(mx, my)
                        pan_x += (after[0] - before[0]) * zoom
                        pan_y += (after[1] - before[1]) * zoom
                    elif event.button == 5:
                        before = screen_to_image(mx, my)
                        zoom = max(0.02, zoom / 1.1)
                        after = screen_to_image(mx, my)
                        pan_x += (after[0] - before[0]) * zoom
                        pan_y += (after[1] - before[1]) * zoom

            elif event.type == pygame.MOUSEBUTTONUP:
                mx, my = event.pos
                if event.button == 1 and mx < LEFT_W:
                    if panning:
                        panning = False
                        status = "Pan ended."
                    elif selecting:
                        selecting = False
                        if p0_img and p1_img:
                            x0f, y0f = p0_img
                            x1f, y1f = p1_img
                            rx = int(math.floor(min(x0f, x1f)))
                            ry = int(math.floor(min(y0f, y1f)))
                            rw = int(math.ceil(abs(x1f - x0f)))
                            rh = int(math.ceil(abs(y1f - y0f)))
                            roi_img_rect = pygame.Rect(rx, ry, rw, rh)
                            roi_img_rect = clamp_roi_rect(roi_img_rect, img_w, img_h)

                            if roi_img_rect.w < 2 or roi_img_rect.h < 2:
                                roi_img_rect = None
                                status = "ROI too small after clamping."
                            else:
                                recompute_from_roi()

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if selecting and mx < LEFT_W:
                    p1_img = screen_to_image(mx, my)
                if panning:
                    dx = mx - last_mouse[0]
                    dy = my - last_mouse[1]
                    pan_x += dx
                    pan_y += dy
                    last_mouse = (mx, my)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Background controls
                elif event.key == pygame.K_LEFTBRACKET:
                    bg_tol = clamp(bg_tol - 1, BG_TOL_MIN, BG_TOL_MAX)
                    rebuild_rgba()
                elif event.key == pygame.K_RIGHTBRACKET:
                    bg_tol = clamp(bg_tol + 1, BG_TOL_MIN, BG_TOL_MAX)
                    rebuild_rgba()
                elif event.key == pygame.K_p:
                    do_perimeter_bg = not do_perimeter_bg
                    rebuild_rgba()
                elif event.key == pygame.K_m:
                    do_manual_bg = not do_manual_bg
                    rebuild_rgba()
                elif event.key == pygame.K_x:
                    manual_seeds = []
                    rebuild_rgba()

                # Guard tuning
                elif event.key == pygame.K_g:
                    guard_frac = clamp(guard_frac + 0.02, 0.0, 0.45)
                    if roi_img_rect is not None:
                        recompute_from_roi()
                elif event.key == pygame.K_h:
                    guard_frac = clamp(guard_frac - 0.02, 0.0, 0.45)
                    if roi_img_rect is not None:
                        recompute_from_roi()

                # Preview scale
                elif event.key == pygame.K_MINUS:
                    preview_scale = max(1, preview_scale - 1)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    preview_scale = min(64, preview_scale + 1)

                # Save prompt
                elif event.key == pygame.K_s:
                    begin_save_prompt()

                # Reset
                elif event.key == pygame.K_r:
                    roi_img_rect = None
                    selecting = False
                    panning = False
                    p0_img = None
                    p1_img = None
                    extracted_rgb = np.zeros((0, 0, 3), dtype=np.uint8)
                    extracted_rgba = np.zeros((0, 0, 4), dtype=np.uint8)
                    manual_seeds = []
                    x_pitch = None
                    y_pitch = None
                    x_off_diff = 0.0
                    y_off_diff = 0.0
                    last_x_bounds = None
                    last_y_bounds = None
                    status = "Reset. Left-drag ROI. Shift+drag pan."

        # ============================================================
        # DRAW
        # ============================================================
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, pygame.Rect(0, 0, LEFT_W, WINDOW_H))
        pygame.draw.rect(screen, PANEL, pygame.Rect(LEFT_W, 0, RIGHT_W, WINDOW_H))

        # Left: image
        scaled_w = max(1, int(img_w * zoom))
        scaled_h = max(1, int(img_h * zoom))
        img_scaled = pygame.transform.smoothscale(loaded_surf, (scaled_w, scaled_h))
        screen.blit(img_scaled, (pan_x, pan_y))

        # ROI
        if selecting and p0_img and p1_img:
            sx0, sy0 = image_to_screen(*p0_img)
            sx1, sy1 = image_to_screen(*p1_img)
            r = rect_from_points((sx0, sy0), (sx1, sy1))
            pygame.draw.rect(screen, ROI_COLOR, r, 2)
        elif roi_img_rect is not None:
            sx, sy = image_to_screen(roi_img_rect.x, roi_img_rect.y)
            sw = roi_img_rect.w * zoom
            sh = roi_img_rect.h * zoom
            r = pygame.Rect(int(sx), int(sy), int(sw), int(sh))
            pygame.draw.rect(screen, ROI_COLOR, r, 2)

            # Grid overlay
            if x_pitch is not None and y_pitch is not None and last_x_bounds is not None and last_y_bounds is not None:
                for bx in last_x_bounds:
                    ix = roi_img_rect.x + bx
                    sxl, syl = image_to_screen(ix, roi_img_rect.y)
                    sxe, sye = image_to_screen(ix, roi_img_rect.y + roi_img_rect.h)
                    pygame.draw.line(screen, GRID_COLOR, (sxl, syl), (sxe, sye), 1)
                for by in last_y_bounds:
                    iy = roi_img_rect.y + by
                    sxl, syl = image_to_screen(roi_img_rect.x, iy)
                    sxe, sye = image_to_screen(roi_img_rect.x + roi_img_rect.w, iy)
                    pygame.draw.line(screen, GRID_COLOR, (sxl, syl), (sxe, sye), 1)

        # Divider
        pygame.draw.line(screen, (40, 40, 50), (LEFT_W, 0), (LEFT_W, WINDOW_H), 2)

        # Right: UI + preview
        right_rect = pygame.Rect(LEFT_W, 0, RIGHT_W, WINDOW_H)
        draw_checker(screen, right_rect, s=14)

        draw_text(screen, font, "Pitch constraints", LEFT_W + 12, 12, TEXT)
        draw_text(screen, font_small, "Drag sliders to limit FFT search range (reduces harmonics).", LEFT_W + 12, 36, (200, 200, 210))
        slider_min.draw(screen, font_small)
        slider_max.draw(screen, font_small)

        draw_text(screen, font_small,
                  f"BG tol: {bg_tol} ([/])  | Perimeter mask: {'ON' if do_perimeter_bg else 'OFF'} (P)  "
                  f"| Manual seeds: {'ON' if do_manual_bg else 'OFF'} (M)  | Clear seeds: X",
                  LEFT_W + 12, 186, (200, 200, 210))

        draw_text(screen, font_small,
                  f"Click on preview to seed-fill bg removal. bg={bg_color} seeds={len(manual_seeds)} transparent={last_transparent_count}",
                  LEFT_W + 12, 206, (200, 200, 210))

        # Preview layout
        preview_blit_rect = None
        preview_scale_used = 1
        preview_grid_size = (0, 0)

        top_margin = 230
        if extracted_rgba.size > 0:
            prev = rgba_array_to_surface(extracted_rgba)
            pw, ph = prev.get_width(), prev.get_height()
            preview_grid_size = (pw, ph)

            scale = max(1, int(preview_scale))
            prev_big = pygame.transform.scale(prev, (pw * scale, ph * scale))

            avail_h = WINDOW_H - top_margin - 20
            avail_w = RIGHT_W - 20

            draw_surf = prev_big
            dw, dh = draw_surf.get_width(), draw_surf.get_height()
            preview_scale_used = scale

            if dw > avail_w or dh > avail_h:
                sfit = min(avail_w / pw, avail_h / ph)
                sfit_int = max(1, int(sfit))
                draw_surf = pygame.transform.scale(prev, (pw * sfit_int, ph * sfit_int))
                dw, dh = draw_surf.get_width(), draw_surf.get_height()
                preview_scale_used = sfit_int

            x = LEFT_W + (RIGHT_W - dw) // 2
            y = top_margin + (avail_h - dh) // 2
            screen.blit(draw_surf, (x, y))
            preview_blit_rect = pygame.Rect(x, y, dw, dh)

            # Draw seed markers
            for (sy, sx) in manual_seeds:
                cx = x + sx * preview_scale_used + preview_scale_used // 2
                cy = y + sy * preview_scale_used + preview_scale_used // 2
                pygame.draw.circle(screen, (255, 120, 120), (cx, cy), max(2, preview_scale_used // 4), 1)

            draw_text(screen, font, f"Extracted: {pw} x {ph} (W x H)", LEFT_W + 12, 212, TEXT)
        else:
            draw_text(screen, font, "Extracted preview will appear below after ROI selection.", LEFT_W + 12, 212, TEXT)

        # Help + status
        help_lines = [
            "Mouse: Left-drag ROI | Shift+drag pan | Wheel zoom | Click preview to add bg seed",
            "Save: S (type name, Enter) | Reset: R | Quit: ESC",
            "Guard: G/H | Preview: -/+ | BG tol: [ / ] | Perimeter mask: P | Manual seed mask: M | Clear seeds: X",
        ]
        y = WINDOW_H - 70
        for line in help_lines:
            draw_text(screen, font_small, line, 12, y, (200, 200, 210))
            y += 18
        draw_text(screen, font_small, f"Status: {status}", 12, WINDOW_H - 90, ACCENT)

        # Save prompt overlay
        if saving:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))

            box = pygame.Rect(WINDOW_W // 2 - 320, WINDOW_H // 2 - 80, 640, 160)
            pygame.draw.rect(screen, (35, 35, 45), box, border_radius=14)
            pygame.draw.rect(screen, (70, 170, 255), box, 2, border_radius=14)

            draw_text(screen, font_big, "Save extraction", box.x + 20, box.y + 18, TEXT)
            draw_text(screen, font_small, save_hint, box.x + 20, box.y + 50, (200, 200, 210))

            input_box = pygame.Rect(box.x + 20, box.y + 78, box.w - 40, 38)
            pygame.draw.rect(screen, (20, 20, 26), input_box, border_radius=8)
            pygame.draw.rect(screen, (90, 90, 110), input_box, 2, border_radius=8)

            caret = "|" if (pygame.time.get_ticks() // 400) % 2 == 0 else ""
            draw_text(screen, font_big, f"{save_name}{caret}", input_box.x + 10, input_box.y + 6, TEXT)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
