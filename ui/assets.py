"""Pillow-generated logo and navigation icons.

The PNG files are shipped in ``assets/``; ``ensure_assets()`` re-creates any that are
missing, so the application never fails because of a deleted image.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from modules.config import COLORS, ICONS_DIR, LOGO_PATH

ICON_NAMES = ["dashboard", "classify", "history", "analytics", "reports", "settings"]
_SS = 4  # supersampling factor for smooth edges
_SIZE = 64


def _hex(color: str, alpha: int = 255) -> tuple:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    size = _SIZE * _SS
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image), size


def _finish(image: Image.Image, size: int = _SIZE) -> Image.Image:
    return image.resize((size, size), Image.LANCZOS)


def _draw_icon(name: str, color: tuple) -> Image.Image:
    image, d, s = _canvas()
    u = s / 64  # 1 unit on a 64px grid
    w = int(5 * u)
    if name == "dashboard":
        for x, y in ((10, 10), (36, 10), (10, 36), (36, 36)):
            d.rounded_rectangle([x * u, y * u, (x + 18) * u, (y + 18) * u], radius=4 * u, fill=color)
    elif name == "classify":  # magnifier with a plus
        d.ellipse([9 * u, 9 * u, 41 * u, 41 * u], outline=color, width=w)
        d.line([(37 * u, 37 * u), (54 * u, 54 * u)], fill=color, width=int(6 * u))
        d.line([(25 * u, 19 * u), (25 * u, 31 * u)], fill=color, width=int(4 * u))
        d.line([(19 * u, 25 * u), (31 * u, 25 * u)], fill=color, width=int(4 * u))
    elif name == "history":  # clock
        d.ellipse([8 * u, 8 * u, 56 * u, 56 * u], outline=color, width=w)
        d.line([(32 * u, 32 * u), (32 * u, 18 * u)], fill=color, width=int(5 * u))
        d.line([(32 * u, 32 * u), (43 * u, 38 * u)], fill=color, width=int(5 * u))
    elif name == "analytics":  # bar chart
        d.rounded_rectangle([10 * u, 34 * u, 22 * u, 54 * u], radius=2 * u, fill=color)
        d.rounded_rectangle([26 * u, 20 * u, 38 * u, 54 * u], radius=2 * u, fill=color)
        d.rounded_rectangle([42 * u, 8 * u, 54 * u, 54 * u], radius=2 * u, fill=color)
    elif name == "reports":  # document with folded corner and text lines
        d.polygon([(14 * u, 6 * u), (38 * u, 6 * u), (50 * u, 18 * u), (50 * u, 58 * u), (14 * u, 58 * u)],
                  outline=color, width=w)
        d.line([(38 * u, 6 * u), (38 * u, 18 * u), (50 * u, 18 * u)], fill=color, width=int(4 * u))
        for y in (30, 40, 50):
            d.line([(22 * u, y * u), (42 * u, y * u)], fill=color, width=int(4 * u))
    elif name == "settings":  # gear
        cx = cy = s / 2
        for k in range(8):
            a = math.radians(k * 45)
            ca, sa = math.cos(a), math.sin(a)
            pts = []
            for r, off in ((20 * u, -5 * u), (20 * u, 5 * u), (29 * u, 5 * u), (29 * u, -5 * u)):
                pts.append((cx + r * ca - off * sa, cy + r * sa + off * ca))
            d.polygon(pts, fill=color)
        d.ellipse([cx - 22 * u, cy - 22 * u, cx + 22 * u, cy + 22 * u], fill=color)
        d.ellipse([cx - 10 * u, cy - 10 * u, cx + 10 * u, cy + 10 * u], fill=(0, 0, 0, 0))
    return _finish(image)


def _draw_logo(size: int = 256) -> Image.Image:
    s = size * 2
    top, bottom = _hex(COLORS["primary"]), _hex(COLORS["secondary"])
    gradient = Image.new("RGBA", (s, s))
    gd = ImageDraw.Draw(gradient)
    for y in range(s):
        t = y / (s - 1)
        gd.line([(0, y), (s, y)], fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(4)))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=255)
    logo = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    logo.paste(gradient, (0, 0), mask)
    d = ImageDraw.Draw(logo)
    u = s / 100
    white = (255, 255, 255, 255)
    # heartbeat / pulse line
    pts = [(14, 60), (32, 60), (40, 46), (50, 78), (60, 40), (68, 60), (86, 60)]
    d.line([(x * u, y * u) for x, y in pts], fill=white, width=int(5.5 * u), joint="curve")
    # small ribbon-style loop above the line
    d.ellipse([40 * u, 14 * u, 60 * u, 34 * u], outline=white, width=int(4.5 * u))
    return logo.resize((size, size), Image.LANCZOS)


def ensure_assets() -> None:
    """Create logo and icon PNGs if they do not exist yet."""
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOGO_PATH.exists():
        _draw_logo().save(LOGO_PATH)
    for name in ICON_NAMES:
        normal = ICONS_DIR / f"{name}.png"
        active = ICONS_DIR / f"{name}_active.png"
        if not normal.exists():
            _draw_icon(name, _hex("#B6C2D6")).save(normal)
        if not active.exists():
            _draw_icon(name, _hex(COLORS["primary"])).save(active)


def load_icons(size: int = 22) -> dict:
    """Return {name: (normal PhotoImage, active PhotoImage)}. Empty dict if Pillow/Tk cannot do it."""
    try:
        from PIL import ImageTk

        ensure_assets()
        icons = {}
        for name in ICON_NAMES:
            pair = []
            for suffix in ("", "_active"):
                img = Image.open(ICONS_DIR / f"{name}{suffix}.png").convert("RGBA").resize((size, size), Image.LANCZOS)
                pair.append(ImageTk.PhotoImage(img))
            icons[name] = tuple(pair)
        return icons
    except Exception:
        return {}


def load_logo(size: int = 40):
    try:
        from PIL import ImageTk

        ensure_assets()
        return ImageTk.PhotoImage(Image.open(LOGO_PATH).convert("RGBA").resize((size, size), Image.LANCZOS))
    except Exception:
        return None

