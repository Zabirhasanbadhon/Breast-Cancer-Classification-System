"""Breast Cancer Classification System - desktop application entry point.

    python app.py

Requires the trained model (run ``python train_model.py`` once first).
This is an educational machine-learning project, not a medical device.
"""
from __future__ import annotations

import sys

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:  # Tkinter is optional on some Linux distributions
    print(
        "Tkinter is not available in this Python installation.\n"
        "  Windows / macOS: reinstall Python from python.org and keep 'tcl/tk' ticked.\n"
        "  Debian / Ubuntu: sudo apt install python3-tk\n"
        "  Fedora:          sudo dnf install python3-tkinter"
    )
    sys.exit(1)

try:
    from modules.config import APP_NAME, APP_VERSION, COLORS as C, DISCLAIMER_SHORT
    from modules.database import Database
    from modules.predictor import ModelLoadError, ModelNotFoundError, Predictor
    from ui.analytics_page import AnalyticsPage
    from ui.assets import load_icons, load_logo
    from ui.classify import ClassifyPage
    from ui.dashboard import DashboardPage
    from ui.history import HistoryPage
    from ui.reports import ReportsPage
    from ui.settings import SettingsPage
    from ui.theme import font
    from ui.widgets import apply_ttk_style
except ImportError as exc:
    print(
        f"A required package is missing: {exc}\n\n"
        "Install all dependencies with:\n    python -m pip install -r requirements.txt"
    )
    sys.exit(1)

NAV_ITEMS = [
    ("dashboard", "Dashboard"),
    ("classify", "Classify Tumor"),
    ("history", "Prediction History"),
    ("analytics", "Analytics"),
    ("reports", "Reports"),
    ("settings", "Settings"),
]
PAGE_CLASSES = {
    "dashboard": DashboardPage,
    "classify": ClassifyPage,
    "history": HistoryPage,
    "analytics": AnalyticsPage,
    "reports": ReportsPage,
    "settings": SettingsPage,
}


class NavButton(tk.Frame):
    """Sidebar entry: accent bar + icon + text."""

    def __init__(self, parent, key, text, icons, command):
        super().__init__(parent, bg=C["sidebar"], cursor="hand2")
        self.key, self.icons, self._command = key, icons, command
        self.bar = tk.Frame(self, width=4, bg=C["sidebar"])
        self.bar.pack(side="left", fill="y")
        self.label = tk.Label(self, text="  " + text, font=font(11), fg="#B6C2D6", bg=C["sidebar"], anchor="w",
                              padx=12, pady=11, compound="left" if icons.get(key) else "none")
        if icons.get(key):
            self.label.configure(image=icons[key][0])
        self.label.pack(side="left", fill="x", expand=True)
        for widget in (self, self.label, self.bar):
            widget.bind("<Button-1>", lambda _e: self._command(self.key))
            widget.bind("<Enter>", lambda _e: self._hover(True))
            widget.bind("<Leave>", lambda _e: self._hover(False))
        self.active = False

    def _hover(self, inside):
        if not self.active:
            color = "#1E2D47" if inside else C["sidebar"]
            self.configure(bg=color)
            self.label.configure(bg=color)
            self.bar.configure(bg=color)

    def set_active(self, active):
        self.active = active
        bg = "#21314F" if active else C["sidebar"]
        self.configure(bg=bg)
        self.label.configure(bg=bg, fg=C["text"] if active else "#B6C2D6",
                             font=font(11, "bold" if active else "normal"))
        self.bar.configure(bg=C["primary"] if active else bg)
        if self.icons.get(self.key):
            self.label.configure(image=self.icons[self.key][1 if active else 0])


class App(tk.Tk):
    def __init__(self, predictor: Predictor, db: Database):
        super().__init__()
        self.predictor = predictor
        self.metadata = predictor.metadata
        self.db = db
        self.title(f"{APP_NAME}  v{APP_VERSION}")
        self.geometry("1280x820")
        self.minsize(1040, 660)
        self.configure(bg=C["bg"])
        apply_ttk_style(self)
        self.icons = load_icons()
        self.logo = load_logo(44)
        if self.logo is not None:
            try:
                self.iconphoto(True, self.logo)
            except tk.TclError:
                pass

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self.container = tk.Frame(self, bg=C["bg"])
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        self.pages = {}
        for key, cls in PAGE_CLASSES.items():
            page = cls(self.container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page
        self.current_key = None
        self.show_page("dashboard")

    def _build_sidebar(self):
        sidebar = tk.Frame(self, bg=C["sidebar"], width=250)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.pack_propagate(False)

        brand = tk.Frame(sidebar, bg=C["sidebar"], padx=16, pady=20)
        brand.pack(fill="x")
        if self.logo is not None:
            tk.Label(brand, image=self.logo, bg=C["sidebar"]).pack(side="left")
        text = tk.Frame(brand, bg=C["sidebar"])
        text.pack(side="left", padx=(10, 0))
        tk.Label(text, text="ZaBir HaSan", font=font(12, "bold"), fg=C["text"], bg=C["sidebar"], anchor="w").pack(fill="x")
        tk.Label(text, text="Classification System", font=font(9), fg=C["muted"], bg=C["sidebar"], anchor="w").pack(fill="x")
        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x", padx=16, pady=(0, 10))

        self.nav_buttons = {}
        for key, label in NAV_ITEMS:
            button = NavButton(sidebar, key, label, self.icons, self.show_page)
            button.pack(fill="x", pady=1)
            self.nav_buttons[key] = button

        footer = tk.Frame(sidebar, bg=C["sidebar"], padx=16, pady=16)
        footer.pack(side="bottom", fill="x")
        tk.Label(footer, text=DISCLAIMER_SHORT, font=font(8), fg=C["muted"], bg=C["sidebar"], justify="left",
                 anchor="w", wraplength=210).pack(fill="x")
        tk.Label(footer, text=f"Model: {self.predictor.model_name}", font=font(8, "bold"), fg=C["primary"],
                 bg=C["sidebar"], anchor="w", wraplength=210, justify="left").pack(fill="x", pady=(8, 0))

    def show_page(self, key: str) -> None:
        page = self.pages[key]
        page.tkraise()
        self.current_key = key
        for name, button in self.nav_buttons.items():
            button.set_active(name == key)
        page.on_show()


def _fatal(title: str, message: str) -> None:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()
    sys.exit(1)


def main() -> None:
    if sys.platform.startswith("win"):  # crisp text on high-DPI Windows displays
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    try:
        predictor = Predictor()
    except ModelNotFoundError as exc:
        _fatal("Model not found", str(exc))
    except ModelLoadError as exc:
        _fatal("Model cannot be loaded", str(exc))
    try:
        db = Database()
    except Exception as exc:
        _fatal("Database error", f"The history database could not be opened or created:\n\n{exc}")
    App(predictor, db).mainloop()


if __name__ == "__main__":
    main()
