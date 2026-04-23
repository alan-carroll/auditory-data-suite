"""
All Tkinter usage for the project lives here.

Two layers:
  * Primitives — thin wrappers over tkinter's filedialog / simpledialog /
    messagebox so callers never touch tk directly.
  * Composite dialogs — multi-widget flows (currently just the
    analysis picker) that need a real Tk window.

Every primitive runs inside `_hidden_root()`, which owns the
create/withdraw/destroy dance and guarantees the root is cleaned up
even if the dialog raises.
"""
import time
from contextlib import contextmanager
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, TclError

__all__ = [
    "get_file", "get_folder", "save_file", "ask_string", "confirm", 
    "load_analysis"
]

# -- internals -------------------------------------------------------------

_LAST_DIALOG_DIR = Path.cwd()

@contextmanager
def _hidden_root():
    """A withdrawn Tk root, destroyed on exit. Backbone of every primitive."""
    root = tk.Tk()
    root.withdraw()
    try:
        yield root
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _dialog_kwargs(kwargs):
    dialog_kwargs = dict(kwargs)
    dialog_kwargs.setdefault("initialdir", str(_LAST_DIALOG_DIR))
    return dialog_kwargs


def _remember_dialog_path(path, *, is_folder=False):
    global _LAST_DIALOG_DIR

    if not path:
        return

    picked = Path(path)
    _LAST_DIALOG_DIR = picked if is_folder else picked.parent

# -- file / folder primitives ---------------------------------------------

def get_folder(**kwargs):
    """
    Directory picker. Returns the path with a trailing slash, or empty
    string on cancel. (Trailing slash preserved for callers that build
    paths by string concat; TODO pathlib follow-up.)
    """
    with _hidden_root():
        folder = filedialog.askdirectory(**_dialog_kwargs(kwargs))
    _remember_dialog_path(folder, is_folder=True)
    return (folder + "/") if folder else ""

def get_file(**kwargs):
    """File-open picker. Returns path string, or empty string on cancel."""
    with _hidden_root():
        path = filedialog.askopenfilename(**_dialog_kwargs(kwargs))
    _remember_dialog_path(path)
    return path

def save_file(**kwargs):
    """File-save picker. Returns path string, or empty string on cancel."""
    with _hidden_root():
        path = filedialog.asksaveasfilename(**_dialog_kwargs(kwargs))
    _remember_dialog_path(path)
    return path

# -- simple prompt primitives ---------------------------------------------

def ask_string(title, prompt):
    """Single-line text prompt. Returns the string, or None on cancel."""
    with _hidden_root():
        return simpledialog.askstring(title, prompt)

def confirm(title, message):
    """Yes/No box. Returns bool."""
    with _hidden_root():
        return messagebox.askyesno(title, message)

def show_error(title, message):
    with _hidden_root():
        messagebox.showerror(title, message)

# -- subprocess focus pump -------------------------------------------------

def pump_until(done, interval_s=0.05):
    """
    Keep a hidden Tk root alive and pump its event loop until `done()`
    returns truthy. On macOS a parent process must keep servicing its
    NSApplication for a child window to take focus; on other platforms
    this is a harmless busy-sleep.
    """
    with _hidden_root() as root:
        while not done():
            try:
                root.update()
            except TclError:
                return
            time.sleep(interval_s)

# -- composite: analysis picker -------------------------------------------

def load_analysis(analyses):
    """
    Modal picker over an iterable of analysis-metadata dicts (each with
    name, comments, start_date, last_modified, frozen, _id).

    Returns (selection, create_new):
      * selection — the chosen metadata dict, or None
        if the user closed without choosing.
      * create_new — True if "Create New From Selected" was clicked.

    Frozen analyses cannot be loaded directly but may be used as a
    create-new template.
    """
    choices = {f"{a['name']}: {a['comments']}": a for a in analyses}
    result = {"selection": None, "create_new": False}

    root = tk.Tk()
    root.title("Select Analysis or Create New")
    frame = tk.Frame(root)
    frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    frame.pack(pady=100, padx=100)

    var = tk.StringVar(root)
    meta_var = tk.StringVar(root, value="name: \n\nstart_date: \n"
                                        "last_modified: \n\ncomments: \n\n"
                                        "frozen: \nid: \n\n")

    tk.Label(frame, textvariable=meta_var).grid(row=1, column=2)
    tk.Label(frame,
             text="Load an existing analysis, or create a new one (select "
                  "an existing analysis to use as a starting template):"
             ).grid(row=2, column=2)
    tk.OptionMenu(frame, var, *choices).grid(row=3, column=2)

    def _current():
        return choices.get(var.get())

    def on_change(*_):
        sel = _current()
        if sel is None:
            return
        meta_var.set(f"name: {sel['name']}\n\n"
                     f"start_date: {sel['start_date']}\n"
                     f"last_modified: {sel['last_modified']}\n\n"
                     f"comments: {sel['comments']}\n\n"
                     f"frozen: {sel['frozen']}\n"
                     f"id: {sel['_id']}\n")

    def on_load(*_):
        sel = _current()
        if sel is None:
            return
        if sel["frozen"]:
            messagebox.showerror(
                "Frozen",
                "Selected analysis is frozen and cannot be updated.")
            return
        result["selection"] = sel
        close()

    def on_create(*_):
        sel = _current()
        if sel is None:
            return
        result["selection"] = sel
        result["create_new"] = True
        close()

    def close(*_):
        root.quit()
        root.destroy()

    tk.Button(frame, text="Load Selected Analysis",
              command=on_load).grid(row=4, column=1)
    tk.Button(frame, text="Create New From Selected Analysis",
              command=on_create).grid(row=4, column=3)

    var.trace_add("write", on_change)
    root.bind("<Escape>", close)
    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()

    return result["selection"], result["create_new"]