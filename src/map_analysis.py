import matplotlib
matplotlib.use("TkAgg")

import os
import sys
import subprocess
import analysis_functions as afunc
import subject_analysis
from colorama import Fore, Style
import logging
import json
import pandas as pd
from dataclasses import dataclass, field
from db_adapter import JSONStore
import cli_utils as cli
from dialogs import pump_until


logging.basicConfig(filename="check_after_crash.log", level=logging.DEBUG)


@dataclass
class _MenuState:
    version: str
    config_dict: dict = field(default_factory=lambda: {"project_name": None})
    running: bool = True


def _pick_map_and_analysis():
    """
    Collect everything the GUI needs before it launches: cortical vs
    IC, the DB file, and which analysis to open (or create-new-from-
    template).

    Returns (db_path, analysis_id, is_ic), or None if the user backs
    out at the file dialog or analysis picker.
    """
    is_ic = cli.ask_choice("Cortical [c] or IC [i] map? > ", ("c", "i")) == "i"
    db_path = afunc.get_file(title="Select database JSON file",
                             filetypes=[("JSON", ".json")])
    if not db_path:
        return None

    db = JSONStore(db_path)
    meta_coll = db.analysis_metadata

    selection, create_new = afunc.load_analysis(meta_coll.find({}))
    if selection is None:
        return None

    if not create_new:
        return db_path, selection["_id"], is_ic

    # Create-new clones the chosen analysis as a template, across BOTH
    # the cortical and IC analysis collections so the new analysis_id
    # is valid whichever side is opened later. "main" is whichever
    # matches is_ic; "bonus" is the other. An empty bonus collection is
    # harmless if the project has no data for that side.
    if is_ic:
        main_coll = db.densetc_IC_analysis
        bonus_coll = db.densetc_analysis
    else:
        main_coll = db.densetc_analysis
        bonus_coll = db.densetc_IC_analysis
    new_meta = afunc.new_analysis_metadata_document()
    analysis_id = afunc.create_new_densetc_analysis(
        selection["_id"], new_meta, meta_coll, main_coll, bonus_coll)
    return db_path, analysis_id, is_ic


def _launch_gui(db_path, analysis_id, is_ic):
    """
    Run Field_Selection_GUI.py in its own interpreter and block until
    it exits, returning the child's exit code.

    The GUI runs out-of-process so Kivy/SDL never initializes here
    (SDL and Tk cannot share macOS's NSApplication singleton, leading
    to crashes, and the separate process enables running the GUI
    multiple times within the same CLI run).

    An additional poll loop pumps a hidden Tk root, which keeps the
    parent responsive enough to allow the child to take focus
    (necessary on macOS, harmless on Windows/Linux). Side effect is
    creation of an extra Python icon while the app is running.
    """
    gui_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Field_Selection_GUI.py")
    argv = [sys.executable, gui_script,
            "--db", db_path, "--analysis-id", str(analysis_id)]
    if is_ic:
        argv.append("--ic")
    p = subprocess.Popen(argv)
    pump_until(lambda: p.poll() is not None)
    p.wait()  # if pump exited via TclError, child may still be running
    return p.returncode


def _ocr_template_tools():
    # TODO
    print("Not yet implemented.")
    return

    while True:
        print(Fore.CYAN + Style.BRIGHT)
        print("\nOCR template tools:")
        print(f" * [{Fore.WHITE}b{Fore.CYAN}]ootstrap new template set")
        print(f" * [{Fore.WHITE}p{Fore.CYAN}]review template set")
        print(f" * e[{Fore.WHITE}x{Fore.CYAN}]it OCR tools")
        print(Style.RESET_ALL)

        # TODO finish implementing
        ch = input("> ").strip().lower()
        if ch == "b":
            pass
        elif ch == "p":
            pass
        elif ch == "x":
            break


def _require_config(state):
    """Guard for actions that need a loaded project configuration."""
    if state.config_dict["project_name"] is None:
        cli.warn("Load a project config file, or create a new one.")
        return False
    return True


def _action_new_config(state):
    try:
        cfg = afunc.create_config_file()
        if cfg:
            state.config_dict = cfg
    except Exception as e:
        cli.fail(e, "Something went terribly wrong. Scream into void.")


def _action_load_config(state):
    path = afunc.get_file(title="Load Configuration",
                          filetypes=[("JSON", ".json")])
    if not path:
        return
    try:
        with open(path) as f:
            state.config_dict = json.load(f)
    except Exception as e:
        cli.fail(e, "Failed to open file. Do better.")


def _action_analyze(state):
    if not _require_config(state):
        return
    try:
        subject_analysis.run_program(state.config_dict, state.version)
        cli.banner("\nIt's over! :)\n\n")
    except Exception as e:
        cli.fail(e, f"Analysis crashed: {e}\n"
                    "Traceback in check_after_crash.log.")


def _action_generate_from_final(state):
    if not _require_config(state):
        return
    return_sdf = cli.ask_yes_no(
        "Do you want an SDF calculated for each tuning curve PSTH "
        "[y/n]? (slower) > "
    )
    file = afunc.get_file(title="Select final file",
                          filetypes=[("XLS", ".xls")])
    if not file:
        return
    try:
        # Use .xls final file. Uses v-plot format.
        usecols = [1,2,6,7,8,11,12,13,16,17,18,21,22,23,25,34,40,41,42,43]
        colnames = ["cf","thresh","bw10a","bw10b","bw10","bw20a","bw20b",
                    "bw20","bw30a","bw30b","bw30","bw40a","bw40b","bw40",
                    "onset","offset","x","y","field","number",]
        map_df = pd.read_excel(file, header=None, usecols=usecols,
                               names=colnames)
        subject_analysis.run_program(state.config_dict, state.version,
                                     final_file=map_df, return_sdf=return_sdf)
    except Exception as e:
        cli.fail(e, f"Final-file generation crashed: {e}\n"
                    "Traceback in check_after_crash.log.")


def _action_ocr_tools(state):
    try:
        _ocr_template_tools()
    except Exception as e:
        cli.fail(e, f"OCR tool chaos: {e}")


def _action_select_fields(state):
    try:
        picked = _pick_map_and_analysis()
    except Exception as e:
        cli.fail(e, f"Couldn't open that database: {e}")
        return
    if picked is not None:
        _launch_gui(*picked)


def _action_exit(state):
    cli.info("Well fine ... \n")
    state.running = False


# (label, handler) pairs. Label uses [k] to mark the highlighted key
# letter; _print_menu styles it. Dict order is menu order.
_ACTIONS = {
    "n": ("[n]ew configuration file",            _action_new_config),
    "l": ("[l]oad project configuration",        _action_load_config),
    "a": ("[a]nalyze subject",                   _action_analyze),
    "g": ("[g]enerate analysis from final file", _action_generate_from_final),
    "o": ("[o]cr template tools",                _action_ocr_tools),
    "s": ("[s]elect fields GUI",                 _action_select_fields),
    "f": ("[f]inal-file generation",
          lambda s: afunc.create_final_file()),
    "i": ("[i]c final-file generation",
          lambda s: afunc.create_final_file(ic_bool=True)),
    "x": ("e[x]it program",                      _action_exit),
}


def _print_menu(state):
    print(Fore.CYAN + Style.BRIGHT)
    print("----------------------------------------------")
    print("|                  Alan's                    |")
    print(f"|            MAPPING ANALYSIS v{state.version}           |")
    print("|              Wonder Emporium               |")
    print("|                                            |")
    print("----------------------------------------------")
    print()
    print(f"Configuration loaded: {Fore.GREEN}"
          f"{state.config_dict['project_name']}{Fore.CYAN}")
    print()
    print("Available actions:")
    for label, _ in _ACTIONS.values():
        styled = (label.replace("[", f"[{Fore.WHITE}")
                       .replace("]", f"{Fore.CYAN}]"))
        print(f" * {styled}")
    print(Style.RESET_ALL)


if __name__ == "__main__":
    state = _MenuState(version="1.0")
    while state.running:
        _print_menu(state)
        ch = input("> ").strip().lower()
        action = _ACTIONS.get(ch)
        if action:
            action[1](state)
