import matplotlib
matplotlib.use("TkAgg")

import os
import sys
import time
import tkinter
import subprocess
import analysis_functions as afunc
import densetc_analysis
from colorama import Fore, Style
import logging
import json
import pandas as pd
from db_adapter import JSONStore
import cli_utils as cli


logging.basicConfig(filename="check_after_crash.log", level=logging.DEBUG)


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

    selection, create_new = afunc.load_analysis(meta_coll)
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
    pump = tkinter.Tk()
    pump.withdraw()
    p = subprocess.Popen(argv)
    try:
        while p.poll() is None:
            try:
                pump.update()
            except tkinter.TclError:
                p.wait()
                break
            time.sleep(0.05)
    finally:
        try:
            pump.destroy()
        except Exception:
            pass
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


if __name__ == "__main__":
    version = "1.0"
    continue_program = 1
    config_dict = {"project_name": None}
    while continue_program:
        print(Fore.CYAN + Style.BRIGHT)
        print("----------------------------------------------")
        print("|                  Alan's                    |")
        print(f"|            MAPPING ANALYSIS v{version}           |")
        print("|              Wonder Emporium               |")
        print("|                                            |")
        print("----------------------------------------------")
        print()
        print(f"Configuration loaded: {Fore.GREEN}"
              f"{config_dict['project_name']}{Fore.CYAN}")
        print()
        print("Available actions:")
        print(f" * [{Fore.WHITE}n{Fore.CYAN}]ew configuration file")
        print(f" * [{Fore.WHITE}l{Fore.CYAN}]oad project configuration")
        print(f" * [{Fore.WHITE}a{Fore.CYAN}]nalyze subject")
        print(f" * [{Fore.WHITE}g{Fore.CYAN}]enerate analysis from final file")
        print(f" * [{Fore.WHITE}o{Fore.CYAN}]cr template tools")
        print(f" * [{Fore.WHITE}s{Fore.CYAN}]elect fields GUI")
        print(f" * [{Fore.WHITE}f{Fore.CYAN}]inal-file generation")
        print(f" * [{Fore.WHITE}i{Fore.CYAN}]c final-file generation")
        print(f" * e[{Fore.WHITE}x{Fore.CYAN}]it program")
        print(Style.RESET_ALL)
        ch = input("> ").strip().lower()
        if ch == "l":
            try:
                with open(afunc.get_file(title="Load Configuration", 
                                         filetypes=[("JSON", ".json")])) as f:
                    config_dict = json.load(f)
            except Exception as e:
                cli.fail(e, "Failed to open file. Do better.")
        if ch == "n":
            try:
                if not (config_dict := afunc.create_config_file()):
                    config_dict = {"project_name": None}
            except Exception as e:
                cli.fail(e, "Something went terribly wrong. Scream into void.")
        if ch == "a":
            if config_dict["project_name"] is None:
                cli.warn("Load a project config file, or create a new one.")
                continue
            densetc_analysis.run_program(config_dict, version)
            cli.banner("\nIt's over! :)\n\n")
        if ch == "g":
            if config_dict["project_name"] is None:
                cli.warn("Load a project config file, or create a new one.")
                continue
            yes_or_no = cli.ask_yes_no(
                "Do you want an SDF calculated for each tuning curve PSTH "
                "[y/n]? (slower) > "
            )
            return_sdf = True if yes_or_no == "y" else False
            file = afunc.get_file(title="Select final file", 
                                  filetypes=[("XLS", ".xls")])
             # Use .xls final file. Uses v-plot format.
            usecols = [1,2,6,7,8,11,12,13,16,17,18,21,22,23,25,34,40,41,42,43]
            colnames = ["cf","thresh","bw10a","bw10b","bw10","bw20a","bw20b",
                        "bw20","bw30a","bw30b","bw30","bw40a","bw40b","bw40",
                        "onset","offset","x","y","field","number",]
            map_df = pd.read_excel(file, header=None, usecols=usecols, 
                                   names=colnames)
            densetc_analysis.run_program(config_dict, version, 
                                         final_file=map_df,
                                         return_sdf=return_sdf)
        if ch == "o":
            try:
                _ocr_template_tools()
            except Exception as e:
                logging.exception(e)
                print(Style.BRIGHT + Fore.RED +
                      f"OCR tool chaos: {e}" +
                      Style.RESET_ALL)
        if ch == "f":
            afunc.create_final_file()
        if ch == "i":
            afunc.create_final_file(ic_bool=True)
        if ch == "s":
            try:
                picked = _pick_map_and_analysis()
            except Exception as e:
                cli.fail(e, f"Couldn't open that database: {e}")
                continue
            if picked is None:
                continue
            rc = _launch_gui(*picked)
        if ch == "x":
            cli.info("Well fine ... \n")
            continue_program = 0
