import matplotlib
matplotlib.use("TkAgg")

import analysis_functions as afunc
import densetc_analysis
import colorama
from colorama import Fore, Style, Back
import logging
import json
import pandas as pd

colorama.init()
logging.basicConfig(filename="check_after_crash.log", level=logging.DEBUG)


if __name__ == "__main__":
    version = "1.0"
    continue_program = 1
    app_run = False
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
                logging.exception(e)
                print(Style.BRIGHT+Fore.RED+
                      "Failed to open file. Do better."+
                      Style.RESET_ALL)
        if ch == "n":
            try:
                if not (config_dict := afunc.create_config_file()):
                    config_dict = {"project_name": None}
            except Exception as e:
                logging.exception(e)
                print(Style.BRIGHT+Fore.RED+
                      "Something went terribly wrong. Scream into void."+
                      Style.RESET_ALL)
        if ch == "a":
            if config_dict["project_name"] is None:
                print(Style.BRIGHT+Fore.YELLOW+
                      "Load a project config file or create a new one first."+
                      Style.RESET_ALL)
                continue
            densetc_analysis.run_program(config_dict, version)
            print(Style.BRIGHT+Back.GREEN+
                  "\nIt's over! :)\n\n"+
                  Style.RESET_ALL)
        if ch == "g":
            if config_dict["project_name"] is None:
                print(Style.BRIGHT+Fore.YELLOW+
                      "Load a project config file or create a new one first."+
                      Style.RESET_ALL)
                continue
            while (yes_or_no := input("Do you want an SDF calculated for each "
                                      "tuning curve PSTH [y/n]? (slower) > ")):
                continue
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
        if ch == "f":
            afunc.create_final_file()
        if ch == "i":
            afunc.create_final_file(ic_bool=True)
        if ch == "s":
            if app_run:
                print("GUI can only be run once per session."
                      " Exit and restart program to re-run GUI! Sorry :(")
                continue
            app_run = True
            import Field_Selection_GUI
            app = Field_Selection_GUI.FieldSelectionApp()
            app.run()
        if ch == "x":
            print(Style.BRIGHT+Fore.YELLOW+"Well fine ... \n"+Style.RESET_ALL)
            continue_program = 0
