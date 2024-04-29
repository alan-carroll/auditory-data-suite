import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import itertools
import json
import bisect
import pandas as pd
#import tkinter as tk
import mttkinter.mtTkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import re
from collections import defaultdict
import burst_detection as bd
from shapely.ops import cascaded_union, polygonize
import math
from scipy.spatial import Delaunay
import numpy as np
import shapely.geometry as geometry
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
import logging
import uuid
import colorama
import datetime
import os
from tinymongo_fix.tinymongo_fix import TinyMongoClient
from shapely.geometry import Point
import voronoi_picker
from scipy.spatial import Voronoi
from scipy.stats import ttest_ind
import colorama
from colorama import Fore, Style


def get_folder(**kwargs):
    """
    Open simple Tk dialog to select a directory.
    Returns string containing dir path
    """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(**kwargs) + "/"
    root.destroy()
    return folder


def get_file(**kwargs):
    """
    Open simple Tk dialog to select a file.
    Returns string containing dir path + filename
    """
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filename


def save_file(**kwargs):
    """
    Open simple Tk dialog to save a filename + dir
    Returns string pointing to save location
    """
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.asksaveasfilename(**kwargs)
    root.destroy()
    return filename


def create_config_file():
    """
    Create a new analysis config file. These simple config files will allow 
    users to make project specific adjustments to the analysis process (such as
    analyzing new stimulus sets) or auto attaching comments / metadata with
    each subject. These also prevent the need to hard code and require specific
    file naming conventions, and allow a user to run multiple analyses for a 
    project without having to manually specify the same information over and 
    over again for each new subject in a project.
    
    Current implementation is stupid-simple and just asks a list of questions
    about known possible sets/data.
    """
    print(colorama.Style.BRIGHT+colorama.Fore.YELLOW+
          "Select a location and file name to save your config file to: "+
          colorama.Style.RESET_ALL)
    save_location = save_file(title="Save configuration file", 
                              filetypes=[("JSON", ".json")])
    print(save_location+".json")

    creation_date = str(datetime.datetime.now())
    project_name = input("What is the project name? > ")
    config_id = str(uuid.uuid4()).replace(u"-", u"")
    config_dict = {
        "config_created_on": creation_date,
        "project_name": project_name,
        "config_id": config_id,
    }

    # Tuning curve configuration options
    densetc_confirm = False
    while not densetc_confirm:
        while (densetc_bool := input("Are you doing TC analysis [y/n]? > ")
               .lower().strip()) not in ["y", "n"]:
            continue
        if densetc_bool == "n":
            config_dict["do_densetc"] = 0
            densetc_confirm = True
            break
        
        print("What do your file names uniquely start with? \n"
              "This will associate data files with TC analysis.\n"+
              colorama.Fore.MAGENTA+
              "eg. For 'DenseTC_MPK_digitalatten_JRAC#001G_RZ5-1_007.src', "
              "type 'DenseTC' (without quotes, case-sensitive)."+
              colorama.Style.RESET_ALL)
        densetc_filename = input("> ")

        print(colorama.Fore.YELLOW+
              "\nSelect .csv file containing list of frequencies (Hz) and "
              "intensities (dB SPL) used.\n"+
              colorama.Fore.CYAN+
              "MUST be Frequency column THEN Intensity column.\n"+
              colorama.Fore.YELLOW+
              "Each row corresponds to a tone (no headers, just values):"+
              colorama.Style.RESET_ALL)
        try:
            densetc_df = pd.read_csv(
                get_file(title="Open DenseTC .csv tone list", 
                         filetypes=[("CSV", ".csv")]), 
                header=None, names=["frequency", "intensity"])

            print("\nThe frequencies range from: "
                  f"{min(densetc_df['frequency'].values)} Hz to "
                  f"{max(densetc_df['frequency'].values)} Hz."
                  "\nThe intensities range from: "
                  f"{min(densetc_df['intensity'].values)} dB to "
                  f"{max(densetc_df['intensity'].values)} dB")
            print(f"There are {len(densetc_df)} total tones.")
            print(f"'{densetc_filename}' will be used to identify your files "
                  "for tuning curve analysis.")
            while (yes_no := input("\nIs this correct [y/n]? > ")
                   .lower().strip()) not in ["y", "n"]:
                continue
            if yes_no == "n":
                continue
            densetc_confirm = True
            config_dict["do_densetc"] = 1
            config_dict["densetc_file"] = densetc_filename
            config_dict["densetc_frequency_hz"] = np.unique(
                densetc_df["frequency"].values).tolist()
            config_dict["densetc_intensity_db"] = np.unique(
                densetc_df["intensity"].values).tolist()
            config_dict["densetc_num_tones"] = len(densetc_df)

        except Exception as exp:
            logging.exception(exp)
            print(colorama.Fore.RED+
                  "*** Selected file caused an error. Double check it is "
                  "correct, or scream into void. ***\n"+
                  colorama.Style.RESET_ALL)

    # Speech configuration options
    speech_confirm = False
    while not speech_confirm:
        while (speech_bool := input("Are you doing speech analysis [y/n]? > ")
               .lower().strip()) not in ["y", "n"]:
            continue
        if speech_bool == "n":
            config_dict["do_speech"] = 0
            speech_confirm = True
            break
        
        print("What do your file names uniquely start with? \n"
              "This will associate data files with speech analysis.\n"+
              colorama.Fore.MAGENTA+
              "eg. For 'vnsspeech_60dB_RZ5_w5dBdummyPA5#001G_RZ5-1_007.src', "
              "type 'vnsspeech' (without quotes, case-sensitive)."+
              colorama.Style.RESET_ALL)
        speech_filename = input("> ")

        print(colorama.Fore.YELLOW+
              "\nSelect .csv file containing list of speech sounds "
              "(name/description) and numbers (integers) used.\n"+
              colorama.Fore.CYAN+
              "MUST be Description column THEN Number column.\n"+
              colorama.Fore.YELLOW+
              "Each row corresponds to a sound (no headers, just values):"+
              colorama.Style.RESET_ALL)
        try:
            speech_df = pd.read_csv(
                get_file(title="Open Speech .csv list",
                         filetypes=[("CSV", ".csv")]),
                header=None, names=["speech", "number"])

            print("\n"+speech_df.to_string(index=False))
            print(f"There are {len(speech_df)} total speech sounds.")
            print(f"'{speech_filename}' will be used to identify your files "
                  "for speech analysis.")
            while (yes_no := input("\nIs this correct [y/n]? > ")
                   .lower().strip()) not in ["y", "n"]:
                continue
            if yes_no == "n":
                continue
            speech_confirm = True
            config_dict["do_speech"] = 1
            config_dict["speech_file"] = speech_filename
            config_dict["speech"] = [{"number": row.number, 
                                      "speech": row.speech} for row in 
                                     speech_df.itertuples()]

        except Exception as exp:
            logging.exception(exp)
            print(colorama.Fore.RED+
                  "*** Selected file caused an error. Double check it is "
                  "correct, or scream into void. ***\n"+
                  colorama.Style.RESET_ALL)

    # Noiseburst configuration options
    burst_confirm = False
    while not burst_confirm:
        while (burst_bool := input("Are you doing noiseburst analysis [y/n]?"
                                   " > ").lower().strip()) not in ["y", "n"]:
            continue
        
        if burst_bool == "n":
            config_dict["do_burst"] = 0
            burst_confirm = True
            break
        print("What do your file names uniquely start with? \n"
              "This will associate data files with noiseburst analysis.\n"+
              colorama.Fore.MAGENTA+
              "eg. For 'bb_noise_train#001G1_7.src', "
              "type 'bb_noise' (without quotes, case-sensitive)."+
              colorama.Style.RESET_ALL)
        burst_filename = input("> ")

        print(colorama.Fore.YELLOW +
              "\nSelect .csv file containing list of noise-burst ISIs (ms) "
              "and number of bursts (integers) used.\n"+
              colorama.Fore.CYAN+
              "MUST be ISI column THEN Number column.\n"+
              colorama.Fore.YELLOW+
              "Each row corresponds to noise stim (no headers, just values):"+
              colorama.Style.RESET_ALL)
        try:
            burst_df = pd.read_csv(
                get_file(title="Open noiseburst .csv parameters list",
                         filetypes=[("CSV", ".csv")]),
                header=None, names=["ISI", "number"])

            print("\n"+burst_df.to_string(index=False))
            print(f"\nThere are {len(burst_df)} total noiseburst stimuli.")
            print(f"\n'{burst_filename}' will be used to identify your files "
                  "for noiseburst analysis.")
            while (yes_no := input("\nIs this correct [y/n]? > ")
                   .lower().strip()) not in ["y", "n"]:
                continue
            if yes_no == "n":
                continue
            burst_confirm = True
            config_dict["do_burst"] = 1
            config_dict["burst_file"] = burst_filename
            config_dict["burst"] = [{"ISI_ms": row.ISI, 
                                     "num_bursts": row.number} for row in 
                                    burst_df.itertuples()]

        except Exception as exp:
            logging.exception(exp)
            print(colorama.Fore.RED+
                  "*** Selected file caused an error. Double check it is "
                  "correct, or scream into void. ***\n"+
                  colorama.Style.RESET_ALL)

    # IC mapping configuration
    ic_confirm = False
    while not ic_confirm:
        while (ic_bool := input("Will ANY subjects in this project have IC "
                                "maps [y/n]? > ").lower().strip()
               ) not in ["y", "n"]:
            continue
        
        if ic_bool == "n":
            config_dict["do_IC"] = 0
            ic_confirm = True
            break
        print(colorama.Back.MAGENTA+
              "\nWhen analyzing subjects in this project, you will be "
              "prompted to indicate which subjects have IC data."
              "\nAny subject with IC data requires an additional .csv file "
              "listing the mapping Penetration numbers associated with IC "
              "files, and the Depths at those sites."
              "\nFilenames for any stimulus presented in IC maps are assumed "
              "to use the same naming pattern as files for Cortical maps.\n\n"+
              colorama.Style.RESET_ALL)
        config_dict["do_IC"] = 1
        ic_confirm = True

    # Save configuration dict to .json file
    try:
        with open(save_location+".json", "w") as file:
            json.dump(config_dict, file, indent=4)

        print(colorama.Back.GREEN+
              f"\nSaved config file {save_location+'.json'} !! :)"+
              colorama.Style.RESET_ALL)
        return config_dict

    except Exception as exp:
        logging.exception(exp)
        print(colorama.Fore.RED+
              "Something went horribly wrong during saving! Why? WHY?! :( :("+
              colorama.Style.RESET_ALL)


def pick_voronoi(map_points_df, map_width, map_height):
    """
    Generate Voronoi tessellation for map and store data.

    Uses Shapely for geometry -- https://shapely.readthedocs.io/en/latest/

    Creates extended border around original map points to eliminate voronoi 
    cells going to infinity. Newly created points are buffered away from 
    existing border points by the average edge length between border points.
    
    Returns a list of sites containing xy coords and voronoi vertices for all 
    real map points, and the list of extra points used to define boundaries.
    """
    base_pts = map_points_df[["x", "y"]].values
    shape_pts = [Point(pnt) for pnt in base_pts]
    
    # alpha=5 for relaxed boundaries, 10 for tighter
    concave_hull, edge_points = alpha_shape(shape_pts, alpha=8)

    # 3 = Square cap; 3 = Bevel join
    perimeter_length = concave_hull.exterior.length
    num_perimeter_pts = len(concave_hull.exterior.coords)
    avg_edge_length = perimeter_length / num_perimeter_pts
    bonus = concave_hull.boundary.buffer(avg_edge_length, 
                                         cap_style=3, 
                                         join_style=3)
    bonus_pts = np.array([[x, y] for x, y in bonus.exterior.coords])

    # Use vispy-based voronoi program to pick additional border points
    print(Style.BRIGHT+Fore.MAGENTA+
          "Add additional border points to voronoi diagram as necessary."
          "\nLeft-click adds a point."
          "\nRight-click removes last added point."
          "\n<Esc> or exit window to accept points and continue."+
          Style.RESET_ALL)
    bonus_pts = voronoi_picker.pick_points(size=(round(map_width / 1.5), 
                                                 round(map_height / 1.5)),
                                           input_points=base_pts, 
                                           buffer_points=bonus_pts)
    vor_input = list(base_pts) + list(bonus_pts)
    vor = Voronoi(vor_input)

    sites_list = []
    for idx in map_points_df.index:
        xy = map_points_df.iloc[idx].loc[['x', 'y']].values
        map_num = int(map_points_df.iloc[idx].number)
        vor_indices = np.where(vor.points == xy)[0]
        
        """
        Sometimes 2+ points share an identical floating point coordinate value 
        on either the x or y position. These cases will match a single coord 
        against voronoi indices, and can result in a duplicate polygon in one 
        xy position while failing to draw its own polygon.
        
        This problem is solved by identifying and selecting the index which 
        matches twice (both x and y coordinates).
        """
        matches, counts = np.unique(vor_indices, return_counts=True)
        vor_index = matches[counts == 2][0]

        region_idx = vor.point_region[vor_index]
        vertex_idx = vor.regions[region_idx]
        polygon = vor.vertices[vertex_idx]
        centroid = geometry.Polygon(polygon).centroid.coords.xy

        sites_list.append({
            "number": map_num,
            "x": xy[0],
            "y": xy[1],
            "voronoi_centroid": np.array([centroid[0][0], 
                                          centroid[1][0]]).tolist(),
            "voronoi_vertices": polygon.tolist(),
        })
        
    return sites_list, bonus_pts


def check_voronoi(sites_list, bonus_pts):
    """
    Simple figure showing voronoi diagram and associated map #'s
    """
    _, ax = plt.subplots()
    for row in sites_list:
        polygon = row["voronoi_vertices"]
        ax.fill(*zip(*polygon), alpha=0.4)
        ax.text(row["x"], row["y"], row["number"])
    ax.plot(*zip(*bonus_pts), "bd")

    plt.ion()
    plt.show()


def check_points(numbers_image, points_dict):
    """
    Simple figure matching map #'s with points.
    Used to test/debug initial map image OCR work.
    """
    _, ax = plt.subplots()
    ax.imshow(numbers_image)
    for i in points_dict:
        ax.plot(points_dict[i]["point"][1],
                points_dict[i]["point"][0],
                'ko', ms=5)
        ax.text(points_dict[i]["point"][1],
                points_dict[i]["point"][0],
                points_dict[i]["ocr"],
                color="#5A86AD")
    plt.ion()
    plt.show()


def check_csv_points(points_df):
    """
    Simple figure matching map #'s with points.
    Used to test/debug initial map .csv work.
    """
    _, ax = plt.subplots()
    ax.scatter(points_df.x, points_df.y)
    for row in points_df.itertuples():
        ax.text(row.x, row.y, row.number)
    plt.ion()
    plt.show()


def snap(grid_vals, input_value, allow_zero=True):
    """
    Useful function to snap any cf, thresh, BW's, freq or int to nearest values 
      of known input, grid_vals.
      eg. grid_vals is list of possible frequencies, 
          input_val is frequency slightly off from grid of possible vals.
          
    Returns the value in grid_vals that input_value is closest to.
    
    Idiosyncrasy: 'final file' values use 0 to indicate a null value. 
      To deal with this and properly return a null value instead of a snapped 
      value, pass keyword argument allow_zero=False.
    """
    if (input_value == 0) and (allow_zero is False):
        return None

    ix = bisect.bisect_right(grid_vals, input_value)
    if ix == 0:
        return grid_vals[0]
    elif ix == len(grid_vals):
        return grid_vals[-1]
    else:
        return min(grid_vals[ix - 1], grid_vals[ix], 
                   key=lambda grid_value: abs(grid_value - input_value))


def scale_coordinates(input_coor, min_coor, max_coor, min_scale, max_scale):
    """
    Simple scaling function for map coordinates.
    """
    return ((max_scale - min_scale) * (input_coor - min_coor) / 
            (max_coor - min_coor)) + min_scale
    

def get_bandwidth(bw_start, bw_stop):
    """
    Expects a start and stop frequency (Hz) marking the edges of the bandwidth.
    Uses log2 to transform back into an octave range.
    """
    return np.log2(bw_stop / bw_start)


def load_analysis(analysis_metadata_collection):
    """
    Load menu allowing user to choose existing analysis or create a new one.
    
    Takes a tinymongo collection as input.
    Frozen analyses cannot be loaded from this function.
    
    Returns a pandas Series of analysis metadata (or None if user exits menu),
      and a new_analysis flag (True/False) to indicate if selected analysis 
      should be loaded or a new analysis created from the selection.
    """
    global selected_analysis
    global new_analysis
    selected_analysis = None
    new_analysis = False
    existing_analyses = pd.DataFrame([entry for entry in 
                                      analysis_metadata_collection.find({})])
    choices = {f"{val[0]}: {val[1]}": idx for idx, val in
               enumerate(existing_analyses[["name", "comments"]].values)}

    root = tk.Tk()
    root.title("Select Analysis or Create New")
    mainframe = tk.Frame(root)
    mainframe.grid(column=0, row=0, stick=(tk.N, tk.W, tk.E, tk.S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)
    mainframe.pack(pady=100, padx=100)

    tk_var = tk.StringVar(root)
    popup_menu = tk.OptionMenu(mainframe, tk_var, *choices)
    tk.Label(mainframe, 
             text="Load an existing analysis, or create a new one (select an "
             "existing analysis to use as a starting template):"
             ).grid(row=2, column=2)
    metadata_var = tk.StringVar(root)
    metadata_var.set("name: \n\nstart_date: \nlast_modified: \n\ncomments: "
                     "\n\nfrozen: \nid: \n\n")
    tk.Label(mainframe, textvariable=metadata_var).grid(row=1, column=2)
    popup_menu.grid(row=3, column=2)

    def change_dropdown(*args):
        selection = existing_analyses.iloc[choices[tk_var.get()]]
        metadata_var.set(f"name: {selection['name']}\n\n"
                         f"start_date: {selection['start_date']}\n"
                         f"last_modified: {selection['last_modified']}\n\n"
                         f"comments: {selection['comments']}\n\n"
                         f"frozen: {selection['frozen']}\n"
                         f"id: {selection['_id']}\n")

    def select_analysis(*args):
        global selected_analysis
        selection = tk_var.get()
        if not selection:
            return

        selected_analysis = existing_analyses.iloc[choices[selection]]
        if selected_analysis['frozen']:
            selected_analysis = None
            simpledialog.messagebox.showerror(
                "Frozen", 
                "Selected analysis is frozen, and cannot be updated.")
            return
        close_window()

    def create_analysis(*args):
        global selected_analysis
        global new_analysis
        selection = tk_var.get()
        if not selection:
            return

        selected_analysis = existing_analyses.iloc[choices[selection]]
        new_analysis = True
        close_window()

    def close_window(*args):
        root.quit()
        root.destroy()

    tk.Button(mainframe, text="Load Selected Analysis", 
              command=select_analysis).grid(row=4, column=1)
    tk.Button(mainframe, text="Create New From Selected Analysis", 
              command=create_analysis).grid(row=4, column=3)

    tk_var.trace("w", change_dropdown)

    root.bind("<Escape>", lambda e: close_window())
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()
    
    return selected_analysis, new_analysis


def create_new_densetc_analysis(template_id, new_metadata, 
                                analysis_metadata_collection, 
                                densetc_analysis_collection, 
                                bonus_analysis_collection):
    """
    Create a new analysis for a subject.
    Adds metadata and duplicates entries from an existing analysis.
    
    Expects a dictionary of new analysis metadata and the analysis metadata and
      densetc_analysis tinymongo collections to update. Duplicates existing 
      analysis and replaces id with new analysis id.
      
    Returns new analysis_metadata _id.
    """
    # TODO allow blank analysis, with just empty fields
    analysis_id = analysis_metadata_collection.insert_one(
        new_metadata).inserted_id
    # NEW: Must wrap dict(site), stupid tinydb change. Won't insert otherwise.
    #  https://github.com/msiemens/tinydb/issues/354
    template_analysis = [
        dict(site) for site in 
        densetc_analysis_collection.find({"analysis_id": template_id})]
    # If there are no IC/cortical sites, an empty collection is created in the 
    #   database, harming no one
    # If there are, they are duplicated like expected and can be accessed from
    #   the same analysis ID
    bonus_analysis = [
        dict(site) for site in 
        bonus_analysis_collection.find({"analysis_id": template_id})]
    for site in template_analysis:
        site["analysis_id"] = analysis_id
        del site["_id"]
    for site in bonus_analysis:
        site["analysis_id"] = analysis_id
        del site["_id"]
    densetc_analysis_collection.insert_many(template_analysis)
    bonus_analysis_collection.insert_many(bonus_analysis)
    
    return analysis_id


def new_analysis_metadata_document():
    """
    Creates a new analysis_metadata document. 
    
    Returns a dictionary containing metadata with fields corresponding to
    analysis_metadata collection of tinymongo database
    """
    today = str(datetime.datetime.now())
    root = tk.Tk()
    root.withdraw()
    
    name = ""
    comments = ""
    answer = False
    while not answer:
        while (name := simpledialog.askstring(
            "Input Name", "Who is doing the analysis?")) is None:
            continue
        while (comments := simpledialog.askstring(
            "Comment", "Write a brief comment about analysis:")) is None:
            continue
        answer = messagebox.askyesno(
            "Verify", 
            f"Is this correct?\nName: {name}\n\nComments: {comments}")

    analysis_metadata = {
        "name": name,
        "comments": comments,
        "start_date": today,
        "last_modified": today,
        "frozen": False,
    }
    root.destroy()
    
    return analysis_metadata
    

def get_map_number(filename):
    """
    Parses filename for penetration and electrode #'s, then returns map #
    Assumes common filenaming patterns from our lab's Brainware files:
      eg. src -> DenseTC_MPK_digitalatten_JRAC#001G_RZ5-1_007.src
          f32 -> naive2dense_001e1.f32
    Tries a few rarer alternatives for .src naming convention before raising
    an error on non-conforming filename.
    
    (electrode_pattern, penetration_pattern):
    .src:
      Standard: (3 #'s then ".src", 3 #'s preceded by "#")
      f32-style: (1-3 #'s then ".src", 3 #'s followed by "e")
      f32-style w/ underscore: (1-3 #'s then ".src", 3 #'s preceded by "_")
    .f32:
      Standard: (1 # then ".f32", 3 #'s followed by "e")
    """
    ext = filename[-3:]
    if ext == "src":
        patterns = [
            ("[0-9]{3}(?=\.src)", "(?<=\#)[0-9]{3}"),
            ("[0-9]{1,3}(?=\.src)", "[0-9]{3}(?=e)"),
            ("[0-9]{1,3}(?=\.src)", "(?<=_)[0-9]{3}"),
            ]
    elif ext == "f32":
        patterns = [("[0-9]{1}(?=\.f32)", "[0-9]{3}(?=e)")]
    else:
        raise ValueError("Expected file extension of either 'src' or 'f32'"
                         f"\nGot {ext} from {filename}")
        
    found_match = False
    for (electrode_re, penetration_re) in patterns:
        try:
            elec_int = int(re.search(electrode_re, filename).group())
            pen_num = int(re.search(penetration_re, filename).group())
            found_match = True
            break
        except AttributeError:
            # No match, try again
            continue
    if not found_match:
        raise AttributeError(f"Can't parse map # from filename {filename}."
                             "\nFilename did not match any known pattern.")
    
    if (elec_int == 7) or (elec_int == 1):
        elec_num = 1
    elif (elec_int == 8) or (elec_int == 2):
        elec_num = 2
    elif (elec_int == 9) or (elec_int == 3):
        elec_num = 3
    elif (elec_int == 10) or (elec_int == 4):
        elec_num = 4
    else:
        raise ValueError("Expected electrode #'s between 1-4 or 7-10. "
                         f"Found number {elec_int}")

    return ((pen_num - 1) * 4) + elec_num


def get_penetration_number(filename):
    """
    Short version of get_map_number(), just returns penetration number. 
    Useful for IC files.
    Assumes common filenaming patterns from our lab's Brainware files:
      eg. src -> DenseTC_MPK_digitalatten_JRAC#001G_RZ5-1_007.src
          f32 -> naive2dense_001e1.f32
    Tries a few rarer alternatives for .src naming convention before raising
    an error on non-conforming filename.
    .src:
      Standard: 3 #'s preceded by "#"
      f32-style: 3 #'s followed by "e"
      f32-style w/ underscore: 3 #'s preceded by "_"
    .f32:
      Standard: (1 # then ".f32", 3 #'s followed by "e")
    """
    ext = filename[-3:]
    if ext == "src":
        patterns = [
            "(?<=\#)[0-9]{3}", 
            "[0-9]{3}(?=e)", 
            "(?<=_)[0-9]{3}",
            ]
    elif ext == "f32":
        patterns = ["[0-9]{3}(?=e)"]
    else:
        raise ValueError("Expected file extension of either 'src' or 'f32'"
                         f"\nGot {ext} from {filename}")
    for penetration_re in patterns:
        try:
            pen_num = int(re.search(penetration_re, filename).group())
            return pen_num
        except AttributeError:
            # No match, try again
            continue
    raise AttributeError(f"Can't parse penetration # from filename {filename}."
                         "\nFilename did not match any known pattern.")


def adjust_numbers(number):
    """
    Adjusts 'final file' format map number to real map numbers.
      Format is pen #, strictly 2 digit electrode #, eg.
        Penetration 1, electrodes 1-3: 101 -> 1, 102 -> 2, 103 -> 3 ...
        Penetration 5, electrodes 1-2: 501 -> 17, 502 -> 18, etc.
    """
    if not (100 < number < 10000):
        raise ValueError("Expected number: 100 < num < 10000"
                         "\nDo you really have over 100 penetrations? Wow!")
    str_number = str(number)
    if len(str_number) == 4:
        penetration_number = int(str_number[0:2])
    else:
        penetration_number = int(str_number[0])
    electrode_number = int(str_number[-1])
    map_number = 4 * (penetration_number - 1) + electrode_number
    return map_number


def get_spike_dict(blk, use_f32=False, dataset=None):
    """
    Takes a block (neo-processed brainware file) and a 'dataset' for a 
      stimulus set (tuning curves, speech, etc.)
    Returns a dict with entries for each set of stimulus parameters presented
      during recording and the resulting driven spiketimes.
      The keys are tuples of the stimulus parameters, and spiketimes held in
      a list of lists (one for each presentation of the stimulus).
      eg. {(freq,int): [[..spiketime vals in ms..]]} ->
          {(1000,20): [[8.074, 9.6783, 16.9794, 63.5782, 150.2598]]}
      eg. for 20 repeats of speech: 
        {(speech_num): [[sweep 1 spikes], [sweep 2 spikes], [...] ...]}
    
    Dataset parameters are from Brainware Stimulus set parameters.
    More parameters exist in some stimulus sets. The ones listed here are the
    ones known to be relevant to analysis. Any new stimulus set with additional
    parameters should be added here later.
    
    .src files store the parameter names directly.
    .f32 files only store vague 'Param0', 'Param1', etc. These correspond to
    the .src parameters in the order that Brainware saves them. If you add a
    new dataset here later for f32, make sure 'ParamX' matches what you expect.
    Dataset types:
      densetc: freq [Hz], int [dB]
      speech: offset
      burst: RepSepNoise [msec]
    """
    known_datasets = {
        "densetc": {"src": ["freq [Hz]", "int [dB]"], 
                    "f32": ["Param0", "Param1"]},
        "speech": {"src": ["offset"],
                   "f32": ["Param0"]}, 
        "burst": {"src": ["RepSepNoise [msec]"],
                  "f32": ["Param0"]}
        }
    if dataset not in known_datasets:
        raise ValueError("Must specify dataset type to parse segment with."
                         f"\nChoose from {list(known_datasets.keys())}")
    params = known_datasets[dataset]
    
    spike_dict = defaultdict(list)
    for seg in blk.segments:
        try:
            if use_f32:
                key = tuple(int(seg.annotations[p]) for p in params["f32"])
                idx = 0
            else:
                key = tuple(int(seg.annotations[p]) for p in params["src"])
                idx = 1
            # Calling tolist() removes neo-added Quantities metadata unit (ms)
            # It does not serialize during data storage
            spikes = seg.spiketrains[idx].times.magnitude.tolist()
            spike_dict[key].append(spikes)
        except KeyError:
            # Skip any segment that is empty or metadata
            continue
        
    return spike_dict


def prettify_spike_dict(spike_dict, dataset=None):
    """
    Turns spike_dict's into more friendly JSON and pandas Dataframe form.
    Returns a list of dicts with keys specific to the dataset type of the 
      spike dict.
      eg. Instead of (freq, int) tuples as dict keys for DenseTC spike_dict, 
      you will have a list of dicts each with separate frequency, intensity, 
      and spiketimes keys.
    
    Dataset types:
      densetc: frequency_hz, intensity_db, spikes_ms
      speech: speech_number, spikes_ms
      burst: ISI_ms, spikes_ms
    """
    known_datasets = {
        "densetc": ["frequency_hz", "intensity_db"],
        "speech": ["speech_number"], 
        "burst": ["ISI_ms"]
        }
    if dataset not in known_datasets:
        raise ValueError("Must specify dataset type to prettify with."
                         f"\nChoose from {list(known_datasets.keys())}")
    keys = known_datasets[dataset]
    pretty_list = []
    for params, value in spike_dict.items():
        prettify = {"spikes_ms": value}
        for idx, key in enumerate(keys):
            prettify[key] = params[idx]
        pretty_list.append(prettify)
        
    return pretty_list


def get_times_from_spike_dict(spike_dict, is_pretty=False):
    """
    Quick function just to contain the semantic ugliness of getting spiketimes.
    Takes a spike_dict made from processing Brainware file/block in 
      get_spike_dict().
    If is_pretty, expects a list of prettified spike_dicts with spikes_ms keys
    
    Returns flattened array of spiketimes (useful for PSTH generation)
    
    No, this doesn't need to be a 3x nested list comprehension, but I had fun
    writing it and it's just a utility function.
    """
    if is_pretty:
        satans_list_comp = [
            time for sweep in
            [sweep for stim_dict in
             [stim_dict["spikes_ms"] for stim_dict in spike_dict]
             for sweep in stim_dict]
             for time in sweep
        ]
    else:
        satans_list_comp = [
            time for sweep in
            [sweep for stimulus in 
            [stimulus for stimulus in spike_dict.values()]
            for sweep in stimulus]
            for time in sweep
        ]
    return satans_list_comp


def alpha_shape(points, alpha):
    """
    Utility for pick_voronoi().
    Compute the alpha shape (concave hull) of a set of points.
    
    Code from:
      http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
    
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. 
      Smaller numbers don't fall inward as much as larger numbers. Too large, 
      and you lose everything!
    """
    if len(points) < 4:
        # If triangle, there is no sense in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semi-perimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Radius filter
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    return cascaded_union(triangles), edge_points


def count_spikes(sweep, onset, offset):
    """
    Returns count of spikes in sweep between onset/offset ranges.
    
    Adds a check that 'sweep' is not None:
      A few recordings have missing sweeps for a single tone, leaving a 
      metaphorical hole in the data. The conditional handles this extremely 
      rare case when mass-applying this function, and treats it as 0 spikes.
    """
    if ((sweep is None) or (np.any(np.isnan(sweep)))):
        return 0
    return sum(np.logical_and((onset <= sweep), (sweep <= offset)))


def get_tuning_curve_dataframe(site_dataframe):
    """
    Generate multi-index dataframe to use for tuning curve array generation.
    Takes a pandas dataframe for a single mapping site. 
      Each row is a single sweep. Should have three named columns:
        'frequency_hz': The frequency played during a sweep
        'intensity_db': The intensity played during a sweep
        'spikes_ms': A list of spiketimes for spikes recording during a sweep
          In refactored code, storing spiketrains is generalized to allow
          multiple sweeps (a list of lists). All the TC analysis funcs
          expect a single sweep. Since this function is essentially the
          entry point for TC analysis in this program, this function
          grabs the 'first' sweep for each 'set' of sweeps in spikes_ms if
          it is generated from refactored code. Otherwise it is just the sweep.
      See prettify_spike_dict().
    """
    tuning_curve_df = site_dataframe.copy()
    tuning_curve_df["spikes_ms"] = tuning_curve_df["spikes_ms"].apply(
        lambda x: np.array(check_sweeps(x)))
    tuning_curve_df = tuning_curve_df.pivot(index="intensity_db", 
                                            columns="frequency_hz")
    return tuning_curve_df


def check_sweeps(sweep):
    """
    Older code stored single TC sweep as a list of spiketimes in dataframe.
    Refactored code generalizes to allow multiple sweeps, but the TC analysis
    functions in program expect single sweep everywhere. So this function bridges
    the incompatibility.
    """
    if len(sweep) == 0:
        return sweep
    elif type(sweep[0])==list:
        return sweep[0]
    else:
        return sweep



def get_tuning_curve_array(tuning_curve_dataframe, onset_ms=0, offset_ms=400):
    """
    Generate 2d array, intensity x frequency, of # of spikes per stim pair.
    Takes a tuning_curve_dataframe generated by get_tuning_curve_dataframe(),
      and a given onset and offset time to filter spiketimes.
    """
    tc = np.array(tuning_curve_dataframe.map(
        lambda x: count_spikes(x, onset_ms, offset_ms))).astype(np.uint8)
    return tc


def get_driven_vs_spont_spike_counts(tuning_curve_dataframe, driven_onset_ms=8, 
                                     driven_offset_ms=38, spont_onset_ms=370,
                                     spont_offset_ms=400):
    """
    Generates two 2d arrays of spikecounts, 'driven' and 'spont'.
    Takes a tuning_curve_dataframe generated by get_tuning_curve_dataframe(),
      and given onset and offset times.
    Spont period length should be the same as driven.
    
    Spont array can be subtracted element-wise from driven array to get an
    estimate of driven vs. spont activity for each TC stimulus sweep.
    """
    if ((driven_offset_ms - driven_onset_ms) != 
         (spont_offset_ms - spont_onset_ms)):
        raise AssertionError("Driven and Spont ranges should be the same.")
    driven_counts = get_tuning_curve_array(tuning_curve_dataframe, 
                                           onset_ms=driven_onset_ms,
                                           offset_ms=driven_offset_ms)
    spont_counts = get_tuning_curve_array(tuning_curve_dataframe,
                                          onset_ms=spont_onset_ms,
                                          offset_ms=spont_offset_ms)

    return driven_counts, spont_counts


def ttest_driven_vs_spont_tc(driven_counts, spont_counts):
    """
    Basic strategy to estimate significantly driven spiking activity in
      response to TC stimuli vs. normal spontaneous activity.
    Takes arrays created by get_driven_vs_spont_spike_counts()
    
    Returns array of same shape containing 'driven' spikecounts, smoothed
      with neighboring 'driven' spikes.
    
    Thresholds 'driven' TC responses based on t-tests p<0.05. 
      If it passes, it's 'driven' (and its neighborhood smoothed). 
      If not, it's set to 0.
    Data is log-transformed from counts to improve normality. Both the 
      log-transformatioon and use of t-tests are technically incorrect for this
      data (log-transform because the counts are usually too low and may fail 
      to meet standard statistical assumptions, t-tests and p-values because 
      they are theoretically unjustified and arbitrary).
    However they are very fast and empirically do a good job for inspection of
      TC and analysis, so they are here for now (alternative methods for 
      smoothing and/or estimating 'driven' vs. 'spont' are typically even 
      worse, both theoretically and in their results).
      
    The test treats a window of tones from the freq/int grid space as if they 
      are repeats of a central tone under consideration.
        eg. 1 kHz 50 dB is very similar to 1 kHz 55 dB and to 1.1 kHz 50 dB so
        we may consider them as practically equivalent.
    The current window implementation is a 3x5 for 15 'repeats' around the 
      center tone (3 intensities, 5 frequencies). 
    Edges of the input TC arrays are repeated as necessary to fill out window, 
      and indices are adjusted accordingly.
    """
    intensities, freqs = driven_counts.shape
    ttest_tc = np.zeros(driven_counts.shape)
    intensities = list(range(intensities))
    freqs = list(range(freqs))

    # Log transform data
    driven_counts = np.log(driven_counts + 1)
    spont_counts = np.log(spont_counts + 1)

    # Repeat edges to satisfy 3x5 window at every freq/int grid point:
    # Pad 1 more int on bottom/top, and 2 more freqs on left/right edges
    driven_counts = np.pad(driven_counts, [(1, 1), (2, 2)], "edge")
    spont_counts = np.pad(spont_counts, [(1, 1), (2, 2)], "edge")

    # Adjust freq/db indices to account for padding
    intensities = np.array(intensities) + 1
    freqs = np.array(freqs) + 2

    for freq, db in itertools.product(freqs, intensities):
        spont_spikes = spont_counts[(db-1):(db+1)+1, 
                                    (freq-2):(freq+2)+1].flatten()
        driven_spikes = driven_counts[(db-1):(db+1)+1, 
                                      (freq-2):(freq+2)+1].flatten()

        # Only accept 'driven' responses ABOVE spontaneous 
        # (not ones significantly below). One-sided test.
        ttest = ttest_ind(driven_spikes, spont_spikes, equal_var=False, 
                          alternative="greater")
        if (ttest.pvalue < 0.05):
            # Convert from log-normal back to spike-count data and store mean 
            # driven response of window for point
            ttest_tc[db-1, freq-2] = np.mean(np.exp(driven_spikes) - 1)

    return ttest_tc


def get_spont(psth, num_sweeps, start_idx=-100, stop_idx=None):
    """
    Estimate spontaneous firing rate for a given PSTH.
    Takes PSTH (expects 1 ms bin size), number of recording sweeps, 
      and indices marking range to use for spontaneous activity estimation.
      Provide a negative start_idx to use end of PSTH.
      Default is last 100 ms of PSTH.
      
    Returns mean and std spontaneous rate in Hz.
    """
    if not stop_idx:
        spont_range = psth[start_idx:]
    else:
        spont_range = psth[start_idx:stop_idx]
    
    if not len(spont_range):
        raise AssertionError("No spontaneous values obtained. " 
                             "Check start/stop_idx arguments work with PSTH.")
    spont = (np.mean(spont_range) * 1000) / num_sweeps
    spont_std = (np.std(spont_range) * 1000) / num_sweeps
    return spont, spont_std


def get_peak_driven_rate(response, spont_hz, n_sweeps):
    """
    Takes an onset:offset bounded PSTH array (expect 1 ms bin size).
    Returns peak driven firing rate in Hz
    
    Driven firing rate only considers activity above spontaneous rate.
      eg. Peak rate of 25 Hz, spontaneous of 8 Hz, peak Driven -> 17 Hz.
    """
    if not response.any():
        return 0
        
    peak_driven_rate = ((max(response) * 1000) / n_sweeps) - spont_hz
    if peak_driven_rate < 0:
        return 0
        
    return peak_driven_rate


def remove_spont(sweep, driven_onset_ms=8, driven_offset_ms=38, 
                 spont_onset_ms=370, spont_offset_ms=400):
    """
    Takes a single sweep and onset/offset times of the driven and spont periods
      in the sweep.
    Spont period length should be the same as driven.
    
    Returns count of (driven - spont) spikes.
    """
    if ((driven_offset_ms - driven_onset_ms) != 
         (spont_offset_ms - spont_onset_ms)):
        raise AssertionError("Driven and Spont ranges should be the same.")

    driven_counts = count_spikes(sweep, driven_onset_ms, driven_offset_ms)
    spont_counts = count_spikes(sweep, spont_onset_ms, spont_offset_ms)

    spikes = driven_counts - spont_counts
    if spikes < 0:
        spikes = 0
        
    return spikes


def analyze_tuning_curve(tc_array):
    """
    TODO Deprecate.
    Older analysis attempt, prefer ttest_analyze_tuning_curve now.
    
    Takes a tc_array from get_tuning_curve_array().
    
    Returns a filtered TC array and analysis items like CF, Threshold, etc.
    
    Blurs data using a median filter, calculates region properties from a 
      thresholded and binarized version, and then selects the biggest blob as
      the probable tuning curve.
    
    Filters and params were determined empirically for "good enough" results.
    """
    med_im = gaussian(tc_array, sigma=1.5)
    try:
        med_thresh = threshold_otsu(med_im)
    except ValueError:
        # Image is empty or only has 1 unique value, and threshold_otsu 
        # doesn't like that
        med_thresh = 0
    med_binary = med_thresh < med_im
    med_label = label(med_binary)
    med_regions = regionprops(med_label)

    big_r = 0
    if med_regions:
        for r_idx, r in enumerate(med_regions):  # Select biggest region
            if med_regions[r_idx].area > med_regions[big_r].area:
                big_r = r_idx
        med_bb = med_regions[big_r].bbox
        med_minr, med_minc, med_maxr, med_maxc = med_bb
        thresh = med_minr
        # Get CF by removing all data from original TC image except what is in 
        # the selected TC blob and then searching for the column index which
        # has the most "spikes"
        tc_im_copy = med_im.copy()
        non_tc_ind = np.where(med_label != [med_regions[big_r].label])
        tc_im_copy[non_tc_ind] = 0
        cf = int(np.argmax(tc_im_copy[med_minr, :]))
        # Get final_file-style BW's; assign None to any of 10-40 if they exceed 
        # range of intensity.
        # Using tolist() to make data JSON-serializable
        # Using convex_image instead of TC to smooth jagged edges of blur and 
        # produce more consistent contours
        cvx_im = tc_im_copy.copy()
        cvx_im[med_minr:med_maxr, med_minc:med_maxc] = \
            med_regions[big_r].convex_image
        try:
            response_idx = np.where(tc_im_copy[med_minr + 2, :])
            bw10 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw10 = [None, None]
        try:
            response_idx = np.where(tc_im_copy[med_minr + 4, :])
            bw20 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw20 = [None, None]
        try:
            response_idx = np.where(tc_im_copy[med_minr + 6, :])
            bw30 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw30 = [None, None]
        try:
            response_idx = np.where(tc_im_copy[med_minr + 8, :])
            bw40 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw40 = [None, None]

        # Get continuous BWs
        cont_bw = []
        bw_level = med_minr
        # Since not prescribing existence of bw10->bw40, no need for try/except
        # clause.
        # maxr and maxc map to the values just beyond real values. Therefore, 
        # indexing at maxr exceeds the index of the image. 
        #   Using maxr-1 to avoid index error
        while bw_level < med_maxr-1:
            # Start 1 level higher than thresh. Should end on maxr
            response_idx = np.where(tc_im_copy[bw_level+1, :])
            cont_bw.append(np.array([response_idx[0][0], 
                                     response_idx[0][-1]]).tolist())
            bw_level = bw_level + 1

    else:
        # If latency found, but no region is found (rare/impossible, yes?), 
        # don't assign cf, thresh, or bw's
        thresh = cf = None
        bw10 = bw20 = bw30 = bw40 = [None, None]
        cont_bw = [None]
        tc_im_copy = med_im  # Just return image (should be empty).
        cvx_im = med_im

    filtered_tc = med_im.copy()
    filtered_tc[~med_binary] = 0
    result = (tc_im_copy, filtered_tc, cf, thresh, bw10, bw20, bw30, bw40, 
              cont_bw, cvx_im)
    
    return result


def ttest_analyze_tuning_curve(tc_array):
    """
    TODO Rewrite. Uses the same vars/logic as older analyze_tuning_curve(), but
      that was just for compatibility at the time. This can be improved.
      
    Blob analysis on a ttest TC generated from ttest_driven_vs_spont_tc().
    
    Returns the TC back and analysis items like CF, Threshold, etc.
    """
    med_binary = 0 < tc_array
    med_label = label(med_binary)
    med_regions = regionprops(med_label)

    big_r = 0
    if med_regions:
        for r_idx, r in enumerate(med_regions):  # Select biggest region
            if med_regions[r_idx].area > med_regions[big_r].area:
                big_r = r_idx
        med_bb = med_regions[big_r].bbox
        med_minr, med_minc, med_maxr, med_maxc = med_bb
        thresh = med_minr
        # Get CF by removing all data from original TC image except what is in 
        # the selected TC blob and then searching for the column index which 
        # has the most "spikes"
        tc_im_copy = tc_array.copy()
        non_tc_ind = np.where(med_label != [med_regions[big_r].label])
        tc_im_copy[non_tc_ind] = 0
        cf = int(np.argmax(tc_im_copy[med_minr, :]))
        # Get final_file-style BW's; assign None to any of 10-40 if they exceed
        # range of intensity.
        # Using tolist() to make data JSON-serializable
        # Using convex_image instead of TC to smooth jagged edges of blur and 
        # produce more consistent contours
        cvx_im = tc_im_copy.copy()
        cvx_im[med_minr:med_maxr, med_minc:med_maxc] = \
            med_regions[big_r].convex_image
        try:
            response_idx = np.where(tc_im_copy[med_minr + 2, :])
            bw10 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw10 = [None, None]
        try:
            response_idx = np.where(tc_im_copy[med_minr + 4, :])
            bw20 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw20 = [None, None]
        try:
            response_idx = np.where(tc_im_copy[med_minr + 6, :])
            bw30 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw30 = [None, None]
        try:
            response_idx = np.where(tc_im_copy[med_minr + 8, :])
            bw40 = np.array([response_idx[0][0], response_idx[0][-1]]).tolist()
        except IndexError:
            bw40 = [None, None]

        # Get continuous BWs
        cont_bw = []
        bw_level = med_minr
        # Since not prescribing existence of bw10->bw40, no need for try/except
        # clause.
        # maxr and maxc map to the values just beyond real values. Therefore, 
        # indexing at maxr exceeds the index of the image.
        #   Using maxr-1 to avoid index error
        while bw_level < med_maxr-1:
            # Start 1 level higher than thresh. Should end on maxr
            response_idx = np.where(tc_im_copy[bw_level+1, :])
            cont_bw.append(np.array([response_idx[0][0], 
                                     response_idx[0][-1]]).tolist())
            bw_level = bw_level + 1

    else:
        # If latency found, but no region is found (rare/impossible, yes?), 
        # don't assign cf, thresh, or bw's
        thresh = cf = None
        bw10 = bw20 = bw30 = bw40 = [None, None]
        cont_bw = [None]
        tc_im_copy = tc_array  # Just return image (should be empty).
        cvx_im = tc_array

    filtered_tc = tc_array.copy()
    filtered_tc[~med_binary] = 0
    result = (tc_im_copy, filtered_tc, cf, thresh, bw10, bw20, bw30, bw40, 
              cont_bw, cvx_im)
    
    return result


def create_final_file(ic_bool=False):
    """
    Load an analysis for a subject and export to v-plot style 'final file'.
    Writes a .csv file. Import into MATLAB to save as .mat file.
    
    IC final files have a reduced export.
    """
    print(colorama.Style.BRIGHT+colorama.Fore.YELLOW+
          "Select subject database file: ")
    db_path = get_file(title="Select database JSON file", 
                       filetypes=[("JSON", ".json")])
    if (db_path is None) or (db_path == ""):
        return

    # Initialize length of final-file matrix. Currently 88 from MATLAB vplot
    # style, but arbitrary
    array_length = 88

    # Initialize tinymongo database
    mongo_connection = TinyMongoClient(os.path.dirname(db_path))
    subject_database = getattr(mongo_connection, 
                               os.path.splitext(os.path.basename(db_path))[0])
    if ic_bool:
        densetc_analysis_collection = subject_database.densetc_IC_analysis
    else:
        densetc_analysis_collection = subject_database.densetc_analysis
    analysis_metadata_collection = subject_database.analysis_metadata

    # Load analysis. 
    # TODO Prevent creation of new analysis here? Currently it is allowed.
    analysis_loaded = False
    while not analysis_loaded:
        analysis_selection, create_new_analysis = \
            load_analysis(analysis_metadata_collection)
        if analysis_selection is None:  # Menu exited without selection
            return
        elif create_new_analysis:
            new_analysis_metadata = new_analysis_metadata_document()
            if new_analysis_metadata is None:  # User hit cancel.
                continue
            template_id = analysis_selection["_id"]
            analysis_id = create_new_densetc_analysis(
                template_id,
                new_analysis_metadata,
                analysis_metadata_collection,
                densetc_analysis_collection)
            analysis_loaded = True
        else:
            analysis_id = analysis_selection["_id"]
            analysis_loaded = True

    densetc_analysis = {analysis["number"]: analysis for analysis in
                        densetc_analysis_collection.find(
                            {"analysis_id": analysis_id})}
    analysis_df = pd.DataFrame(densetc_analysis)
    if ic_bool:
        analysis_df = analysis_df.transpose().reset_index()
        sites_df = analysis_df.copy()
        final_file = np.zeros([len(analysis_df), array_length])
    else:
        sites_collection = subject_database.sites
        sites = [site for site in sites_collection.find({})]
        sites_df = pd.DataFrame(sites)
        map_areas = {num: geometry.polygon.Polygon(poly).area for num, poly in
                     sites_df[["number", "voronoi_vertices"]].values}
        final_file = np.zeros([len(sites), array_length])
        
    for idx, row in sites_df.iterrows():
        site_number = row["number"]
        if ic_bool:
            analysis_entry = row.copy() 
        else:
            analysis_entry = analysis_df[site_number]
        
        field = analysis_entry["field_assignment"]
        if (field == "A1") or (field == ""):
            field = 0
        elif field == "AAF":
            field = 1
        elif field == "PAF":
            field = 2
        elif field == "Other":
            field = 3  # v-plot treats as DAF
        elif field == "VAF":
            field = 4
        elif field == "NAR":
            field = 5
        elif field == "SRAF":
            field = 6  # Not coded in v-plot yet, but might be later
        
        if ic_bool:
            electrode = site_number % 2
            if electrode == 0:
                electrode = 2
            penetration = row["penetration_number"]
            file_number = f"{penetration}0{electrode}"
        else:
            electrode = site_number % 4
            if electrode == 0:
                electrode = 4
                penetration = int(site_number / 4)
            else:
                penetration = int(site_number / 4) + 1
            file_number = f"{penetration}0{electrode}"
        
        spont = analysis_entry["spont_firing_rate_hz"]

        if analysis_entry["cf_idx"] is None:
            cf = 0
            thresh = 0
            a10 = b10 = bw10 = 0
            a20 = b20 = bw20 = 0
            a30 = b30 = bw30 = 0
            a40 = b40 = bw40 = 0
            onset = offset = peak = 0
            peak_driven_rate = 0
        else:
            cf = analysis_entry["cf_khz"]
            thresh = analysis_entry["threshold_db"]
            if analysis_entry["bw10_idx"][0] is not None:
                bw10_khz = analysis_entry["bw10_khz"]
                bw10_octave = analysis_entry["bw10_octave"]
                a10, b10 = bw10_khz
                bw10 = bw10_octave
            else:
                a10 = b10 = bw10 = 0
            if analysis_entry["bw20_idx"][0] is not None:
                bw20_khz = analysis_entry["bw20_khz"]
                bw20_octave = analysis_entry["bw20_octave"]
                a20, b20 = bw20_khz
                bw20 = bw20_octave
            else:
                a20 = b20 = bw20 = 0
            if analysis_entry["bw30_idx"][0] is not None:
                bw30_khz = analysis_entry["bw30_khz"]
                bw30_octave = analysis_entry["bw30_octave"]
                a30, b30 = bw30_khz
                bw30 = bw30_octave
            else:
                a30 = b30 = bw30 = 0
            if analysis_entry["bw40_idx"][0] is not None:
                bw40_khz = analysis_entry["bw40_khz"]
                bw40_octave = analysis_entry["bw40_octave"]
                a40, b40 = bw40_khz
                bw40 = bw40_octave
            else:
                a40 = b40 = bw40 = 0

            onset = analysis_entry["onset_ms"]
            peak = analysis_entry["peak_ms"]
            offset = analysis_entry["offset_ms"]

            peak_driven_rate = analysis_entry["peak_driven_rate_hz"]
            
        final_file[idx, 0] = file_number
        final_file[idx, 1] = cf
        final_file[idx, 2] = thresh
        final_file[idx, 6] = a10
        final_file[idx, 7] = b10
        final_file[idx, 8] = bw10
        final_file[idx, 11] = a20
        final_file[idx, 12] = b20
        final_file[idx, 13] = bw20
        final_file[idx, 16] = a30
        final_file[idx, 17] = b30
        final_file[idx, 18] = bw30
        final_file[idx, 21] = a40
        final_file[idx, 22] = b40
        final_file[idx, 23] = bw40
        final_file[idx, 25] = onset
        final_file[idx, 26] = peak_driven_rate
        final_file[idx, 33] = peak
        final_file[idx, 34] = offset
        final_file[idx, 37] = spont
        final_file[idx, 42] = field
        final_file[idx, 43] = site_number

        if not ic_bool:
            site_x = row["x"]
            site_y = row["y"]
            site_area = map_areas[site_number]
            site_poly = row["voronoi_vertices"]
            final_file[idx, 40] = site_x
            final_file[idx, 41] = site_y
            final_file[idx, 48] = site_area
            x_coor_start_idx = 49
            y_coor_start_idx = 50
            for poly_idx, point in enumerate(site_poly):
                x_idx = x_coor_start_idx + (poly_idx * 2)
                y_idx = y_coor_start_idx + (poly_idx * 2)
                final_file[idx, x_idx] = point[0]
                final_file[idx, y_idx] = point[1]

    print(colorama.Style.BRIGHT+colorama.Fore.YELLOW+
          "Select a location and file name to save your 'final file' to: ")
    save_location = save_file(title="Save final file", 
                              filetypes=[("XLS", ".xls")])
    print(save_location+".xls")
    # Write final file out to .csv file. 
    # MATLAB will be used to convert to .mat file
    excel_writer = pd.ExcelWriter(save_location+".xls")
    df = pd.DataFrame(final_file)
    df.to_excel(excel_writer, index=False, header=False)
    excel_writer.close()
