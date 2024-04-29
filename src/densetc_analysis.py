import pytesseract
import cv2
import numpy as np
import os
from neo.io.brainwaresrcio import BrainwareSrcIO
from neo.io.brainwaref32io import BrainwareF32IO
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from tinymongo_fix.tinymongo_fix import TinyMongoClient
import datetime
from colorama import Fore, Style
import bayesian_bins as bb
import analysis_functions as afunc
from functools import partial

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray


def run_program(config_dict, version, final_file=None, return_sdf=True):
    analysis_version = f"map_auto_analysis v{version}"
    today = str(datetime.datetime.now())
    subject_name = input("What is the subject's name? > ").strip()
    
    print(Style.BRIGHT+Fore.YELLOW+"Select folder to save to: "+
          Style.RESET_ALL)
    save_dir_path = afunc.get_folder(title="Folder to save to")

    use_f32 = False
    while (file_type := input("Using .[s]rc or .[f]32 file type? > ")
           .lower().strip()) not in ["f", "s"]:
        continue
    if file_type == "f":
        use_f32 = True

    # If config allows IC analysis, prompt user whether rat has IC data
    ic_bool = False
    ic_only = False
    use_images = False
    image_or_point_list = None
    ic_points_df = pd.DataFrame([{'number': None}])
    if config_dict["do_IC"]:
        while (response := input("Does this subject have IC data [y/n]? > ")
               .lower().strip()) not in ["y", "n"]:
            continue
        if response == "y":
            ic_bool = True
            print(Style.BRIGHT+Fore.YELLOW+"Select .csv file listing IC map "
                  "Penetration numbers with their corresponding depths "
                  "(Number, then Depth column, no headers): "+Style.RESET_ALL)
            ic_csv = afunc.get_file(title="IC Num,Depth .csv", 
                                    filetypes=[('CSV', '.csv')])
            ic_points_df = pd.read_csv(ic_csv, header=None, 
                                       names=["number", "depth"])
            # IC must be sorted by penetration: used for filename -> map#
            ic_points_df = ic_points_df.sort_values("number")
            ic_points_df.reset_index(inplace=True, drop=True)
            
            while (response := input("Is this an IC only map [y/n]? > ")
                   .lower().strip()) not in ["y", "n"]:
                continue
            if response == "y":
                ic_only = True

    image_or_point_list = ""
    if (not ic_only) and (final_file is None):
        while (image_or_point_list := input("Using [i]mages, [f]inal file, or "
                                            ".[c]sv for map point data? > ")
               .lower().strip()) not in ["i", "f", "c"]:
            continue
        if image_or_point_list == "i":
            use_images = True

    # Database is saved as subject name.
    # This will combine data/analysis if a database already exists.
    # That is probably not what you want. Recommend moving or deleting any
    # previous database files for the subject.
    connection = TinyMongoClient(save_dir_path + subject_name)

    # Grab database that was just created.
    # getattr result is same as eg. connection.RAT1 if subject_name == RAT1
    # Also, connection.RAT1.1 (with a decimal) is normally invalid syntax, 
    # but using getattr() and passing "RAT1.1" works as expected.
    db = getattr(connection, subject_name)

    # Generate db collections
    db_metadata = db.metadata
    db_sites = db.sites
    db_analysis_metadata = db.analysis_metadata
    
    if config_dict["do_densetc"]:
        db_densetc_data = db.densetc_data
        db_densetc_analysis = db.densetc_analysis
        if ic_bool:
            db_densetc_IC_data = db.densetc_IC_data
            db_densetc_IC_analysis = db.densetc_IC_analysis
    if config_dict["do_speech"]:
        db_speech_data = db.speech_data
        if ic_bool:
            db_speech_IC_data = db.speech_IC_data
    if config_dict["do_burst"]:
        db_noiseburst_data = db.noiseburst_data
        if ic_bool:
            db_noiseburst_IC_data = db.noiseburst_IC_data

    meta_id = db_metadata.insert_one({"program_version": analysis_version, 
                                      "program_run_date": today}).inserted_id
    # Automatic-analysis metadata
    # Prevent anyone manually changing this by including a 'frozen' entry
    if final_file:
        analysis_comment = "Tuning curve analysis generated from a final file"
    else:
        analysis_comment = "Auto tuning curve analysis and data pre-processing"
    analysis_id = db_analysis_metadata.insert_one({
        "name": analysis_version,
        "start_date": today,
        "last_modified": today,
        "configuration": config_dict,
        "frozen": True,
        "comments": analysis_comment,
    }).inserted_id

    map_width = 1
    map_height = 1
    map_points_df = pd.DataFrame([{'number': None}])
    if use_images:
        """
        Extract numbers/points from mapping images using tesseract
        See https://tesseract-ocr.github.io/tessdoc/Home.html 
          and https://github.com/UB-Mannheim/tesseract/wiki
        Uses custom OCR file, mapOCR. Don't lose this file! 
          Can always re-train, but pain in the ass
        File goes in /tessdata/ dir of Tesseract install location 
          (hardcoded for default install location)
        """
        ocr_lang = "mapOCR"
        tess_loc = "C:/Program Files/Tesseract-OCR"
        pytesseract.pytesseract.tesseract_cmd = f"{tess_loc}/tesseract"
        tessdata_config = f"--tessdata-dir '{tess_loc}/tessdata'"

        print(Style.BRIGHT+Fore.YELLOW+"Select map POINTS image:")
        points_im_filename = afunc.get_file(title="Select Map Points image", 
                                            filetypes=[("PNG", ".png")])
        points_image = cv2.imread(points_im_filename, cv2.IMREAD_GRAYSCALE)
        points_binary = points_image < 128
        points_label = label(points_binary)
        points_regions = regionprops(points_label)

        print(Style.BRIGHT+Fore.YELLOW+"Select map NUMBERS image:")
        numbers_im_filename = afunc.get_file(title="Select Map Numbers image", 
                                             filetypes=[("PNG", ".png")])
        numbers_image = cv2.imread(numbers_im_filename, cv2.IMREAD_GRAYSCALE)
        norm_row_max, norm_col_max = numbers_image.shape

        print(Style.BRIGHT+Fore.YELLOW+"Select map MASK image:")
        mask_im_filename = afunc.get_file(title="Select Map Mask image", 
                                          filetypes=[("PNG", ".png")])
        mask_image = cv2.imread(mask_im_filename, cv2.IMREAD_GRAYSCALE)
        mask_binary = mask_image < 128
        mask_label = label(mask_binary)
        mask_regions = regionprops(mask_label)
        
        print(Style.RESET_ALL)

        if len(mask_regions) != len(points_regions):
            raise AssertionError(Style.BRIGHT+Fore.RED+
                                 "Unequal number of Points and Numbers.\n" 
                                 "Were these files made correctly?"+
                                 Style.RESET_ALL)

        map_height, map_width = points_image.shape
        points_list = []
        points_centroids = np.array([c for c in 
                                     [r.centroid for r in points_regions]])
        # TODO Check if all available points have been matched
        # For each mask, OCR map # and match to point shortest distance away
        for mask_props in mask_regions:
            bbox = mask_props.bbox
            number = numbers_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            ocr_num = pytesseract.image_to_string(number, 
                                                  config=tessdata_config, 
                                                  lang=ocr_lang)
            # In apparently random rare cases, OCR fails to find a number.
            # Prompt user in this case:
            if ocr_num == "":
                print(Style.BRIGHT+Fore.MAGENTA+
                      "OCR failure! What is this number? "+Style.RESET_ALL)
                plt.imshow(number)
                plt.pause(0.1)
                ocr_num = input("Answer: > ")

            distances = np.linalg.norm(
                points_centroids - np.array(mask_props.centroid), axis=1)
            point = points_centroids[np.argmin(distances)]
            x = point[1] / norm_col_max
            y = point[0] / norm_row_max

            points_list.append({"number": int(ocr_num), "x": x, "y": y})

        map_points_df = pd.DataFrame(points_list)
        # y-axis coordinates are image-indexed (flipped)
        map_points_df["y"] = map_points_df["y"].apply(lambda x: 1 - x)

    elif not ic_only:
        if image_or_point_list == "f":
            print(Style.BRIGHT+Fore.YELLOW+
                  "Select spreadsheet containing 'final file' format data:"+
                  Style.RESET_ALL)
            coords_sheet = afunc.get_file(title="Select final file",
                                          filetypes=[("XLS", ".xls")])
            map_points_df = pd.read_excel(coords_sheet, 
                                          header=None, 
                                          usecols=[40, 41, 43], 
                                          names=["x", "y", "number"])
        elif image_or_point_list == "c":
            print(Style.BRIGHT+Fore.YELLOW+"Select .csv file containing "
                  "Number, x, y columns IN THAT ORDER (no headers):"+
                  Style.RESET_ALL)
            coords_sheet = afunc.get_file(title="Select Map Num,x,y file", 
                                          filetypes=[("CSV", ".csv")])
            map_points_df = pd.read_csv(coords_sheet, 
                                        header=None, 
                                        names=["number", "x", "y"])
            
        # Pseudo width/height to store something similar to normal images
        map_width = map_points_df.x.max() * 1000
        map_width = int(map_width + (map_width * 0.1))
        map_height = map_points_df.y.max() * 1000
        map_height = int(map_height + (map_height * 0.1))
        
    # Update database with map width/height data.
    db_metadata.update_one({"_id": meta_id}, {"map_height": map_height, 
                                              "map_width": map_width,})

    # Add breathing room to point coordinates so map is easier to display
    # min=0, max=max+(max*0.1), then scale the data in range of 0.1-0.9
    if not ic_only:
        max_coor = map_points_df[["x", "y"]].max().values.max()
        min_coor = 0
        max_scale = 0.9
        min_scale = 0.1
        map_points_df[["x", "y"]] = map_points_df[["x", "y"]].apply(
            lambda x: afunc.scale_coordinates(input_coor=x, 
                                              min_coor=min_coor, 
                                              max_coor=max_coor, 
                                              min_scale=min_scale, 
                                              max_scale=max_scale))
        
        print("Working on voronoi data...")
        sites_list, bonus_pts = afunc.pick_voronoi(map_points_df,
                                                   map_width, map_height)

        # Verify voronoi by plotting patches, then save!
        #print(Fore.GREEN+"Displaying voronoi data (close whenever)..."+
        #      Style.RESET_ALL)
        #afunc.check_voronoi(sites_list, bonus_pts)
        print(Fore.GREEN+"\nSaving map sites / voronoi data ... \n\n"+ 
              Style.RESET_ALL)
        db_sites.insert_many(sites_list)

    """
    Analysis! Run through options of DenseTC, speech, etc.
    Open all .src/.f32 files for subject and store handles in list.
    Skips any files missing a corresponding point in points_list.
    If using IC, it checks against penetration number instead of map number.
    """
    print(Style.BRIGHT+Fore.YELLOW+
          f"Select dir containing all Brainware files for subject "
          "(subfolders will be skipped):"+
          Style.RESET_ALL)
    dir_path = afunc.get_folder(title=f"Select Brainware data dir")
    
    nums = map_points_df.number.values
    ic_pens = ic_points_df.number.values if ic_bool else []
    if use_f32:
        bw_func = BrainwareF32IO
        ext = ".f32"
    else:
        bw_func = BrainwareSrcIO
        ext = ".src"
        
    # Aggregate all Brainware files ahead of time
    potential_datasets = [("do_densetc", "densetc_file"),
                          ("do_speech", "speech_file"),
                          ("do_burst", "burst_file")]
    patterns = {p[0]: config_dict[p[1]] for 
                p in potential_datasets if config_dict[p[0]]}
    bw_files = {}
    for dataset, pattern in patterns.items():
        data_dir = os.scandir(dir_path)
        check_file = lambda n: (n.endswith(ext) and n.startswith(pattern) and
                                (afunc.get_map_number(n) in nums or 
                                 afunc.get_penetration_number(n) in ic_pens))
        bw_files[dataset] = [bw_func(filename=dir_path+entry.name) for 
                             entry in data_dir if check_file(entry.name)]
        print(Fore.GREEN+
              f"{len(bw_files[dataset])} {ext} {pattern} files found "
              "and matched to mapping sites."+Style.RESET_ALL)

    # Tuning curves
    if config_dict["do_densetc"]:
        freqs = np.array(config_dict["densetc_frequency_hz"])
        ints = np.array(config_dict["densetc_intensity_db"])
        
        # Parallelize!
        @ray.remote
        def analyze(idx, file):
            return densetc_bw_loop(idx=idx, file=file,
                                   total=len(bw_files["do_densetc"]),
                                   use_f32=use_f32,
                                   n_sweeps=config_dict["densetc_num_tones"],
                                   freqs=freqs,
                                   ints=ints,
                                   ic_pens=ic_pens,
                                   final_file=final_file,
                                   return_sdf=return_sdf)
        # TODO Fix/handle random timeout error for failing to start ray process?
        results = ray.get([analyze.remote(idx, file) for 
                           idx, file in enumerate(bw_files["do_densetc"])])
        #results = [analyze(idx, file) for 
        #           idx, file in enumerate(bw_files["do_densetc"])]

        # Store data. Keep raw data separate from all analysis on data. 
        # Map # used as id (IC adds penetration number and depth as well)
        densetc_data_list = []
        densetc_analysis_list = []
        densetc_IC_data_list = []
        densetc_IC_analysis_list = []
        for r in results:
            r["analysis_dict"]["analysis_id"] = analysis_id
            if ic_bool and (r["penetration_number"] in ic_pens):
                row = ic_points_df.number == r["penetration_number"]
                depth = ic_points_df[row]["depth"].values.tolist()[0]
                r["data_dict"]["depth"] = depth
                densetc_IC_data_list.append(r["data_dict"])
                densetc_IC_analysis_list.append(r["analysis_dict"])
            else:
                densetc_data_list.append(r["data_dict"])
                densetc_analysis_list.append(r["analysis_dict"])

        # End of DenseTC analysis section
        print(Fore.GREEN+"\nSaving densetc data/analysis..."+Style.RESET_ALL)
        db_densetc_data.insert_many(densetc_data_list)
        db_densetc_analysis.insert_many(densetc_analysis_list)
        if ic_bool:
            db_densetc_IC_data.insert_many(densetc_IC_data_list)
            db_densetc_IC_analysis.insert_many(densetc_IC_analysis_list)
    
    # Speech
    # TODO Save speech name with pretty spiketrain dict
    if config_dict["do_speech"]:
        # Parallelize!
        @ray.remote
        def analyze(idx, file):
            return speech_bw_loop(idx=idx, file=file,
                                  total=len(bw_files["do_speech"]),
                                  use_f32=use_f32,
                                  ic_pens=ic_pens)

        results = ray.get([analyze.remote(idx, file) for
                           idx, file in enumerate(bw_files["do_speech"])])
        
        # Save speech data
        speech_data_list = []
        speech_IC_data_list = []
        for r in results:
            if ic_bool and (r["penetration_number"] in ic_pens):
                row = ic_points_df.number == r["penetration_number"]
                r["depth"] = ic_points_df[row]["depth"].values.tolist()[0]
                speech_IC_data_list.append(r)
            else:
                speech_data_list.append(r)
        print(Fore.GREEN+"\nSaving speech data..."+Style.RESET_ALL)
        db_speech_data.insert_many(speech_data_list)
        if ic_bool:
            db_speech_IC_data.insert_many(speech_IC_data_list)
    
    # Noisebursts
    # TODO Save number of bursts with pretty spiketrain dict
    if config_dict["do_burst"]:
        # Parallelize
        @ray.remote
        def analyze(idx, file):
            return burst_bw_loop(idx=idx, file=file,
                                 total=len(bw_files["do_burst"]),
                                 use_f32=use_f32,
                                 ic_pens=ic_pens)
            
        results = ray.get([analyze.remote(idx, file) for
                           idx, file in enumerate(bw_files["do_burst"])])
        
        # Save noiseburst data
        burst_data_list = []
        burst_IC_data_list = []
        for r in results:
            if ic_bool and (r["penetration_number"] in ic_pens):
                row = ic_points_df.number == r["penetration_number"]
                r["depth"] = ic_points_df[row]["depth"].values.tolist()[0]
                burst_IC_data_list.append(r)
            else:
                burst_data_list.append(r)
        print(Fore.GREEN+"\nSaving noiseburst data..."+Style.RESET_ALL)
        db_noiseburst_data.insert_many(burst_data_list)
        if ic_bool:
            db_noiseburst_IC_data.insert_many(burst_IC_data_list)


def densetc_bw_loop(idx, file, total, use_f32, n_sweeps, freqs, ints, 
                    ic_pens=[], final_file=None, return_sdf=True):
    """
    Parse DenseTC files and attempt auto-analysis of TCs and latencies.
    Returns dict containing:
      data_dict: Raw data from DenseTC file.
        number: map # as id
        penetration_number: ibid
        filename: for posteriority and prosperity
        spiketrains: A list of dicts
          {frequency_hz: Tone Hz, intensity_db: Tone dB, spikes_ms: spiketimes}
          Use it to create a pandas dataframe for easier analysis
      analysis_dict: The auto-analyzed latency and TC properties
        Too many items to list out here. See end of function ya loon.
      penetration_number: Used for IC depth parsing, if needed.
    """
    get_spikes = partial(afunc.get_spike_dict, use_f32=use_f32, 
                         dataset="densetc")
    prettify_func = partial(afunc.prettify_spike_dict, dataset="densetc")
    bw_dict = read_bw_block(file, use_f32, get_spikes, ic_pens, 
                            prettify_func=prettify_func)
    print(f"Working on {idx+1} of {total} DenseTC files\n"
          f"\tMap number is: {bw_dict['number']}")
    
    spike_dict = bw_dict["spiketrains"]
    all_spikes = afunc.get_times_from_spike_dict(spike_dict, is_pretty=True)
    # TODO fix hard coding of sweep len
    psth = np.histogram(all_spikes, bins=range(400))[0]
    spont, spont_std = afunc.get_spont(psth, n_sweeps)
    
    # Default latency_dict. No alternative latency strategy exists for raw
    # data, but if using a final_file and don't want SDF, these default values
    # make the rest of the program function.
    latency_dict = {
        "onset": 50,
        "offset": 300,
        "peak": None,
        "lats": None, 
        "signal": None,
        "max_prob": None,
        "total_prob": None, 
        "sdf": None,
        "m_priors": None,
        "sigma": None,
        "gamma": None
        }
    if return_sdf:
        latency_dict = get_densetc_bb_lats(psth, n_sweeps, spont)
    elif final_file:
        map_number = bw_dict["map_number"]
        row = final_file[final_file["number"] == map_number]
        onset = int(row["onset"].values)
        if not onset:  # Marked as non-responding site in final file
            onset, peak, offset = 50, None, 300
        else:
            offset = int(row["offset"].values)
            if not offset:  # TC Explorer couldn't determine offset after onset
                offset = 399  # End of sweep TODO eliminate hardcoding
            peak = int(np.argmax(psth[onset:offset])) + onset
        latency_dict["onset"] = onset
        latency_dict["peak"] = peak
        latency_dict["offset"] = offset
    
    onset = latency_dict["onset"]
    offset = latency_dict["offset"]
    peak_driven_rate = afunc.get_peak_driven_rate(psth[onset:offset], spont, 
                                                  n_sweeps)
    
    if latency_dict["peak"] is None:  # Non-responsive site, no analysis needed
        peak_driven_rate = 0
        cf = cf_khz = thresh = thresh_db = None
        bw10 = bw20 = bw30 = bw40 = [None, None]
        continuous_bw = [None, None]
        bw10_octave = bw20_octave = bw30_octave = bw40_octave = None
        continuous_bw_octave = None
        bw10_khz = bw20_khz = bw30_khz = bw40_khz = [None, None]
        continuous_bw_khz = [None, None]
    elif final_file:
        # Don't measure continuous for analysis pulled from final file
        continuous_bw = [None, None]
        continuous_bw_khz = [None, None]
        continuous_bw_octave = None
        row = final_file[final_file["number"] == map_number]
        if row["cf"].values == 0:  # Non-responsive / unanalyzed site
            cf = thresh = cf_khz = thresh_db = None
        else:
            # Snap variable final file analysis values to true values
            # eg. Threshold of 10.7 dB snaps to 10 dB
            cf_khz = afunc.snap(freqs/1000, row["cf"].values)
            thresh_db = int(afunc.snap(ints, row["thresh"].values))
            cf = int(np.where(int(cf_khz*1000) == freqs.astype(int))[0][0])
            thresh = int(np.where(thresh_db == ints)[0][0])
            
        if row["bw10a"].values == 0:
            bw10 = [None, None]
            bw10_khz = [None, None]
            bw10_octave = None
        else:
            bw10a = afunc.snap(freqs/1000, row["bw10a"].values)
            bw10b = afunc.snap(freqs/1000, row["bw10b"].values)
            bw10 = [int(np.where(int(bw10a*1000) == freqs.astype(int))[0][0]),
                    int(np.where(int(bw10b*1000) == freqs.astype(int))[0][0])]
            bw10_khz = [bw10a, bw10b]
            bw10_octave = row["bw10"].values[0]

        if row["bw20a"].values == 0:
            bw20 = [None, None]
            bw20_khz = [None, None]
            bw20_octave = None
        else:
            bw20a = afunc.snap(freqs/1000, row["bw20a"].values)
            bw20b = afunc.snap(freqs/1000, row["bw20b"].values)
            bw20 = [int(np.where(int(bw20a*1000) == freqs.astype(int))[0][0]),
                    int(np.where(int(bw20b*1000) == freqs.astype(int))[0][0])]
            bw20_khz = [bw20a, bw20b]
            bw20_octave = row["bw20"].values[0]

        if row["bw30a"].values == 0:
            bw30 = [None, None]
            bw30_khz = [None, None]
            bw30_octave = None
        else:
            bw30a = afunc.snap(freqs/1000, row["bw30a"].values)
            bw30b = afunc.snap(freqs/1000, row["bw30b"].values)
            bw30 = [int(np.where(int(bw30a*1000) == freqs.astype(int))[0][0]),
                    int(np.where(int(bw30b*1000) == freqs.astype(int))[0][0])]
            bw30_khz = [bw30a, bw30b]
            bw30_octave = row["bw30"].values[0]

        if row["bw40a"].values == 0:
            bw40 = [None, None]
            bw40_khz = [None, None]
            bw40_octave = None
        else:
            bw40a = afunc.snap(freqs/1000, row["bw40a"].values)
            bw40b = afunc.snap(freqs/1000, row["bw40b"].values)
            bw40 = [int(np.where(int(bw40a*1000) == freqs.astype(int))[0][0]),
                    int(np.where(int(bw40b*1000) == freqs.astype(int))[0][0])]
            bw40_khz = [bw40a, bw40b]
            bw40_octave = row["bw40"].values[0]
        
    else:  # Normal analysis on responsive site
        try:  # Cont BW crashes program if TC doesn't return "good enough" data
            tc_df = afunc.get_tuning_curve_dataframe(
                pd.DataFrame(spike_dict))
            # TODO Fix hardcoding sweep len
            ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
                tc_df, 
                driven_onset_ms=onset, 
                driven_offset_ms=offset,
                spont_onset_ms=400 - (offset - onset),
                spont_offset_ms=400)
            # TODO check this
            ttest_tc = afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts)
            # TODO gross, clean up
            _, _, cf, thresh, bw10, bw20, bw30, bw40, continuous_bw, _ = afunc.ttest_analyze_tuning_curve(ttest_tc)

            if bw10[0] is not None:
                bw10_khz = (freqs[bw10] / 1000).tolist()
                bw10_octave = afunc.get_bandwidth(*freqs[bw10]).tolist()
            else:
                bw10_khz = [None, None]
                bw10_octave = None
            if bw20[0] is not None:
                bw20_khz = (freqs[bw20] / 1000).tolist()
                bw20_octave = afunc.get_bandwidth(*freqs[bw20]).tolist()
            else:
                bw20_khz = [None, None]
                bw20_octave = None
            if bw30[0] is not None:
                bw30_khz = (freqs[bw30] / 1000).tolist()
                bw30_octave = afunc.get_bandwidth(*freqs[bw30]).tolist()
            else:
                bw30_khz = [None, None]
                bw30_octave = None
            if bw40[0] is not None:
                bw40_khz = (freqs[bw40] / 1000).tolist()
                bw40_octave = afunc.get_bandwidth(*freqs[bw40]).tolist()
            else:
                bw40_khz = [None, None]
                bw40_octave = None

            continuous_bw_khz = [(freqs[bw] / 1000).tolist() for 
                                 bw in continuous_bw]
            continuous_bw_octave = [afunc.get_bandwidth(*freqs[bw]).tolist() 
                                    for bw in continuous_bw]
            cf_khz = freqs[cf] / 1000
            thresh_db = ints[thresh].tolist()
        except TypeError:  # Cont BW called without a bw_stop argument
            # If site is that bad, just treat as non-responsive
            # Still keeps any estimated onset/offset though.. hm.
            peak_driven_rate = 0
            cf = cf_khz = thresh = thresh_db = None
            bw10 = bw20 = bw30 = bw40 = [None, None]
            continuous_bw = [None, None]
            bw10_octave = bw20_octave = bw30_octave = bw40_octave = None
            continuous_bw_octave = None
            bw10_khz = bw20_khz = bw30_khz = bw40_khz = [None, None]
            continuous_bw_khz = [None, None]

    # Apparently sometimes nan's for latency probs?
    # Just in case, nan_to_num: nan's -> 0, and inf's -> large values
    latency_dict["lats"] = np.nan_to_num(latency_dict["lats"])
    latency_dict["total_prob"] = np.nan_to_num(latency_dict["total_prob"])
    latency_dict["max_prob"] = np.nan_to_num(latency_dict["max_prob"])
    
    analysis_dict = {
        "number": bw_dict["number"],
        "penetration_number": bw_dict["penetration_number"],
        "cf_khz": cf_khz, 
        "threshold_db": thresh_db, 
        "cf_idx": cf,
        "threshold_idx": thresh,
        "bw10_khz": bw10_khz, 
        "bw20_khz": bw20_khz,
        "bw30_khz": bw30_khz,
        "bw40_khz": bw40_khz,
        "bw10_idx": bw10,
        "bw20_idx": bw20, 
        "bw30_idx": bw30,
        "bw40_idx": bw40,
        "bw10_octave": bw10_octave,
        "bw20_octave": bw20_octave,
        "bw30_octave": bw30_octave, 
        "bw40_octave": bw40_octave,
        "continuous_bw_khz": continuous_bw_khz, 
        "continuous_bw_idx": continuous_bw,
        "continuous_bw_octave": continuous_bw_octave,
        "onset_ms": onset, 
        "peak_ms": latency_dict["peak"], 
        "offset_ms": offset,
        "psth": psth.tolist(), 
        "peak_driven_rate_hz": peak_driven_rate,
        "spont_firing_rate_hz": spont,
        "m_probs": latency_dict["m_priors"].tolist(),
        "bb_signal": latency_dict["signal"],
        "sigma": latency_dict["sigma"], 
        "gamma": latency_dict["gamma"],
        "latency_array": latency_dict["lats"].tolist(),
        "bb_latency_prob": latency_dict["max_prob"],
        "bb_total_lat_prob": latency_dict["total_prob"],
        "bb_sdf": latency_dict["sdf"].tolist(),
        "field_assignment": "",
    }
    return_dict = {"data_dict": bw_dict, 
                   "analysis_dict": analysis_dict, 
                   "penetration_number": bw_dict["penetration_number"]}

    return return_dict


def speech_bw_loop(idx, file, total, use_f32, ic_pens=[]):
    """
    Currently this just parses files and stores data for offline analysis.
    Potential to add analysis like DenseTC section in the future.
    Returns dict containing:
      spiketrains: A list of dicts
        {speech_number: #, spikes_ms: spiketimes}
        Use it to create pandas dataframe for easier analysis
      penetration_number: ibid
      filename: for posteriority and prosperity
      number: map # for id
    """
    get_spikes = partial(afunc.get_spike_dict, use_f32=use_f32, 
                         dataset="speech")
    prettify_func = partial(afunc.prettify_spike_dict, dataset="speech")
    bw_dict = read_bw_block(file, use_f32, get_spikes, ic_pens,
                            prettify_func=prettify_func)
    print(f"Working on {idx+1} of {total} Speech files\n"
          f"\tMap number is: {bw_dict['number']}")
    
    return bw_dict


def burst_bw_loop(idx, file, total, use_f32, ic_pens=[]):
    """
    Currently this just parses files and stores data for offline analysis.
    Potential to add analysis like DenseTC section in the future.
    Returns dict containing:
      spiketrains: A list of dicts
        {ISI_ms: Interstimulus interval, spikes_ms: spiketimes}
        Use it to create pandas dataframe for easier analysis
      penetration_number: ibid
      filename: for posteriority and prosperity
      number: map # for id
    """
    get_spikes = partial(afunc.get_spike_dict, use_f32=use_f32,
                         dataset="burst")
    prettify_func = partial(afunc.prettify_spike_dict, dataset="burst")
    bw_dict = read_bw_block(file, use_f32, get_spikes, ic_pens,
                            prettify_func=prettify_func)
    print(f"Working on {idx+1} of {total} Noiseburst files\n"
          f"\tMap number is: {bw_dict['number']}")
    
    return bw_dict
    

def read_bw_block(file, use_f32, get_spikes, ic_pens=[], prettify_func=None):
    """
    Utility func to read Brainware blocks and parse map numbers.
    Returns dict with metadata and a dataset-specific spiketimes data block.
    """
    if use_f32:
        blk = file.read_block()
    else:
        blk = file.read_all_blocks()[0]
    filename = blk.file_origin
    penetration_number = afunc.get_penetration_number(filename)
    map_number = afunc.get_map_number(filename)
    if penetration_number in ic_pens:
        num_offset = np.where(ic_pens == penetration_number)[0][0]
        map_number = map_number - (num_offset * 2)
    
    spike_dict = get_spikes(blk)
    if prettify_func:
        spike_dict = prettify_func(spike_dict)
    return {"spiketrains": spike_dict,
            "filename": filename, 
            "penetration_number": int(penetration_number), 
            "number": int(map_number),}


def get_densetc_bb_lats(psth, n_sweeps, spont, return_sdf=True):
    """
    Bayesian bin latency and SDF
    
    Previous testing suggests that most sites can be analyzed with max_m 
    (num bins) = 10, so leaving that here.
    
    Onset determined by bayesian binning algorithm
    Offset is trickier because it is not always clear when the "signal" returns
    to "spont levels"
    - Lots of possible rules, but many produce terrible, inconsistent results
    - Most successful for me so far is offset approximation based on SDF 
      derivative (slope). This doesn't depend on spontaneous levels or maximal 
      firing rate, but instead on the relative changes in activity for a site.
      - Offset must proceed after a significant reduction in firing rate 
        (mean slope - 1 std) after onset. This helps prevent false-positives 
        from small reductions in firing rate. If this requirement is not 
        satisfied, offset defaults to 300 (some sites are just strange)
      - Offset is considered as the first ms that SDF derivative is approx.
        equal to its mean for AT LEAST 10 ms. Basically a steady return to 
        baseline spontaneous firing rate. If this requirement is not satisfied,
        offset is defaulted to its last candidate (again, rare, strange)
    """
    max_m = 10
    lat_start = 1
    bb_dict = bb.analyze_psth(psth, n_sweeps, spont, max_t=250, max_m=max_m, 
                              lat_start=lat_start, lat_end=150, l_bound=4, 
                              u_bound=max_m, return_sdf=return_sdf)
    sdf = bb_dict["sdf"]
    lats = bb_dict["lats"][lat_start:]
    max_prob = np.amax(lats)
    # Mark onset as first idx with > 15% probability. Why 15%, you ask?
    #   Large enough to ignore spurious results
    #   Small enough that it probably isn't max prob, but is still meaningful.
    #     Basically a greedy latency estimator.
    onset = np.where(0.15 <= lats)[0]
    if onset.any():
        onset = int(onset[0] + lat_start)
    else:
        onset = int(np.argmax(lats) + lat_start)
    
    # If the estimated probability of a latency existing is "too small", then
    # treat it as a non-response.
    if (bb_dict["total_prob"] < 0.2) or (max_prob < 0.1):
        # Using defaults of 50/300 for on/off to make manual work in GUI easy
        onset, peak, offset = 50, None, 300
    else:
        # Use SDF derivative to estimated offset latency
        # Normalized to -1,1 to ease near-equality comparisons
        d_sdf = np.diff(sdf)
        d_norm_sdf = 2. * (d_sdf - np.min(d_sdf)) / np.ptp(d_sdf) - 1
        norm_mean = np.mean(d_norm_sdf)
        norm_std = np.std(d_norm_sdf)
        # Find where SDF derivative array ~nearly~ equals its mean
        equals_mean = np.isclose(d_norm_sdf[onset:], norm_mean, atol=1e-2)
        # Find first large negative SDF slope after onset
        offsets = np.where(d_norm_sdf[onset:] < (norm_mean - norm_std))[0]
        if offsets.any():
            # Potential offsets are where dSDF returns from negative to 
            # nearly == mean
            potential_offsets = np.where(equals_mean[offsets[0]:] == 1)[0]
            if potential_offsets.any():
                # Find sequences of consecutive values that nearly-equal mean. 
                # Empirically 10+ is good to consider an "offset"
                seqs = 1 + np.where(np.diff(potential_offsets) != 1)[0]
                offset_seqs = np.split(potential_offsets, seqs)
                passing_offsets = np.where(
                    np.array([len(x) for x in offset_seqs]) >= 10)[0]
                if passing_offsets.any():
                    offset = int(offset_seqs[passing_offsets[0]][0] + 
                                 offsets[0] + onset)
                else:  # No sequences pass, use last potential offset (rare)
                    offset = int(offset_seqs[-1][0] + offsets[0] + onset)
            else:  # Never returns to mean-levels, use first offset (rare)
                offset = int(offsets[0])
        else:  # No major deflections after onset, default to 300 (rare)
            offset = 300

        # Get peak ms
        peak = int(np.argmax(psth[onset:offset])) + onset
        
    latency_dict = {"onset": onset, 
                    "offset": offset, 
                    "peak": peak,
                    "sdf": sdf, 
                    "lats": lats, 
                    "max_prob": max_prob, 
                    "total_prob": bb_dict["total_prob"], 
                    "signal": bb_dict["signal"],
                    "m_priors": bb_dict["m_priors"],
                    "sigma": bb_dict["sigma"],
                    "gamma": bb_dict["gamma"],}
        
    return latency_dict
