import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd
from skimage.measure import label, regionprops
import datetime
from brainware import BrainwareSrcIO, BrainwareF32IO
from db_adapter import JSONStore
import cli_utils as cli
import analysis_functions as afunc
from stim_types import enabled_stim_types

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

from digit_ocr import DigitOCR

RESOURCES = Path("../resources")
TEMPLATE_FILE = RESOURCES / "digit_templates.npz"
FONT_FILE = RESOURCES / "OCR-A.otf"


def _ic_depth(ic_points_df, penetration_number):
    row = ic_points_df.number == penetration_number
    return int(ic_points_df[row]["depth"].values[0])


def _route_and_insert(results, stim, db, ic_bool, ic_pens, ic_points_df):
    buckets = {kind: [] for kind in stim.storage.collections}
    ic_buckets = {kind: [] for kind in stim.storage.ic_collections}

    for result in results:
        penetration_number = result["penetration_number"]
        docs = result["docs"]
        if ic_bool and penetration_number in ic_pens:
            if "data" in docs:
                docs["data"]["depth"] = _ic_depth(ic_points_df, penetration_number)
            target = ic_buckets
        else:
            target = buckets
        for kind, doc in docs.items():
            target[kind].append(doc)

    for kind, coll_name in stim.storage.collections.items():
        db.collection(coll_name).insert_many(buckets[kind])
    if ic_bool:
        for kind, coll_name in stim.storage.ic_collections.items():
            db.collection(coll_name).insert_many(ic_buckets[kind])


def _gather_brainware_files(dir_path, enabled_stims, use_f32, nums, ic_pens,
                            config_dict):
    if use_f32:
        bw_io, ext = BrainwareF32IO, ".f32"
    else:
        bw_io, ext = BrainwareSrcIO, ".src"

    bw_files = {}
    for stim in enabled_stims:
        pattern = stim.file_prefix(config_dict)
        files = [
            bw_io(filename=dir_path + entry.name)
            for entry in os.scandir(dir_path)
            if entry.name.endswith(ext)
            and entry.name.startswith(pattern)
            and (afunc.get_map_number(entry.name) in nums
                 or afunc.get_penetration_number(entry.name) in ic_pens)
        ]
        bw_files[stim.key] = files
        cli.success(
            f"{len(files)} {ext} {pattern} files found and matched to mapping sites."
        )
    return bw_files


def run_program(config_dict, version, final_file=None, return_sdf=True):
    analysis_version = f"map_auto_analysis v{version}"
    today = str(datetime.datetime.now())
    subject_name = input("What is the subject's name? > ").strip()

    cli.info("Select folder to save to: ")
    save_dir_path = afunc.get_folder(title="Folder to save to")

    use_f32 = False
    file_type = cli.ask_choice("Using .[s]rc or .[f]32 file type? > ",
                               ("f", "s"))
    if file_type == "f":
        use_f32 = True

    ic_bool = False
    ic_only = False
    use_images = False
    image_or_point_list = None
    ic_points_df = pd.DataFrame([{"number": None}])
    if config_dict["do_IC"]:
        response = cli.ask_yes_no("Does this subject have IC data [y/n]? > ")
        if response == "y":
            ic_bool = True
            cli.info(
                "Select .csv file listing IC map Penetration numbers with "
                "corresponding depths (Number,Depth, no headers): "
            )
            ic_csv = afunc.get_file(title="IC Num,Depth .csv",
                                    filetypes=[("CSV", ".csv")])
            ic_points_df = pd.read_csv(ic_csv, header=None,
                                       names=["number", "depth"])
            ic_points_df = ic_points_df.sort_values("number")
            ic_points_df.reset_index(inplace=True, drop=True)

            response = cli.ask_yes_no("Is this an IC only map [y/n]? > ")
            if response == "y":
                ic_only = True

    image_or_point_list = ""
    if (not ic_only) and (final_file is None):
        image_or_point_list = cli.ask_choice(
            "Using [i]mages, [f]inal file, or .[c]sv for map point data? > ",
            ("i", "f", "c")
        )
        if image_or_point_list == "i":
            use_images = True

    db = JSONStore(save_dir_path + subject_name)
    db_metadata = db.metadata
    db_sites = db.sites
    db_analysis_metadata = db.analysis_metadata

    meta_id = db_metadata.insert_one({"program_version": analysis_version,
                                      "program_run_date": today}).inserted_id
    if final_file is not None:
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
    map_points_df = pd.DataFrame([{"number": None}])
    if use_images:
        if TEMPLATE_FILE.exists():
            OCR = DigitOCR.load(TEMPLATE_FILE)
        elif FONT_FILE.exists():
            OCR = DigitOCR.from_font(FONT_FILE)
        else:
            raise FileNotFoundError(
                "No templates or font file found. Add a font .ttf/.otf file "
                "to or run the digit template bootstrap option."
            )

        cli.info("Select map POINTS image:")
        points_im_filename = afunc.get_file(title="Select Map Points image",
                                            filetypes=[("PNG", ".png")])
        points_image = cv2.imread(points_im_filename, cv2.IMREAD_GRAYSCALE)
        points_binary = points_image < 128
        points_label = label(points_binary)
        points_regions = regionprops(points_label)

        cli.info("Select map NUMBERS image:")
        numbers_im_filename = afunc.get_file(title="Select Map Numbers image",
                                             filetypes=[("PNG", ".png")])
        numbers_image = cv2.imread(numbers_im_filename, cv2.IMREAD_GRAYSCALE)
        norm_row_max, norm_col_max = numbers_image.shape

        cli.info("Select map MASK image:")
        mask_im_filename = afunc.get_file(title="Select Map Mask image",
                                          filetypes=[("PNG", ".png")])
        mask_image = cv2.imread(mask_im_filename, cv2.IMREAD_GRAYSCALE)
        mask_binary = mask_image < 128
        mask_label = label(mask_binary)
        mask_regions = regionprops(mask_label)

        if len(mask_regions) != len(points_regions):
            raise AssertionError(
                "Unequal number of Points and Numbers.\n "
                "Were these files made correctly?"
            )

        map_height, map_width = points_image.shape
        points_centroids = np.array([r.centroid for r in points_regions])
        results = []
        for mask_props in mask_regions:
            bbox = mask_props.bbox
            crop = numbers_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            ocr_result = OCR.recognize(crop)

            distances = np.linalg.norm(
                points_centroids - np.array(mask_props.centroid), axis=1)
            point = points_centroids[np.argmin(distances)]
            ocr_result.metadata["x"] = point[1] / norm_col_max
            ocr_result.metadata["y"] = 1 - (point[0] / norm_row_max)
            results.append(ocr_result)

        OCR.review_results(results)
        map_points_df = pd.DataFrame(
            [{"number": r.number, "x": r.metadata["x"], "y": r.metadata["y"]}
             for r in results]
        )

    elif not ic_only:
        if image_or_point_list == "f":
            cli.info(
                "Select spreadsheet containing 'final file' format data:"
            )
            coords_sheet = afunc.get_file(title="Select final file",
                                          filetypes=[("XLS", ".xls")])
            map_points_df = pd.read_excel(coords_sheet,
                                          header=None,
                                          usecols=[40, 41, 43],
                                          names=["x", "y", "number"])
        elif image_or_point_list == "c":
            cli.info("Select .csv file with cols number,x,y (no headers):")
            coords_sheet = afunc.get_file(title="Select Map number,x,y file",
                                          filetypes=[("CSV", ".csv")])
            map_points_df = pd.read_csv(coords_sheet,
                                        header=None,
                                        names=["number", "x", "y"])

        map_width = map_points_df.x.max() * 1000
        map_width = int(map_width + (map_width * 0.1))
        map_height = map_points_df.y.max() * 1000
        map_height = int(map_height + (map_height * 0.1))

    db_metadata.update_one({"_id": meta_id},
                           {"$set": {"map_height": map_height,
                                     "map_width": map_width}})

    if not ic_only:
        max_coor = map_points_df[["x", "y"]].max().values.max()
        map_points_df[["x", "y"]] = map_points_df[["x", "y"]].apply(
            lambda x: afunc.scale_coordinates(input_coor=x,
                                              min_coor=0,
                                              max_coor=max_coor,
                                              min_scale=0.1,
                                              max_scale=0.9))

        print("Working on voronoi data...")
        sites_list, bonus_pts = afunc.pick_voronoi(map_points_df,
                                                   map_width, map_height)
        cli.success("Saving map sites / voronoi data ... \n\n")
        db_sites.insert_many(sites_list)

    cli.info(
        "Select dir containing all Brainware files for subject"
        "(subfolders will be skipped):"
    )
    dir_path = afunc.get_folder(title="Select Brainware data dir")

    nums = map_points_df.number.values
    ic_pens = ic_points_df.number.values if ic_bool else []
    enabled_stims = enabled_stim_types(config_dict)
    bw_files = _gather_brainware_files(dir_path, enabled_stims, use_f32,
                                       nums, ic_pens, config_dict)

    for stim in enabled_stims:
        files = bw_files.get(stim.key)
        if not files:
            continue

        total = len(files)
        worker_kwargs = stim.worker_kwargs(
            config_dict,
            analysis_id=analysis_id,
            final_file=final_file,
            return_sdf=return_sdf,
        )

        @ray.remote
        def worker(idx, file):
            return stim.analyze_file(
                idx=idx,
                file=file,
                total=total,
                use_f32=use_f32,
                ic_pens=ic_pens,
                **worker_kwargs,
            )

        results = ray.get([worker.remote(i, f) for i, f in enumerate(files)])
        cli.success(f"\nSaving {stim.label} data...")
        _route_and_insert(results, stim, db, ic_bool, ic_pens, ic_points_df)

    db.close()
