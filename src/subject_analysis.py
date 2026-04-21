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

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
TEMPLATE_FILE = RESOURCES / "digit_templates.npz"
FONT_FILE = RESOURCES / "OCR-A.otf"
FONT_EXTENSIONS = (".ttf", ".otf")


def default_digit_ocr_source():
    if TEMPLATE_FILE.exists():
        return TEMPLATE_FILE
    if FONT_FILE.exists():
        return FONT_FILE
    return None


def load_digit_ocr(source_path=None):
    if source_path is None:
        raise FileNotFoundError(
            "No templates or font file found. Add a font .ttf/.otf file to "
            f"{RESOURCES}, run a template OCR bootstrap from the main menu, "
            "or choose a template (.npz) or font file (.ttf/.otf) manually."
        )

    source_path = Path(source_path)
    suffix = source_path.suffix.lower()
    if suffix == ".npz":
        return DigitOCR.load(source_path)
    elif suffix in FONT_EXTENSIONS:
        return DigitOCR.from_font(source_path)
    raise FileNotFoundError(
        "Unsupported OCR source. Choose a .npz template set or a "
        ".ttf/.otf font file."
    )


def extract_number_crops(numbers_image, mask_image, threshold=128):
    if numbers_image is None or mask_image is None:
        raise ValueError("Both numbers and mask images are required.")
    if numbers_image.shape != mask_image.shape:
        raise ValueError("Numbers and mask images must share the same shape.")

    mask_binary = mask_image < threshold
    mask_label = label(mask_binary)
    mask_regions = regionprops(mask_label)
    number_crops = [
        numbers_image[min_row:max_row, min_col:max_col]
        for min_row, min_col, max_row, max_col in
        (region.bbox for region in mask_regions)
    ]
    return mask_regions, number_crops


def _load_grayscale_image(title, prompt):
    cli.info(prompt)
    image_path = afunc.get_file(title=title, filetypes=[("PNG", ".png")])
    if not image_path:
        return None, None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Couldn't read image: {image_path}")
    return image_path, image


def load_number_mask_data():
    numbers_path, numbers_image = _load_grayscale_image(
        "Select Map Numbers image", "Select map NUMBERS image:"
    )
    if numbers_image is None:
        return None

    mask_path, mask_image = _load_grayscale_image(
        "Select Map Mask image", "Select matching map MASK image:"
    )
    if mask_image is None:
        return None

    mask_regions, number_crops = extract_number_crops(numbers_image, mask_image)
    return {
        "numbers_path": numbers_path,
        "mask_path": mask_path,
        "numbers_image": numbers_image,
        "mask_image": mask_image,
        "mask_regions": mask_regions,
        "number_crops": number_crops,
    }


def choose_digit_ocr_source():
    picked = afunc.get_file(
        title="Select OCR template or font file",
        filetypes=[
            ("OCR templates or fonts", ".npz .ttf .otf"),
            ("NumPy template archive", ".npz"),
            ("TrueType font", ".ttf"),
            ("OpenType font", ".otf"),
        ],
    )
    return Path(picked) if picked else None


def prompt_digit_ocr_source():
    default_source = default_digit_ocr_source()
    if default_source is None or not cli.ask_yes_no(
        f"Use default OCR source ({default_source}) [y/n]? > "
    ):
        return choose_digit_ocr_source()
    return default_source


def choose_template_save_path():
    if cli.ask_yes_no(
        f"Save templates to default path {TEMPLATE_FILE} [y/n]? > "
    ):
        return TEMPLATE_FILE

    save_path = afunc.save_file(
        title="Save digit template set",
        defaultextension=".npz",
        initialdir=str(TEMPLATE_FILE.parent),
        initialfile=TEMPLATE_FILE.name,
        filetypes=[("NumPy template archive", ".npz")],
    )
    return Path(save_path) if save_path else None


def _normalize_output_path(path, extension):
    path = Path(path)
    ext = extension.lower()

    while path.name.lower().endswith(ext + ext):
        path = path.with_name(path.name[:-len(ext)])

    if path.suffix.lower() != ext:
        path = path.with_suffix(extension)
    return path


def choose_csv_save_path(initial_dir, initial_name):
    save_path = afunc.save_file(
        title="Save OCR coordinates CSV",
        defaultextension=".csv",
        initialdir=str(initial_dir),
        initialfile=initial_name,
        filetypes=[("CSV", ".csv")],
    )
    if not save_path:
        return None

    save_path = _normalize_output_path(save_path, ".csv")
    if save_path.exists() and not cli.ask_yes_no(
        f"Overwrite existing CSV at {save_path} [y/n]? > "
    ):
        return None
    return save_path


def bootstrap_digit_templates():
    template_path = choose_template_save_path()
    if template_path is None:
        cli.warn("Template bootstrap canceled before choosing a save path.")
        return

    number_data = load_number_mask_data()
    crops = number_data["number_crops"]

    if template_path.exists() and not cli.ask_yes_no(
        f"Overwrite existing template set at {template_path} [y/n]? > "
    ):
        return

    template_path.parent.mkdir(parents=True, exist_ok=True)
    ocr = DigitOCR.bootstrap(crops)
    ocr.save(template_path)
    cli.success(f"Saved OCR template set to {template_path}")


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
        if cli.ask_yes_no("Does this subject have IC data [y/n]? > "):
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

            if cli.ask_yes_no("Is this an IC only map [y/n]? > "):
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
        source_path = prompt_digit_ocr_source()
        OCR = load_digit_ocr(source_path)

        _, points_image = _load_grayscale_image(
            "Select Map Points image", "Select map POINTS image:"
        )
        if points_image is None:
            raise FileNotFoundError("No map points image selected.")
        points_binary = points_image < 128
        points_label = label(points_binary)
        points_regions = regionprops(points_label)

        number_data = load_number_mask_data()
        if number_data is None:
            raise FileNotFoundError("Map numbers/mask image selection was canceled.")
        numbers_image = number_data["numbers_image"]
        norm_row_max, norm_col_max = numbers_image.shape
        mask_regions = number_data["mask_regions"]
        number_crops = number_data["number_crops"]

        if len(mask_regions) != len(points_regions):
            raise AssertionError(
                "Unequal number of Points and Numbers.\n "
                "Were these files made correctly?"
            )

        map_height, map_width = points_image.shape
        points_centroids = np.array([r.centroid for r in points_regions])
        results = []
        for mask_props, crop in zip(mask_regions, number_crops):
            ocr_result = OCR.recognize(crop)

            distances = np.linalg.norm(
                points_centroids - np.array(mask_props.centroid), axis=1)
            point = points_centroids[np.argmin(distances)]
            ocr_result.metadata["x"] = point[1] / norm_col_max
            ocr_result.metadata["y"] = 1 - (point[0] / norm_row_max)
            results.append(ocr_result)

        OCR.review_results(results)
        if cli.ask_yes_no("Save coordinates to .csv file [y/n]? > "):
            csv_path = choose_csv_save_path(
                save_dir_path, f"{subject_name}_coords.csv"
            )
            if csv_path is None:
                cli.warn("CSV export canceled.")
            else:
                OCR.export_results_csv(results, csv_path)
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
                                          filetypes=[("Excel workbook", ".xlsx")])
            map_points_df = pd.read_excel(coords_sheet,
                                          header=None,
                                          usecols=[40, 41, 43],
                                          names=["x", "y", "number"],
                                          engine="openpyxl")
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
