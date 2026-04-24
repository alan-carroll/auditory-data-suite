import cv2
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from skimage.measure import label, regionprops
import datetime
from runtime_config import (
    configure_analysis_process_environment,
    configure_numba_worker_threads,
    ray_numba_threads,
    ray_worker_env_vars,
)

configure_analysis_process_environment()

from brainware import BrainwareSrcIO, BrainwareF32IO
from db_adapter import JSONStore
import cli_utils as cli
import analysis_functions as afunc
from stim_types import enabled_stim_types

import ray

from digit_ocr import DigitOCR

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
TEMPLATE_FILE = RESOURCES / "digit_templates.npz"
FONT_FILE = RESOURCES / "OCR-A.otf"
FONT_EXTENSIONS = (".ttf", ".otf")


@dataclass(slots=True)
class RunContext:
    config_dict: dict
    version: str
    analysis_version: str
    subject_name: str
    save_dir_path: Path
    db_path: Path
    db: JSONStore
    meta_id: str
    analysis_id: str
    use_f32: bool
    ic_bool: bool
    ic_only: bool
    ic_points_df: pd.DataFrame
    final_file_df: pd.DataFrame | None
    return_sdf: bool


@dataclass(slots=True)
class MapData:
    map_points_df: pd.DataFrame
    map_width: int = 1
    map_height: int = 1


def _empty_number_df():
    return pd.DataFrame([{"number": None}])


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


def choose_json_save_path(initial_dir, initial_name):
    save_path = afunc.save_file(
        title="Save subject database JSON",
        defaultextension=".json",
        initialdir=str(initial_dir),
        initialfile=initial_name,
        filetypes=[("JSON", ".json")],
    )
    if not save_path:
        return None
    return _normalize_output_path(save_path, ".json")


def resolve_subject_db_path(save_dir_path, subject_name):
    db_path = _normalize_output_path(Path(save_dir_path) / subject_name, ".json")

    while db_path.exists():
        cli.warn(f"Subject database already exists: {db_path}")
        choice = cli.ask_choice(
            "Choose [o]verwrite, [r]ename, or [c]ancel > ",
            ("o", "r", "c"),
        )
        if choice == "o":
            return db_path, True
        if choice == "c":
            return None, False

        renamed_path = choose_json_save_path(db_path.parent, db_path.name)
        if renamed_path is None:
            cli.warn("Rename canceled. Pick another option.")
            continue
        db_path = renamed_path

    return db_path, False


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
            bw_io(filename=entry.path)
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


@ray.remote
def _analyze_stimulus_file(stim, idx, file, total, use_f32, ic_pens,
                           worker_kwargs, numba_threads):
    try:
        configure_numba_worker_threads(numba_threads)
        return stim.analyze_file(
            idx=idx,
            file=file,
            total=total,
            use_f32=use_f32,
            ic_pens=ic_pens,
            **worker_kwargs,
        )
    except Exception as e:
        file_name = getattr(file, "_filename", None)
        if file_name is None:
            file_name = getattr(file, "filename", "<unknown file>")
        raise RuntimeError(
            f"{stim.key} analysis failed for file {file_name!r} at worker "
            f"index {idx} of {total}"
        ) from e


def _prompt_ic_map_options(config_dict):
    ic_bool = False
    ic_only = False
    ic_points_df = _empty_number_df()

    if not config_dict["do_IC"]:
        return ic_bool, ic_only, ic_points_df

    if not cli.ask_yes_no("Does this subject have IC data [y/n]? > "):
        return ic_bool, ic_only, ic_points_df

    ic_bool = True
    cli.info(
        "Select .csv file listing IC map Penetration numbers with "
        "corresponding depths (Number,Depth, no headers): "
    )
    ic_csv = afunc.get_file(title="IC Num,Depth .csv",
                            filetypes=[("CSV", ".csv")])
    if not ic_csv:
        cli.warn("Analysis canceled before choosing an IC depth file.")
        return None

    ic_points_df = pd.read_csv(ic_csv, header=None,
                               names=["number", "depth"])
    ic_points_df = ic_points_df.sort_values("number").reset_index(drop=True)

    if cli.ask_yes_no("Is this an IC only map [y/n]? > "):
        ic_only = True

    return ic_bool, ic_only, ic_points_df


def prepare_run_context(config_dict, version, final_file_df=None,
                        return_sdf=True):
    analysis_version = f"map_auto_analysis v{version}"
    today = str(datetime.datetime.now())
    subject_name = input("What is the subject's name? > ").strip()
    if not subject_name:
        cli.warn("Analysis canceled because the subject name was blank.")
        return None

    cli.info("Select folder to save to: ")
    save_dir_path = afunc.get_folder(title="Folder to save to")
    if not save_dir_path:
        cli.warn("Analysis canceled before choosing a save folder.")
        return None
    save_dir_path = Path(save_dir_path)

    db_path, overwrite_existing = resolve_subject_db_path(save_dir_path,
                                                          subject_name)
    if db_path is None:
        cli.warn("Analysis canceled before writing a subject database.")
        return None

    use_f32 = False
    file_type = cli.ask_choice("Using .[s]rc or .[f]32 file type? > ",
                               ("f", "s"))
    if file_type == "f":
        use_f32 = True

    ic_options = _prompt_ic_map_options(config_dict)
    if ic_options is None:
        return None
    ic_bool, ic_only, ic_points_df = ic_options

    if overwrite_existing:
        db_path.unlink()
        cli.warn(f"Overwriting existing subject database at {db_path}")

    db = JSONStore(db_path)
    db_metadata = db.metadata
    db_analysis_metadata = db.analysis_metadata

    meta_id = db_metadata.insert_one({
        "program_version": analysis_version,
        "program_run_date": today,
        "project_configuration": config_dict,
    }).inserted_id
    if final_file_df is not None:
        analysis_comment = "Tuning curve analysis generated from a final file"
    else:
        analysis_comment = "Auto tuning curve analysis and data pre-processing"
    analysis_id = db_analysis_metadata.insert_one({
        "name": analysis_version,
        "start_date": today,
        "last_modified": today,
        "frozen": True,
        "comments": analysis_comment,
    }).inserted_id

    return RunContext(
        config_dict=config_dict,
        version=version,
        analysis_version=analysis_version,
        subject_name=subject_name,
        save_dir_path=save_dir_path,
        db_path=db_path,
        db=db,
        meta_id=meta_id,
        analysis_id=analysis_id,
        use_f32=use_f32,
        ic_bool=ic_bool,
        ic_only=ic_only,
        ic_points_df=ic_points_df,
        final_file_df=final_file_df,
        return_sdf=return_sdf,
    )


def _map_dimension_from_coords(values):
    dimension = values.max() * 1000
    return int(dimension + (dimension * 0.1))


def _ingest_map_points_from_images(run_ctx):
    source_path = prompt_digit_ocr_source()
    if source_path is None:
        cli.warn("Analysis canceled before choosing an OCR source.")
        return None
    ocr = load_digit_ocr(source_path)

    _, points_image = _load_grayscale_image(
        "Select Map Points image", "Select map POINTS image:"
    )
    if points_image is None:
        cli.warn("Analysis canceled before choosing a map points image.")
        return None
    points_binary = points_image < 128
    points_label = label(points_binary)
    points_regions = regionprops(points_label)

    number_data = load_number_mask_data()
    if number_data is None:
        cli.warn("Analysis canceled before choosing map numbers/mask images.")
        return None
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
        ocr_result = ocr.recognize(crop)

        distances = np.linalg.norm(
            points_centroids - np.array(mask_props.centroid), axis=1)
        point = points_centroids[np.argmin(distances)]
        ocr_result.metadata["x"] = point[1] / norm_col_max
        ocr_result.metadata["y"] = 1 - (point[0] / norm_row_max)
        results.append(ocr_result)

    ocr.review_results(results)
    if cli.ask_yes_no("Save coordinates to .csv file [y/n]? > "):
        csv_path = choose_csv_save_path(
            run_ctx.save_dir_path, f"{run_ctx.subject_name}_coords.csv"
        )
        if csv_path is None:
            cli.warn("CSV export canceled.")
        else:
            ocr.export_results_csv(results, csv_path)

    return MapData(
        map_points_df=pd.DataFrame(
            [{"number": r.number, "x": r.metadata["x"], "y": r.metadata["y"]}
             for r in results]
        ),
        map_width=map_width,
        map_height=map_height,
    )


def _ingest_map_points_from_final_sheet():
    cli.info("Select spreadsheet containing 'final file' format data:")
    coords_sheet = afunc.get_file(title="Select final file",
                                  filetypes=[("Excel workbook", ".xlsx")])
    if not coords_sheet:
        return None

    map_points_df = pd.read_excel(coords_sheet,
                                  header=None,
                                  usecols=[40, 41, 43],
                                  names=["x", "y", "number"],
                                  engine="openpyxl")
    return MapData(
        map_points_df=map_points_df,
        map_width=_map_dimension_from_coords(map_points_df.x),
        map_height=_map_dimension_from_coords(map_points_df.y),
    )


def _ingest_map_points_from_csv():
    cli.info("Select .csv file with cols number,x,y (no headers):")
    coords_sheet = afunc.get_file(title="Select Map number,x,y file",
                                  filetypes=[("CSV", ".csv")])
    if not coords_sheet:
        return None

    map_points_df = pd.read_csv(coords_sheet,
                                header=None,
                                names=["number", "x", "y"])
    return MapData(
        map_points_df=map_points_df,
        map_width=_map_dimension_from_coords(map_points_df.x),
        map_height=_map_dimension_from_coords(map_points_df.y),
    )


def _persist_map_data(run_ctx, map_data):
    run_ctx.db.metadata.update_one(
        {"_id": run_ctx.meta_id},
        {"$set": {"map_height": map_data.map_height,
                  "map_width": map_data.map_width}},
    )

    if run_ctx.ic_only:
        return map_data

    scaled_map_points_df = map_data.map_points_df.copy()
    max_coor = scaled_map_points_df[["x", "y"]].max().values.max()
    scaled_map_points_df[["x", "y"]] = scaled_map_points_df[["x", "y"]].apply(
        lambda x: afunc.scale_coordinates(input_coor=x,
                                          min_coor=0,
                                          max_coor=max_coor,
                                          min_scale=0.1,
                                          max_scale=0.9))

    print("Working on voronoi data...")
    sites_list, _ = afunc.pick_voronoi(scaled_map_points_df,
                                       map_data.map_width,
                                       map_data.map_height)
    cli.success("Saving map sites / voronoi data ... \n\n")
    run_ctx.db.sites.insert_many(sites_list)
    return MapData(
        map_points_df=scaled_map_points_df,
        map_width=map_data.map_width,
        map_height=map_data.map_height,
    )


def ingest_map_points(run_ctx):
    if run_ctx.ic_only:
        return _persist_map_data(run_ctx, MapData(map_points_df=_empty_number_df()))

    map_point_source = cli.ask_choice(
        "Using [i]mages, [f]inal file, or .[c]sv for map point data? > ",
        ("i", "f", "c")
    )
    if map_point_source == "i":
        map_data = _ingest_map_points_from_images(run_ctx)
    elif map_point_source == "f":
        map_data = _ingest_map_points_from_final_sheet()
    else:
        map_data = _ingest_map_points_from_csv()

    if map_data is None:
        cli.warn("Analysis canceled before loading map-point data.")
        return None

    return _persist_map_data(run_ctx, map_data)


def _analysis_numbers(run_ctx, map_data):
    if run_ctx.final_file_df is not None:
        return run_ctx.final_file_df.number.values
    if map_data is not None:
        return map_data.map_points_df.number.values
    return _empty_number_df().number.values


def run_brainware_analysis(run_ctx, map_data=None):
    config_dict = run_ctx.config_dict
    use_f32 = run_ctx.use_f32
    analysis_id = run_ctx.analysis_id
    final_file_df = run_ctx.final_file_df
    return_sdf = run_ctx.return_sdf
    ic_bool = run_ctx.ic_bool
    ic_points_df = run_ctx.ic_points_df
    db = run_ctx.db

    cli.info(
        "Select dir containing all Brainware files for subject"
        "(subfolders will be skipped):"
    )
    dir_path = afunc.get_folder(title="Select Brainware data dir")
    if not dir_path:
        cli.warn("Analysis canceled before choosing a Brainware data folder.")
        return False

    nums = _analysis_numbers(run_ctx, map_data)
    ic_pens = ic_points_df.number.values if ic_bool else []
    enabled_stims = enabled_stim_types(config_dict)
    bw_files = _gather_brainware_files(dir_path, enabled_stims,
                                       use_f32, nums, ic_pens,
                                       config_dict)

    for stim in enabled_stims:
        files = bw_files.get(stim.key)
        if not files:
            continue

        total = len(files)
        worker_kwargs = stim.worker_kwargs(
            config_dict,
            analysis_id=analysis_id,
            final_file_df=final_file_df,
            return_sdf=return_sdf,
        )
        numba_threads = ray_numba_threads()
        cli.info(
            f"Using Ray workers with {numba_threads} Numba thread"
            f"{'' if numba_threads == 1 else 's'} each for {stim.label}."
        )
        analyze_task = _analyze_stimulus_file.options(
            num_cpus=numba_threads,
            runtime_env={"env_vars": ray_worker_env_vars(numba_threads)},
        )
        results = ray.get([
            analyze_task.remote(
                stim, i, f, total, use_f32, ic_pens, worker_kwargs,
                numba_threads
            )
            for i, f in enumerate(files)
        ])
        cli.success(f"\nSaving {stim.label} data...")
        _route_and_insert(results, stim, db, ic_bool, ic_pens, ic_points_df)

    return True


def run_program(config_dict, version, final_file_df=None, return_sdf=True):
    run_ctx = prepare_run_context(config_dict, version,
                                  final_file_df=final_file_df,
                                  return_sdf=return_sdf)
    if run_ctx is None:
        return False

    try:
        map_data = None
        if run_ctx.final_file_df is None:
            map_data = ingest_map_points(run_ctx)
            if map_data is None:
                return False

        return run_brainware_analysis(run_ctx, map_data=map_data)
    finally:
        run_ctx.db.close()
