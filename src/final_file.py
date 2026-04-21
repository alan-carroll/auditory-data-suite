import numpy as np
import pandas as pd
import shapely.geometry as geometry
from pathlib import Path

import cli_utils as cli
from dialogs import get_file, save_file, load_analysis
from db_adapter import JSONStore
from tc_analysis import BW_LEVELS

__all__ = ["create_final_file"]

# Auditory field name -> integer code used by the v-plot final-file format.
# TODO unify with GUI
_FIELD_CODES = {"": 0, "A1": 0, "AAF": 1, "PAF": 2, "Other": 3,
                "VAF": 4, "NAR": 5, "SRAF": 6}

def create_final_file(ic_bool=False):
    """
    Load an analysis for a subject and export to v-plot style 'final file'.
    Writes a .xlsx file. Import into MATLAB to save as .mat file.
    
    IC final files have a reduced export.
    """
    cli.info("Select subject database file: ")
    db_path = get_file(title="Select database JSON file", 
                       filetypes=[("JSON", ".json")])
    if (db_path is None) or (db_path == ""):
        return

    # Initialize length of final-file matrix. Currently 88 from MATLAB vplot
    # style, but arbitrary
    array_length = 88

    # Starting column for each BW triple (a, b, octave) in v-plot layout.
    BW_FF_COLS = {10: 6, 20: 11, 30: 16, 40: 21}

    # Initialize database
    subject_database = JSONStore(db_path)

    if ic_bool:
        densetc_analysis_collection = subject_database.densetc_IC_analysis
    else:
        densetc_analysis_collection = subject_database.densetc_analysis
    analysis_metadata_collection = subject_database.analysis_metadata

    # Load an existing analysis to export. The create-new button in
    # load_analysis() still exists (shared UI with the GUI's load flow),
    # but creating a fresh analysis during export is pointless — you'd
    # just be exporting the template you cloned. Bail with a message if
    # the user tries it.
    analysis_selection, create_new_analysis = \
        load_analysis(analysis_metadata_collection.find({}))
    if analysis_selection is None:
        return
    if create_new_analysis:
        cli.warn(
            "Can't create a new analysis during final-file export — "
            "there'd be nothing in it yet. Pick an existing one."
        )
        return
    analysis_id = analysis_selection["_id"]

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
        
        field = _FIELD_CODES[analysis_entry["field_assignment"]]
        
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
            cf = thresh = onset = offset = peak = peak_driven_rate = 0
        else:
            cf = analysis_entry["cf_khz"]
            thresh = analysis_entry["threshold_db"]
            for lvl in BW_LEVELS:
                if analysis_entry[f"bw{lvl}_idx"][0] is None:
                    continue  # final_file row is already zero
                a, b = analysis_entry[f"bw{lvl}_khz"]
                col = BW_FF_COLS[lvl]
                final_file[idx, col]     = a
                final_file[idx, col + 1] = b
                final_file[idx, col + 2] = analysis_entry[f"bw{lvl}_octave"]

            onset = analysis_entry["onset_ms"]
            peak = analysis_entry["peak_ms"]
            offset = analysis_entry["offset_ms"]

            peak_driven_rate = analysis_entry["peak_driven_rate_hz"]
            
        final_file[idx, 0] = file_number
        final_file[idx, 1] = cf
        final_file[idx, 2] = thresh
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

    cli.info("Select a location and file name to save your 'final file' to: ")
    save_location = save_file(title="Save final file", 
                              filetypes=[("Excel workbook", ".xlsx")])
    if not save_location:
        return
    save_path = Path(save_location)
    if save_path.suffix.lower() != ".xlsx":
        save_path = save_path.with_suffix(".xlsx")
    print(save_path)
    # Write final file out to .xlsx file.
    # MATLAB will be used to convert to .mat file
    df = pd.DataFrame(final_file)
    with pd.ExcelWriter(save_path, engine="openpyxl") as excel_writer:
        df.to_excel(excel_writer, index=False, header=False)
