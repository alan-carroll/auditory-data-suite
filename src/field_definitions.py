"""
Shared auditory-field definitions used by the GUI and export code.
"""

FIELD_NAMES = ("A1", "VAF", "PAF", "AAF", "SRAF", "NAR", "Other")
MARK_FIELD = "Mark"
GUI_FIELDS = FIELD_NAMES + (MARK_FIELD,)

# Auditory field name -> integer code used by the v-plot final-file format.
FIELD_EXPORT_CODES = {
    "": 0,
    "A1": 0,
    "AAF": 1,
    "PAF": 2,
    "Other": 3,
    "VAF": 4,
    "NAR": 5,
    "SRAF": 6,
}

FIELD_FILL_COLORS = {
    "A1": "#3e82fc",    # xkcd:dodger blue
    "VAF": "#ffff81",   # xkcd:butter
    "PAF": "#90fda9",   # xkcd:foam green
    "AAF": "#fc86aa",   # xkcd:pinky
    "SRAF": "#edc8ff",  # xkcd:light lilac
    "NAR": "#5a7d9a",   # xkcd:steel blue
    "Other": "#b04e0f", # xkcd:burnt sienna
    "Mark": "#c1fd95",  # xkcd:celery
}

FIELD_LINE_COLORS = {
    "A1": "#0348c9",
    "VAF": "#ffff00",
    "PAF": "#37fb65",
    "AAF": "#fa3872",
    "SRAF": "#c44dff",
    "NAR": "#394e60",
    "Other": "#5e2908",
    "Mark": "#60dc04",
}
