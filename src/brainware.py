import re
import itertools
from collections import defaultdict

import numpy as np
from neo.io.brainwaresrcio import BrainwareSrcIO
from neo.io.brainwaref32io import BrainwareF32IO

__all__ = [
    "get_map_number", "get_penetration_number", "adjust_numbers",
    "get_spike_dict", "prettify_spike_dict", "get_times_from_spike_dict",
    "check_sweeps", "read_bw_block",
    "BrainwareSrcIO", "BrainwareF32IO",
]

# (electrode_regex, penetration_regex) pairs, tried in order. First pair
# where both match wins. See get_map_number docstring for the naming
# conventions each row covers.
_FILENAME_PATTERNS = {
    "src": [
        (r"[0-9]{3}(?=\.src)",   r"(?<=#)[0-9]{3}"),
        (r"[0-9]{1,3}(?=\.src)", r"[0-9]{3}(?=e)"),
        (r"[0-9]{1,3}(?=\.src)", r"(?<=_)[0-9]{3}"),
    ],
    "f32": [
        (r"[0-9]{1}(?=\.f32)",   r"[0-9]{3}(?=e)"),
    ],
}

# Hardware electrode IDs 7–10 alias to logical channels 1–4.
_ELECTRODE_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 7: 1, 8: 2, 9: 3, 10: 4}

def _parse_filename(filename):
    """
    Match `filename` against the known Brainware filenaming conventions.
    Returns (electrode_int, penetration_int). Raises ValueError if the
    extension is unrecognized or no pattern pair matches.
    TODO NB example in CLI project prompts, "bb_noise_train#001G1_7.src",
    doens't match here -- fix it!
    """
    ext = filename[-3:]
    if ext not in _FILENAME_PATTERNS:
        raise ValueError(
            f"Expected file extension 'src' or 'f32', got {ext!r} "
            f"from {filename}")
    for elec_re, pen_re in _FILENAME_PATTERNS[ext]:
        e = re.search(elec_re, filename)
        p = re.search(pen_re, filename)
        if e and p:
            return int(e.group()), int(p.group())
    raise ValueError(
        f"Can't parse {filename!r}: no known naming pattern matched.")

def get_map_number(filename):
    """
    Penetration and electrode -> flat map number, parsed from a Brainware
    filename. Some known conventions:
    - DenseTC_..._JRAC#001G_RZ5-1_007.src
    - foo_001e1.src
    - foo_001_1.src
    - naive2dense_001e1.f32
    """
    elec_int, pen = _parse_filename(filename)
    try:
        elec = _ELECTRODE_MAP[elec_int]
    except KeyError:
        raise ValueError(
            f"Expected electrode # in 1-4 or 7-10, got {elec_int} "
            f"from {filename}")
    return (pen - 1) * 4 + elec

def get_penetration_number(filename):
    """Penetration number parsed from a Brainware filename."""
    _, pen = _parse_filename(filename)
    return pen

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
    Flat list of every spiketime across every stimulus and every sweep.

    `spike_dict` is either the raw {params: [[sweep], ...]} mapping from
    get_spike_dict (is_pretty=False) or the list-of-dicts form from
    prettify_spike_dict (is_pretty=True).
    """
    per_stim = ((d["spikes_ms"] for d in spike_dict) if is_pretty
                else spike_dict.values())
    sweeps = itertools.chain.from_iterable(per_stim)
    return list(itertools.chain.from_iterable(sweeps))

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
    
def read_bw_block(file, use_f32, dataset, ic_pens=()):
    """
    Read one neo Brainware block and return its spiketrains plus the
    map / penetration numbers parsed from the filename.

    IC penetrations get a collapsed map-numbering (2 electrodes per
    penetration rather than 4), offset by the penetration's position
    in `ic_pens`.
    """
    blk = file.read_block() if use_f32 else file.read_all_blocks()[0]
    filename = blk.file_origin
    pen = get_penetration_number(filename)
    map_number = get_map_number(filename)
    if pen in ic_pens:
        offset = np.where(ic_pens == pen)[0][0]
        map_number -= offset * 2

    spikes = prettify_spike_dict(
        get_spike_dict(blk, use_f32=use_f32, dataset=dataset),
        dataset=dataset)
    return {"spiketrains": spikes,
            "filename": filename,
            "penetration_number": int(pen),
            "number": int(map_number)}