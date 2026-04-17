import bisect
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from skimage.measure import label, regionprops

from brainware import check_sweeps

__all__ = [
    "BW_LEVELS", "TCResult", "bw_idx_to_units",
    "snap", "snap_idx", "get_bandwidth",
    "count_spikes", "remove_spont", "get_spont", "get_peak_driven_rate",
    "get_tuning_curve_dataframe", "get_tuning_curve_array",
    "get_driven_vs_spont_spike_counts", "ttest_driven_vs_spont_tc",
    "ttest_analyze_tuning_curve",
]

# dB-above-threshold levels at which bandwidths are measured. Mirrors
# StimConfig.bw_levels_db;
# TODO this is just temp solution that'll be subsumed by a more complete refactor
BW_LEVELS = (10, 20, 30, 40)

TCResult = namedtuple(
    "TCResult",
    "tc_image cf thresh bw_idx continuous_bw"
)

def bw_idx_to_units(bw_idx, freqs_hz):
    """
    Convert {level: [lo_idx, hi_idx]} index pairs into physical units.

    Returns (khz, octave) dicts keyed identically to `bw_idx`.
    Absent levels ([None, None]) map to [None, None] and None.
    """
    freqs_hz = np.asarray(freqs_hz)
    khz, octave = {}, {}
    for lvl, idx in bw_idx.items():
        if idx[0] is None:
            khz[lvl] = [None, None]
            octave[lvl] = None
        else:
            lo, hi = idx
            khz[lvl] = (freqs_hz[[lo, hi]] / 1000).tolist()
            octave[lvl] = get_bandwidth(freqs_hz[lo], freqs_hz[hi]).tolist()
    return khz, octave

def get_bandwidth(bw_start, bw_stop):
    """
    Expects a start and stop frequency (Hz) marking the edges of the bandwidth.
    Uses log2 to transform back into an octave range.
    """
    return np.log2(bw_stop / bw_start)

def snap_idx(grid_vals, input_value, allow_zero=True):
    """
    Index into `grid_vals` of the entry closest to `input_value`.
    Same allow_zero convention as snap(); returns None when triggered.
    """
    if input_value == 0 and not allow_zero:
        return None
    ix = bisect.bisect_right(grid_vals, input_value)
    if ix == 0:
        return 0
    if ix == len(grid_vals):
        return ix - 1
    lo, hi = grid_vals[ix - 1], grid_vals[ix]
    return ix - 1 if abs(lo - input_value) <= abs(hi - input_value) else ix


def snap(grid_vals, input_value, allow_zero=True):
    """
    Returns the value in grid_vals that input_value is closest to.

    Idiosyncrasy: 'final file' values use 0 to indicate a null value. 
      To deal with this and properly return a null value instead of a snapped 
      value, pass keyword argument allow_zero=False.
      """
    idx = snap_idx(grid_vals, input_value, allow_zero)
    return None if idx is None else grid_vals[idx]

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

def _extract_tc_properties(value_image, binary_mask):
    """
    Region analysis on a binarized tuning curve.

    Picks the largest connected region in `binary_mask`, masks
    `value_image` down to that region, and reads CF / threshold /
    bandwidths from it. Returns a TCResult; when no region exists all
    analysis fields are None/absent and `tc_image` is the input
    unchanged.
    """
    labeled = label(binary_mask)
    regions = regionprops(labeled)
    if not regions:
        return TCResult(
            tc_image=value_image, cf=None, thresh=None,
            bw_idx={lvl: [None, None] for lvl in BW_LEVELS},
            continuous_bw=[])

    big = max(regions, key=lambda r: r.area)
    minr, _, maxr, _ = big.bbox

    # Zero everything outside the selected region so BW/CF reads only
    # see the tuning curve, not stray blobs.
    tc = value_image.copy()
    tc[labeled != big.label] = 0

    cf = int(np.argmax(tc[minr, :]))

    # BW at each level: first/last responsive column at the row
    # `level/5` steps above threshold. Row off the grid → absent.
    # The /5 assumes 5 dB intensity spacing.
    # TODO Future PR with StimConfig so this becomes cfg.bw_row_offset(lvl).
    bw_idx = {}
    for lvl in BW_LEVELS:
        try:
            cols = np.where(tc[minr + lvl // 5, :])[0]
            bw_idx[lvl] = [int(cols[0]), int(cols[-1])]
        except IndexError:
            bw_idx[lvl] = [None, None]

    # Continuous BW: every row from thresh+1 to the top of the region.
    # A connected region's bbox guarantees at least 1 pixel per row, so no
    # empty-cols guard needed.
    cont_bw = []
    for row in range(minr + 1, maxr):
        cols = np.where(tc[row, :])[0]
        cont_bw.append([int(cols[0]), int(cols[-1])])

    return TCResult(tc_image=tc, cf=cf, thresh=minr,
                    bw_idx=bw_idx, continuous_bw=cont_bw)

def ttest_analyze_tuning_curve(tc_array):
    """
    Region analysis on a t-test-smoothed TC (output of
    ttest_driven_vs_spont_tc). Any nonzero cell is treated as
    responsive. Returns a TCResult.

    TODO preserves callers' logic during refactor, but this should be
    absorbed eventually since it's so simple now
    """
    return _extract_tc_properties(tc_array, tc_array > 0)

# The legacy gaussian/otsu variant is gone. If it's ever needed again:
#
#   def analyze_tuning_curve(tc_array):
#       blurred = gaussian(tc_array, sigma=1.5)
#       try:
#           t = threshold_otsu(blurred)
#       except ValueError:
#           t = 0
#       return _extract_tc_properties(blurred, blurred > t)