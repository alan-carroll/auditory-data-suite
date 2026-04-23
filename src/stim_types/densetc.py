"""DenseTC stimulus definition and file analysis."""
import bayesian_bins as bb
import numpy as np
import pandas as pd

import analysis_functions as afunc
from analysis_functions import bw_idx_to_units
from brainware import read_bw_block
from site_model import StimConfig

from .base import StimulusType, StorageSpec


class DenseTCStimulus(StimulusType):
    key = "densetc"
    label = "DenseTC"
    storage = StorageSpec(
        collections={"data": "densetc_data", "analysis": "densetc_analysis"},
        ic_collections={
            "data": "densetc_IC_data",
            "analysis": "densetc_IC_analysis",
        },
    )

    prompt_label = "TC"
    file_label = "tuning curve"
    example = "'DenseTC_MPK_digitalatten_JRAC#001G_RZ5-1_007.src' -> type 'DenseTC'"
    csv_title = "Open DenseTC .csv tone list"
    csv_desc = "frequencies (Hz) and intensities (dB SPL) used"
    csv_cols = ("frequency", "intensity")
    row_noun = "tones"

    def summarize(self, df):
        return (f"\nThe frequencies range from: "
                f"{df['frequency'].min()} Hz to {df['frequency'].max()} Hz.\n"
                f"The intensities range from: "
                f"{df['intensity'].min()} dB to {df['intensity'].max()} dB")

    def store_config(self, config_dict, df):
        config_dict["densetc_frequency_hz"] = np.unique(
            df["frequency"].values).tolist()
        config_dict["densetc_intensity_db"] = np.unique(
            df["intensity"].values).tolist()
        config_dict["densetc_num_tones"] = len(df)

    def worker_kwargs(self, config_dict, analysis_id=None, final_file_df=None,
                      return_sdf=True):
        return {
            "cfg": StimConfig.from_project_config(config_dict),
            "analysis_id": analysis_id,
            "final_file_df": final_file_df,
            "return_sdf": return_sdf,
        }

    def analyze_file(self, idx, file, total, use_f32, ic_pens=(), cfg=None,
                     analysis_id=None, final_file_df=None, return_sdf=True,
                     **kwargs):
        result = _densetc_bw_loop(
            idx=idx,
            file=file,
            total=total,
            use_f32=use_f32,
            cfg=cfg,
            ic_pens=ic_pens,
            final_file_df=final_file_df,
            return_sdf=return_sdf,
        )
        result["docs"]["analysis"]["analysis_id"] = analysis_id
        return result


def _densetc_bw_loop(idx, file, total, use_f32, cfg, ic_pens=(),
                     final_file_df=None, return_sdf=True):
    freqs = np.asarray(cfg.frequencies_hz)
    ints = np.asarray(cfg.intensities_db)
    n_sweeps = cfg.num_tones

    bw_dict = read_bw_block(file, use_f32, "densetc", ic_pens)
    map_number = bw_dict["number"]
    print(f"Working on {idx+1} of {total} DenseTC files\n"
          f"\tMap number is: {map_number}")

    spike_dict = bw_dict["spiketrains"]
    all_spikes = afunc.get_times_from_spike_dict(spike_dict, is_pretty=True)
    psth = np.histogram(all_spikes, bins=range(cfg.sweep_length_ms))[0]
    spont, _ = afunc.get_spont(psth, n_sweeps)

    latency_dict = {
        "onset": 50,
        "offset": 300,
        "peak": None,
        "lats": np.array([]),
        "signal": None,
        "max_prob": None,
        "total_prob": None,
        "sdf": np.array([]),
        "m_priors": np.array([]),
        "sigma": None,
        "gamma": None,
    }
    if return_sdf:
        latency_dict = get_densetc_bb_lats(psth, n_sweeps, spont)
    elif final_file_df is not None:
        row = final_file_df[final_file_df["number"] == map_number]
        onset = int(row["onset"].values)
        if not onset:
            onset, peak, offset = 50, None, 300
        else:
            offset = int(row["offset"].values)
            if not offset:
                offset = cfg.sweep_length_ms - 1
            peak = int(np.argmax(psth[onset:offset])) + onset
        latency_dict["onset"] = onset
        latency_dict["peak"] = peak
        latency_dict["offset"] = offset

    onset = latency_dict["onset"]
    offset = latency_dict["offset"]
    peak_driven_rate = afunc.get_peak_driven_rate(psth[onset:offset], spont,
                                                  n_sweeps)

    cf = cf_khz = thresh = thresh_db = None
    bw_idx = {lvl: [None, None] for lvl in cfg.bw_levels_db}
    bw_khz, bw_oct = bw_idx_to_units(bw_idx, freqs)
    continuous_bw = continuous_bw_khz = [None, None]
    continuous_bw_octave = None

    if latency_dict["peak"] is None:
        peak_driven_rate = 0

    elif final_file_df is not None:
        row = final_file_df[final_file_df["number"] == map_number]
        if row["cf"].values != 0:
            cf = afunc.snap_idx(freqs / 1000, row["cf"].values)
            cf_khz = freqs[cf] / 1000
            thresh = afunc.snap_idx(ints, row["thresh"].values)
            thresh_db = ints[thresh]
        for lvl in cfg.bw_levels_db:
            a_raw = row[f"bw{lvl}a"].values
            if a_raw == 0:
                continue
            a = afunc.snap(freqs / 1000, a_raw)
            b = afunc.snap(freqs / 1000, row[f"bw{lvl}b"].values)
            bw_idx[lvl] = [
                int(np.where(int(a * 1000) == freqs.astype(int))[0][0]),
                int(np.where(int(b * 1000) == freqs.astype(int))[0][0]),
            ]
            bw_khz[lvl] = [a, b]
            bw_oct[lvl] = row[f"bw{lvl}"].values[0]

    else:
        tc_df = afunc.get_tuning_curve_dataframe(pd.DataFrame(spike_dict))
        spont_on, spont_off = cfg.spont_window(onset, offset)
        ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
            tc_df,
            driven_onset_ms=onset, driven_offset_ms=offset,
            spont_onset_ms=spont_on, spont_offset_ms=spont_off)
        ttest_tc = afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts)
        r = afunc.ttest_analyze_tuning_curve(ttest_tc)
        if r.cf is None:
            peak_driven_rate = 0
        else:
            cf, thresh = r.cf, r.thresh
            cf_khz = freqs[cf] / 1000
            thresh_db = ints[thresh].tolist()
            bw_idx = r.bw_idx
            bw_khz, bw_oct = bw_idx_to_units(bw_idx, freqs)
            continuous_bw = r.continuous_bw
            continuous_bw_khz = [(freqs[bw] / 1000).tolist()
                                 for bw in continuous_bw]
            continuous_bw_octave = [afunc.get_bandwidth(*freqs[bw]).tolist()
                                    for bw in continuous_bw]

    latency_dict["lats"] = np.nan_to_num(latency_dict["lats"])
    latency_dict["total_prob"] = np.nan_to_num(latency_dict["total_prob"])
    latency_dict["max_prob"] = np.nan_to_num(latency_dict["max_prob"])

    analysis_dict = {
        "number": map_number,
        "penetration_number": bw_dict["penetration_number"],
        "cf_khz": cf_khz,
        "threshold_db": thresh_db,
        "cf_idx": cf,
        "threshold_idx": thresh,
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
    for lvl in cfg.bw_levels_db:
        analysis_dict[f"bw{lvl}_idx"] = bw_idx[lvl]
        analysis_dict[f"bw{lvl}_khz"] = bw_khz[lvl]
        analysis_dict[f"bw{lvl}_octave"] = bw_oct[lvl]

    return {
        "penetration_number": bw_dict["penetration_number"],
        "docs": {
            "data": bw_dict,
            "analysis": analysis_dict,
        },
    }


def _densetc_signal_bounds(spont):
    if spont < 25:
        return 0.001, 0.025
    if spont < 50:
        return 0.025, 0.050
    if spont < 100:
        return 0.050, 0.100
    return 0.100, 0.150


def get_densetc_bb_lats(psth, n_sweeps, spont, return_sdf=True, *, max_t=250,
                        max_m=10, lat_start=1, lat_end=150, l_bound=4,
                        u_bound=None, min_sig_bound=None, max_sig_bound=None):
    if u_bound is None:
        u_bound = max_m
    default_min_sig_bound, default_max_sig_bound = _densetc_signal_bounds(spont)
    if min_sig_bound is None:
        min_sig_bound = default_min_sig_bound
    if max_sig_bound is None:
        max_sig_bound = default_max_sig_bound

    bb_dict = bb.analyze_psth(psth, n_sweeps, spont, max_t=max_t, max_m=max_m,
                              lat_start=lat_start, lat_end=lat_end,
                              l_bound=l_bound, u_bound=u_bound,
                              min_sig_bound=min_sig_bound,
                              max_sig_bound=max_sig_bound,
                              return_sdf=return_sdf)
    sdf = bb_dict["sdf"]
    lats = np.nan_to_num(bb_dict["lats"][lat_start:], nan=0.0, posinf=1.0,
                         neginf=0.0)
    max_prob = float(np.amax(lats))
    onset = np.where(0.15 <= lats)[0]
    if onset.any():
        onset = int(onset[0] + lat_start)
    else:
        onset = int(np.argmax(lats) + lat_start)

    total_prob = float(np.nan_to_num(bb_dict["total_prob"], nan=0.0,
                                     posinf=1.0, neginf=0.0))
    if (total_prob < 0.2) or (max_prob < 0.1):
        onset, peak, offset = 50, None, 300
    else:
        d_sdf = np.diff(sdf)
        d_norm_sdf = 2. * (d_sdf - np.min(d_sdf)) / np.ptp(d_sdf) - 1
        norm_mean = np.mean(d_norm_sdf)
        norm_std = np.std(d_norm_sdf)
        equals_mean = np.isclose(d_norm_sdf[onset:], norm_mean, atol=1e-2)
        offsets = np.where(d_norm_sdf[onset:] < (norm_mean - norm_std))[0]
        if offsets.any():
            potential_offsets = np.where(equals_mean[offsets[0]:] == 1)[0]
            if potential_offsets.any():
                seqs = 1 + np.where(np.diff(potential_offsets) != 1)[0]
                offset_seqs = np.split(potential_offsets, seqs)
                passing_offsets = np.where(
                    np.array([len(x) for x in offset_seqs]) >= 10)[0]
                if passing_offsets.any():
                    offset = int(offset_seqs[passing_offsets[0]][0] +
                                 offsets[0] + onset)
                else:
                    offset = int(offset_seqs[-1][0] + offsets[0] + onset)
            else:
                offset = int(offsets[0] + onset)
        else:
            offset = 300

        peak = int(np.argmax(psth[onset:offset])) + onset

    return {
        "onset": onset,
        "offset": offset,
        "peak": peak,
        "sdf": sdf,
        "lats": lats,
        "max_prob": max_prob,
        "total_prob": total_prob,
        "signal": bb_dict["signal"],
        "m_priors": bb_dict["m_priors"],
        "sigma": bb_dict["sigma"],
        "gamma": bb_dict["gamma"],
    }


DENSETC = DenseTCStimulus()
