"""
Per-site data model, decoupled from the Kivy widgets that render it.

One SiteModel per recording site. The overview plots and the detail
view hold a reference to the same model, so building the tuning curve
DataFrame happens once per site instead of once per widget.

Caching policy:
  - tuning_curve_df is the only cached_property. It's built from raw
    spiketrains; nothing the user does can change it.
  - raw_tc / ttest_tc / contour_tc are functions of (onset, offset).
    Each carries a one-slot memo keyed on that pair, so toggling display
    mode or picking CF (redraws that don't touch latencies) are cache
    hits. Dragging a latency line changes the key and forces a
    recompute. No invalidation code anywhere since the key mismatch does it.
"""
from dataclasses import dataclass, field
from functools import cached_property
from copy import deepcopy
import numpy as np
import pandas as pd
import analysis_functions as afunc


@dataclass(frozen=True)
class StimConfig:
    """
    Stimulus / recording parameters. Replaces hard-coded values scattered 
    through Field_Selection_GUI (400 ms sweep len, dB steps, etc). Built once
    from project_configuration in _load() and treated as read-only.
    """
    frequencies_hz: tuple
    intensities_db: tuple
    num_tones: int
    sweep_length_ms: int
    bw_levels_db: tuple = field(default_factory=lambda: (10, 20, 30, 40))

    def __post_init__(self):
        object.__setattr__(self, "frequencies_hz", tuple(self.frequencies_hz))
        object.__setattr__(self, "intensities_db", tuple(self.intensities_db))
        object.__setattr__(self, "bw_levels_db", tuple(self.bw_levels_db))

    @property
    def num_frequency(self):
        return len(self.frequencies_hz)

    @property
    def num_intensity(self):
        return len(self.intensities_db)

    @cached_property
    def intensity_step_db(self):
        """
        Spacing between intensity levels, used to turn a BW level (e.g.
        20 dB above threshold) into a row offset on the TC grid. The
        stimulus has always been uniformly spaced; if that stops being
        true, bw_row_offset needs to become a searchsorted lookup.
        """
        steps = np.diff(self.intensities_db)
        if len(steps) == 0:
            return 1
        if not np.allclose(steps, steps[0]):
            raise ValueError(
                f"Non-uniform intensity steps {steps}; "
                "BW row positioning assumes uniform spacing.")
        return int(steps[0])

    def bw_row_offset(self, level_db):
        q, r = divmod(level_db, self.intensity_step_db)
        if r != 0:
            raise ValueError(
                f"BW level {level_db} dB does not align with "
                f"{self.intensity_step_db} dB intensity spacing.")
        return q

    def spont_window(self, onset_ms, offset_ms):
        """
        (spont_start, spont_end) — same-length window at the tail of the
        sweep. Replaces the `400 - (offset - onset)` expression pasted at
        every remove_spont / get_driven_vs_spont call site.
        """
        driven = offset_ms - onset_ms
        if driven < 0:
            raise ValueError(
                f"offset_ms must be >= onset_ms, got {offset_ms} < {onset_ms}")
        if driven > self.sweep_length_ms:
            raise ValueError(
                f"Driven window {driven} ms exceeds sweep length "
                f"{self.sweep_length_ms} ms")
        return self.sweep_length_ms - driven, self.sweep_length_ms

    @classmethod
    def from_project_config(cls, cfg, fallback_sweep_ms=None):
        # Prefer an explicit config key; fall back to whatever the caller
        # sniffed (e.g. len(psth)); then the historical default. Add
        # densetc_sweep_length_ms to new analysis_metadata docs going
        # forward and this fallback stops mattering.
        sweep = cfg.get("densetc_sweep_length_ms") or fallback_sweep_ms or 400
        return cls(
            frequencies_hz=sorted(cfg["densetc_frequency_hz"]),
            intensities_db=sorted(cfg["densetc_intensity_db"]),
            num_tones=cfg["densetc_num_tones"],
            sweep_length_ms=sweep,
        )


@dataclass(slots=True)
class AnalysisState:
    """
    Mutable per-site analysis values (written back to the DB on save).
        model.working is what the user edits
        model.saved is the last committed snapshot
        reset/commit become one-liners.
    """
    cf_idx: int
    thresh_idx: int
    onset: int
    offset: int
    peak: int
    peak_driven_rate: float
    bw_idx: dict  # {level_db: [lo, hi] or [None, None]}
    continuous_bw_idx: list
    marked: bool

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_db(cls, doc, bw_levels):
        return cls(
            cf_idx=doc["cf_idx"],
            thresh_idx=doc["threshold_idx"],
            onset=doc["onset_ms"],
            offset=doc["offset_ms"],
            peak=doc["peak_ms"],
            peak_driven_rate=doc["peak_driven_rate_hz"],
            bw_idx={lvl: list(doc[f"bw{lvl}_idx"]) for lvl in bw_levels},
            continuous_bw_idx=list(doc["continuous_bw_idx"]),
            marked=doc.get("marked", False),  # absent in pre-mark analyses
        )


class SiteModel:
    def __init__(self, site_number, data_doc, analysis_doc, config):
        self.site_number = site_number
        self.config = config
        self._data_doc = data_doc

        self.raw_psth = np.asarray(analysis_doc["psth"])
        self.spont_rate = analysis_doc["spont_firing_rate_hz"]
        self.sdf = np.asarray(analysis_doc.get("bb_sdf", 0))  # absent pre-SDF

        self.saved = AnalysisState.from_db(analysis_doc, config.bw_levels_db)
        self.working = self.saved.copy()

        # One-slot memos: ((onset, offset), array). See module docstring.
        self._raw_tc_cache = (None, None)
        self._ttest_tc_cache = (None, None)
        self._contour_cache = (None, None)

    @cached_property
    def tuning_curve_df(self):
        site_df = pd.DataFrame(self._data_doc["spiketrains"])
        return afunc.get_tuning_curve_dataframe(site_df)

    def _window(self, onset, offset):
        on = self.working.onset if onset is None else onset
        off = self.working.offset if offset is None else offset

        if on < 0:
            raise ValueError(f"onset must be >= 0, got {on}")
        if off < on:
            raise ValueError(
                f"offset must be >= onset, got onset={on}, offset={off}")
        if off > self.config.sweep_length_ms:
            raise ValueError(
                f"offset {off} exceeds sweep length "
                f"{self.config.sweep_length_ms}")

        return on, off

    def raw_tc(self, onset=None, offset=None):
        # TODO This should be reworked to allow *actual* raw, and then the
        # spont-adjusted one for analysis views separate
        """Spont-subtracted spike counts per (intensity, frequency)."""
        key = self._window(onset, offset)
        if self._raw_tc_cache[0] == key:
            return self._raw_tc_cache[1]

        on, off = key
        s_on, s_off = self.config.spont_window(on, off)
        def per_cell(x):
            if x is None or np.any(np.isnan(x)):
                return 0
            return afunc.remove_spont(
                x, driven_onset_ms=on, driven_offset_ms=off,
                spont_onset_ms=s_on, spont_offset_ms=s_off)
        arr = np.asarray(self.tuning_curve_df.map(per_cell), dtype=np.uint16)

        self._raw_tc_cache = (key, arr)
        return arr

    def ttest_tc(self, onset=None, offset=None):
        """Smoothed (t-test) TC at the given window."""
        key = self._window(onset, offset)
        if self._ttest_tc_cache[0] == key:
            return self._ttest_tc_cache[1]

        on, off = key
        s_on, s_off = self.config.spont_window(on, off)
        counts = afunc.get_driven_vs_spont_spike_counts(
            self.tuning_curve_df,
            driven_onset_ms=on, driven_offset_ms=off,
            spont_onset_ms=s_on, spont_offset_ms=s_off)
        arr = afunc.ttest_driven_vs_spont_tc(*counts)

        self._ttest_tc_cache = (key, arr)
        return arr

    def contour_tc(self, onset=None, offset=None):
        """Binary mask for the contour overlay"""
        key = self._window(onset, offset)
        if self._contour_cache[0] == key:
            return self._contour_cache[1]

        smooth = afunc.ttest_analyze_tuning_curve(self.ttest_tc(*key))[0]
        smooth[smooth > 0] = 1

        self._contour_cache = (key, smooth)
        return smooth

    def reset(self):
        self.working = self.saved.copy()

    def commit(self):
        self.saved = self.working.copy()
