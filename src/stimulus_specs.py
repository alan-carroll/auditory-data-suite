"""
Interactive stimulus-set specification for create_config_file.

Each stimulus type is one StimulusSpec. The spec owns its
prompt strings, CSV column list, and two short callables:
`summarize` produces a human-readable echo of the loaded DataFrame for
the user to confirm, and `store` writes the resulting keys into the
project config dict. Adding a new stimulus type is one StimulusSpec
entry plus its two functions.
"""
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

import cli_utils as cli
from dialogs import get_file

@dataclass(frozen=True)
class StimulusSpec:
    key: str
    prompt_label: str
    file_label: str
    example: str
    csv_title: str
    csv_desc: str
    csv_cols: tuple
    row_noun: str
    summarize: Callable # (df) -> str
    store: Callable     # (cfg, df) -> None, mutates cfg in place

    def prompt(self, config_dict):
        """
        Run the y/n → filename-prefix → CSV → confirm loop for this
        stimulus type and write the resulting keys into `config_dict`.
        Re-prompts from the top on a CSV load error or a "no" at the
        confirm step.
        """
        while True:
            if not cli.ask_yes_no(
                    f"Are you doing {self.prompt_label} analysis [y/n]? > "
                ):
                config_dict[f"do_{self.key}"] = 0
                return

            print("What do your file names uniquely start with?\n"
                  f"This will associate data files with {self.file_label}"
                  " analysis."
            )
            cli.note(f"eg. {self.example} (no quotes, case-sensitive).")
            prefix = input("> ")

            cli.warn(f"\nSelect .csv file containing list of {self.csv_desc}.")
            cols_str = ",".join(f"{c}" for c in self.csv_cols)
            cli.info("Columns must be ordered: " + cols_str)
            cli.warn("Each row as one stimulus entry (no headers):")
            try:
                df = pd.read_csv(
                    get_file(title=self.csv_title,
                             filetypes=[("CSV", ".csv")]),
                    header=None, names=list(self.csv_cols))
            except Exception as e:
                cli.fail(
                    e, 
                    "*** Error loading file. Double check it is correct. ***\n"
                )
                continue

            print(self.summarize(df))
            print(f"There are {len(df)} total {self.row_noun}.")
            print(f"'{prefix}' will be used to identify your files for "
                  f"{self.file_label} analysis.")
            if not cli.ask_yes_no("\nIs this correct [y/n]? > "):
                continue

            config_dict[f"do_{self.key}"] = 1
            config_dict[f"{self.key}_file"] = prefix
            self.store(config_dict, df)
            return

# -- Per-type summarize / store -------------------------------------------

def _summarize_densetc(df):
    return (f"\nThe frequencies range from: "
            f"{df['frequency'].min()} Hz to {df['frequency'].max()} Hz.\n"
            f"The intensities range from: "
            f"{df['intensity'].min()} dB to {df['intensity'].max()} dB")

def _summarize_table(df):
    return "\n" + df.to_string(index=False)

def _store_densetc(cfg, df):
    cfg["densetc_frequency_hz"] = np.unique(df["frequency"].values).tolist()
    cfg["densetc_intensity_db"] = np.unique(df["intensity"].values).tolist()
    cfg["densetc_num_tones"] = len(df)

def _store_speech(cfg, df):
    cfg["speech"] = [{"number": r.number, "speech": r.speech}
                     for r in df.itertuples()]

def _store_burst(cfg, df):
    cfg["burst"] = [{"ISI_ms": r.ISI, "num_bursts": r.number}
                    for r in df.itertuples()]

# -- Registry -------------------------------------------------------------

STIM_SPECS = (
    StimulusSpec(
        key="densetc",
        prompt_label="TC",
        file_label="tuning curve",
        example="'DenseTC_MPK_digitalatten_JRAC#001G_RZ5-1_007.src' "
                "→ type 'DenseTC'",
        csv_title="Open DenseTC .csv tone list",
        csv_desc="frequencies (Hz) and intensities (dB SPL) used",
        csv_cols=("frequency", "intensity"),
        row_noun="tones",
        summarize=_summarize_densetc,
        store=_store_densetc,
    ),
    StimulusSpec(
        key="speech",
        prompt_label="speech",
        file_label="speech",
        example="'vnsspeech_60dB_RZ5_w5dBdummyPA5#001G_RZ5-1_007.src' "
                "→ type 'vnsspeech'",
        csv_title="Open Speech .csv list",
        csv_desc="speech sounds (name/description) and numbers "
                 "(integers) used",
        csv_cols=("speech", "number"),
        row_noun="speech sounds",
        summarize=_summarize_table,
        store=_store_speech,
    ),
    StimulusSpec(
        key="burst",
        prompt_label="noiseburst",
        file_label="noiseburst",
        example="'bb_noise_train#001G1_7.src' → type 'bb_noise'",
        csv_title="Open noiseburst .csv parameters list",
        csv_desc="noise-burst ISIs (ms) and number of bursts "
                 "(integers) used",
        csv_cols=("ISI", "number"),
        row_noun="noiseburst stimuli",
        summarize=_summarize_table,
        store=_store_burst,
    ),
)