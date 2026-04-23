"""Shared base classes and helpers for stimulus-type modules."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cli_utils as cli
from brainware import read_bw_block
from dialogs import get_file
import pandas as pd


@dataclass(frozen=True)
class StorageSpec:
    collections: dict
    ic_collections: dict


class StimulusType(ABC):
    def prompt(self, config_dict):
        self.prompt_for_config(config_dict)

    def enabled_config_key(self):
        return f"do_{self.key}"

    def prefix_config_key(self):
        return f"{self.key}_file"

    def is_enabled(self, config_dict):
        return bool(config_dict.get(self.enabled_config_key()))

    def set_enabled(self, config_dict, enabled):
        config_dict[self.enabled_config_key()] = int(bool(enabled))

    def file_prefix(self, config_dict, default=None):
        return config_dict.get(self.prefix_config_key(), default)

    def set_file_prefix(self, config_dict, prefix):
        config_dict[self.prefix_config_key()] = prefix

    def prompt_for_config(self, config_dict):
        """
        Run the y/n -> filename-prefix -> CSV -> confirm loop for this
        stimulus type and write the resulting keys into `config_dict`.
        Re-prompts from the top on a CSV load error or a "no" at the
        confirm step.
        """
        while True:
            if not cli.ask_yes_no(
                    f"Are you doing {self.prompt_label} analysis [y/n]? > "
                ):
                self.set_enabled(config_dict, False)
                return

            print("What do your file names uniquely start with?\n"
                  f"This will associate data files with {self.file_label}"
                  " analysis."
            )
            cli.note(f"eg. {self.example} (no quotes, case-sensitive).")
            prefix = input("> ")

            cli.warn(f"\nSelect .csv file containing list of {self.csv_desc}.")
            cols_str = ",".join(self.csv_cols)
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

            self.set_enabled(config_dict, True)
            self.set_file_prefix(config_dict, prefix)
            self.store_config(config_dict, df)
            return

    def worker_kwargs(self, config_dict, analysis_id=None, final_file_df=None,
                      return_sdf=True):
        return {}

    def analyze_file(self, idx, file, total, use_f32, ic_pens=(), **kwargs):
        bw_dict = read_bw_block(file, use_f32, self.key, ic_pens)
        print(f"Working on {idx+1} of {total} {self.label} files\n"
              f"\tMap number is: {bw_dict['number']}")
        return {
            "penetration_number": bw_dict["penetration_number"],
            "docs": {"data": bw_dict},
        }

    @abstractmethod
    def summarize(self, df):
        pass

    @abstractmethod
    def store_config(self, config_dict, df):
        pass


def summarize_table(df):
    return "\n" + df.to_string(index=False)
