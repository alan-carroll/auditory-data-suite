"""Noiseburst stimulus definition."""
from .base import StimulusType, StorageSpec, summarize_table


class BurstStimulus(StimulusType):
    key = "burst"
    label = "Noiseburst"
    storage = StorageSpec(
        collections={"data": "noiseburst_data"},
        ic_collections={"data": "noiseburst_IC_data"},
    )

    prompt_label = "noiseburst"
    file_label = "noiseburst"
    example = "'bb_noise_train#001G1_7.src' -> type 'bb_noise'"
    csv_title = "Open noiseburst .csv parameters list"
    csv_desc = "noise-burst ISIs (ms) and number of bursts (integers) used"
    csv_cols = ("ISI", "number")
    row_noun = "noiseburst stimuli"

    def summarize(self, df):
        return summarize_table(df)

    def store_config(self, config_dict, df):
        config_dict["burst"] = [{"ISI_ms": r.ISI, "num_bursts": r.number}
                                for r in df.itertuples()]


BURST = BurstStimulus()
