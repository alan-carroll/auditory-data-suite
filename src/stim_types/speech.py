"""Speech stimulus definition."""
from .base import StimulusType, StorageSpec, summarize_table


class SpeechStimulus(StimulusType):
    key = "speech"
    label = "Speech"
    storage = StorageSpec(
        collections={"data": "speech_data"},
        ic_collections={"data": "speech_IC_data"},
    )

    prompt_label = "speech"
    file_label = "speech"
    example = "'vnsspeech_60dB_RZ5_w5dBdummyPA5#001G_RZ5-1_007.src' -> type 'vnsspeech'"
    csv_title = "Open Speech .csv list"
    csv_desc = "speech sounds (name/description) and numbers (integers) used"
    csv_cols = ("speech", "number")
    row_noun = "speech sounds"

    def summarize(self, df):
        return summarize_table(df)

    def store_config(self, config_dict, df):
        config_dict["speech"] = [{"number": r.number, "speech": r.speech}
                                 for r in df.itertuples()]


SPEECH = SpeechStimulus()
