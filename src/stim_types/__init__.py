"""Registry of available stimulus types."""
from .base import StimulusType, StorageSpec
from .densetc import DENSETC
from .speech import SPEECH
from .burst import BURST

ALL_STIM_TYPES = (
    DENSETC,
    SPEECH,
    BURST,
)

STIM_BY_KEY = {stim.key: stim for stim in ALL_STIM_TYPES}


def enabled_stim_types(config_dict):
    return tuple(stim for stim in ALL_STIM_TYPES if stim.is_enabled(config_dict))


__all__ = [
    "StimulusType",
    "StorageSpec",
    "DENSETC",
    "SPEECH",
    "BURST",
    "ALL_STIM_TYPES",
    "STIM_BY_KEY",
    "enabled_stim_types",
]
