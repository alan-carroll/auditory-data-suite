import unittest
from pathlib import Path
from types import SimpleNamespace

from neo.io.brainwaresrcio import BrainwareSrcIO

from brainware import get_spike_dict, prettify_spike_dict


def _fake_spiketrain(times):
    return SimpleNamespace(
        times=SimpleNamespace(
            magnitude=SimpleNamespace(tolist=lambda: list(times))
        )
    )


class BrainwareParsingTests(unittest.TestCase):
    def test_get_spike_dict_falls_back_to_non_unassigned_group_spikes(self):
        blk = SimpleNamespace(
            file_origin="demo.src",
            segments=[
                SimpleNamespace(annotations={}, spiketrains=[]),
                SimpleNamespace(
                    annotations={"freq [Hz]": 1000, "int [dB]": 20},
                    spiketrains=[],
                ),
                SimpleNamespace(
                    annotations={"freq [Hz]": 1000, "int [dB]": 20},
                    spiketrains=[],
                ),
            ],
            groups=[
                SimpleNamespace(name="UnassignedSpikes",
                                spiketrains=[_fake_spiketrain([]),
                                             _fake_spiketrain([])]),
                SimpleNamespace(name="C62",
                                spiketrains=[_fake_spiketrain([]),
                                             _fake_spiketrain([2.0])]),
            ],
        )

        spike_dict = get_spike_dict(blk, use_f32=False, dataset="densetc")

        self.assertEqual(dict(spike_dict), {(1000, 20): [[], [2.0]]})

    def test_demo_src_matches_known_site_1_sweeps(self):
        demo_file = Path(__file__).resolve().parent.parent / "demo" / "data" / \
            "DenseTC_singleRP2#001G_RZ5-1_007.src"
        blk = BrainwareSrcIO(filename=str(demo_file)).read_all_blocks()[0]

        pretty = prettify_spike_dict(
            get_spike_dict(blk, use_f32=False, dataset="densetc"),
            dataset="densetc",
        )

        self.assertEqual(
            pretty[:3],
            [
                {"spikes_ms": [[]], "frequency_hz": 1000, "intensity_db": 75},
                {
                    "spikes_ms": [[
                        15.032320022583008,
                        15.851519584655762,
                        16.99839973449707,
                    ]],
                    "frequency_hz": 1044,
                    "intensity_db": 75,
                },
                {
                    "spikes_ms": [[
                        14.172160148620605,
                        15.851519584655762,
                        305.7254333496094,
                        307.9372863769531,
                        308.3878479003906,
                    ]],
                    "frequency_hz": 1090,
                    "intensity_db": 75,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
