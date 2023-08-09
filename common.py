from collections import namedtuple

params = {
    "data_path": "data",
    "output_path": "outputs-test",
    "num_rows_per_linescan_frame": 50,
}

CommonParams = namedtuple(
    "CommonParams", field_names=list(params.keys()), defaults=params.values()
)

common = CommonParams()

FrameRange = namedtuple("FrameRange", ["start", "end"])
