import numpy as np
import xarray as xr

from ..resample import resample

trial_data = xr.DataArray(
    np.array(
    [ 
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ]
    ]),
    dims=("Sample", "Antigen", "Receptor"),
    coords={
        "Sample": [0, 1],
        "Receptor": [0, 1],
        "Antigen": ["S", "RBD"],
    }
)

def test_permute():
    resampled = resample(trial_data, replace=False)
    assert resampled.sortby(["Sample"]).equals(trial_data.sortby(["Sample"]))