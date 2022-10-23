import numpy as np
import xarray as xr
from typing import Literal, Optional

from sklearn.preprocessing import normalize

from .core import reshapeParams, optimizeLoss

def resample(cube : xr.DataArray, replace=True):
    '''
    Returns a DataArray with resampled values from the given 'cube'. Note that
    the Sample coordinates in the resulting xarray will not match those of the
    input array (i.e. sample with  2)

    Args:
      cube: Data xarray
      replace: replace should be True if the resampling should be done with
        replacement. If replace is False, this function just permutes the data
        along the samples dimension.

    Returns:
      Resampled data xarray with same shape as original array.
    '''
    sample_idx = np.random.choice(cube.sizes["Sample"], replace=replace, size=cube.sizes["Sample"])
    return cube.sel(Sample=sample_idx)


def bootstrap(cube : xr.DataArray, numResample=10, norm_cols=True, norm: Optional[Literal["l2", "max"]]=None, **opt_kwargs):
    '''
    Runs bootstrapping algorithm on MTD 'numResample' times.

    Args:
        cube: DataArray object with processed data
        num_resample: number of times to run bootstrapping
        param_dict: kwargs that are passed into optimizeLoss
    
    Returns:
        [[samples mean, samples std], [ag mean, ag std]] or [abundance mean, abundance std]
    '''
    if opt_kwargs['lrank']:
        subjects_list, ag_list = [], []
    else:
        abundance_list = []

    for _ in range(numResample):
        data = resample(cube)
        x, _ = optimizeLoss(data, **opt_kwargs)
        x = reshapeParams(x, data, opt_kwargs['lrank'], opt_kwargs['fitKa'])

        if (opt_kwargs['lrank']):
            subjects_list.append(x[0])
            if norm is not None:
                if norm_cols:
                    ag = normalize(x[1], norm=norm, axis=0)
                else:
                    if norm == "max":
                        ag = x[1] / np.max(x[1])
                    else:
                        ag = x[1] / np.linalg.norm(x[1], 2)
            else:
                ag = x[1]
            ag = normalize(x[1], norm=norm, axis=0) if norm is not None else x[1]
            ag_list.append(ag)
        else:
            abundance_list.append(x[0])
    mean_std = lambda l : (np.mean(np.array(l), axis=0), np.std(np.array(l), axis=0))
    return mean_std(subjects_list), mean_std(ag_list) if opt_kwargs['lrank'] else mean_std(abundance_list)

