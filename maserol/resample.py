from typing import Literal, Optional

import numpy as np
import xarray as xr
from sklearn.preprocessing import normalize

from .core import reshapeParams, optimizeLoss, DEFAULT_AB_TYPES, factorAbundance

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
    return cube.isel(Sample=sample_idx)


def bootstrap(cube : xr.DataArray, post_opt_factor=False, numResample=10, norm_cols=True, norm: Optional[Literal["l2", "max"]]=None, **opt_kwargs):
    '''
    Runs bootstrapping algorithm on MTD 'numResample' times.

    Args:
        cube: DataArray object with processed data
        num_resample: number of times to run bootstrapping
        param_dict: kwargs that are passed into optimizeLoss
    
    Returns:
        [[samples mean, samples std], [ag mean, ag std]] or [abundance mean, abundance std]
    '''
    if post_opt_factor:
        assert not opt_kwargs["lrank"]
    if opt_kwargs['lrank'] or post_opt_factor:
        samp_list, ag_list = [], []
    else:
        abundance_list = []
    for _ in range(numResample):
        data = resample(cube)
        x, _ = optimizeLoss(data, **opt_kwargs)
        x = reshapeParams(x, data, opt_kwargs['lrank'], opt_kwargs['fitKa'], ab_types=opt_kwargs.get("ab_types", DEFAULT_AB_TYPES), as_xarray=post_opt_factor)

        if (opt_kwargs['lrank']) or post_opt_factor:
            if post_opt_factor:
                abundance = x[0]
                samp_facs, ag_facs = factorAbundance(abundance, 1)
                samp = samp_facs[:, :, 0]
                ag = ag_facs[:, :, 0]
            else:
                samp = x[0]
                ag = x[1]
            samp_list.append(samp)
            if norm is not None:
                if norm_cols:
                    ag = normalize(ag, norm=norm, axis=0)
                else:
                    if norm == "max":
                        ag = ag / np.max(ag)
                    else:
                        ag = ag / np.linalg.norm(ag, 2)
            else:
                ag = ag
            ag = normalize(ag, norm=norm, axis=0) if norm is not None else ag
            ag_list.append(ag)
        else:
            abundance_list.append(x[0])
    mean_std = lambda l : (np.mean(np.array(l), axis=0), np.std(np.array(l), axis=0))
    return mean_std(samp_list), mean_std(ag_list) if opt_kwargs['lrank'] or post_opt_factor else mean_std(abundance_list)

