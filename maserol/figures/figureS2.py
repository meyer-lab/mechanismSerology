from tensordata.kaplonek import MGH, MGH4D
from tensorpack.decomposition import Decomposition
from .common import getSetup
import numpy as np
import tensorflow as tf
from ..linear import compareR2X


def makeFigure():

    ax, f = getSetup((8, 8), (1, 1))

    decomp3D = Decomposition(data=MGH().tensor, max_rr=10)
    decomp3D.perform_tfac()

    tensor4D = tf.convert_to_tensor(np.asarray(MGH4D().values).astype('float64'))
    decomp4D = Decomposition(data=tensor4D, max_rr=10)
    decomp4D.perform_tfac()

    compareR2X(ax[0], decomp3D, decomp4D, "Kaplonek 3D", "Kaplonek 4D")

    return f

