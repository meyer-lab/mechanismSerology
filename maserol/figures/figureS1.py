from tensorpack import xplot_components
from tensordata.serology import concat4D

def makeFigure():
    combined, M, S, _ = concat4D()
    f, axes = xplot_components(combined, 6)
    axes[0].set_yticks([0, S['Sample'].size, S['Sample'].size + M['Sample'].size])
    axes[0].set_yticklabels(["SpaceX", "MGH", "Zohar"])
    return f