from matplotlib import gridspec, pyplot as plt
import pandas as pd
import seaborn as sns

def makeComponentPlot(comps, axes):
    rank = comps[0].shape[1]
    components = [str(ii + 1) for ii in range(rank)]

    subs = pd.DataFrame(comps[0], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)], index=axes[0])
    rec = pd.DataFrame(comps[1], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)], index=axes[1])
    ant = pd.DataFrame(comps[2], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)], index=axes[2])

    f = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.5)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    sns.heatmap(subs, cmap="PiYG", center=0, xticklabels=components, yticklabels=axes[0], cbar=True, vmin=-1.0,
                vmax=1.0, ax=ax1)
    sns.heatmap(rec, cmap="PiYG", center=0, xticklabels=components, yticklabels=axes[1], cbar=False, vmin=-1.0,
                vmax=1.0, ax=ax2)
    sns.heatmap(ant, cmap="PiYG", center=0, xticklabels=components, yticklabels=axes[2], cbar=False, vmin=-1.0,
                vmax=1.0, ax=ax3)

    ax1.set_xlabel("Components")
    ax1.set_title("Subjects")
    ax2.set_xlabel("Components")
    ax2.set_title("Receptors")
    ax3.set_xlabel("Components")
    ax3.set_title("Antigens")
    return f