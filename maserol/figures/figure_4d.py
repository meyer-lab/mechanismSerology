import seaborn as sns
from tensordata.alter import load_file, data as alter

from maserol.preprocess import Rtot_to_df
from maserol.core import optimize_loss
from maserol.preprocess import assemble_options, prepare_data


def makeFigure():
    data = alter()["Fc"]
    data = data.sel(
        Receptor=[
            "FcgRIIa.H131",
            "FcgRIIa.R131",
            "FcgRIIb",
            "FcgRIIIa.F158",
            "FcgRIIIa.V158",
            "FcgRIIIb",
            "IgG1",
            "IgG3",
        ]
    )
    translate = {
        "FcgRIIa.H131": "FcgRIIA-131H",
        "FcgRIIa.R131": "FcgRIIA-131R",
        "FcgRIIIa.F158": "FcgRIIIA-158F",
        "FcgRIIIa.V158": "FcgRIIIA-158V",
    }
    data = data.assign_coords(
        Receptor=[translate.get(r, r) for r in data.Receptor.values]
    )
    data = prepare_data(data)
    subjects = (
        load_file("meta-subjects")
        .rename(columns={"subject": "Sample"})
        .set_index("Sample")
    )

    # fit
    rcps = ["IgG1", "IgG1f", "IgG3", "IgG3f"]
    opts = assemble_options(data, rcps=rcps)
    params, _ = optimize_loss(data, **opts, return_reshaped_params=True)

    df = Rtot_to_df(params["Rtot"], data, rcps)

    # merge
    samples = df.index.get_level_values("Sample")
    df["class"] = subjects.loc[samples]["class.etuv"].values

    df["f"] = (df["IgG1f"] + df["IgG3f"]) / (
        df["IgG1"] + df["IgG1f"] + df["IgG3"] + df["IgG3f"]
    )
    df = df.reset_index()

    plot = sns.catplot(
        data=df,
        x="Antigen",
        y="f",
        hue="class",
        kind="box",
        dodge=True,
        height=5,
        aspect=3,
        hue_order=["EC", "VC", "TP", "UP"],
    )
    plot.set_xticklabels(rotation=45)
    return plot.figure
