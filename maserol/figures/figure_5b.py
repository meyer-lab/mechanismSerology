import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from tensordata.kaplonekVaccineSA import data as get_covid_vaccination_data

from maserol.core import optimize_loss
from maserol.preprocess import Rtot_to_df, assemble_options, prepare_data



def makeFigure():
    covid_vaccination_data = get_covid_vaccination_data()

    # prepare luminex
    lum_data = prepare_data(covid_vaccination_data["Luminex"])

    # prepare meta
    meta_data = covid_vaccination_data["Meta"].to_dataframe().reset_index(level="Metadata").pivot(columns="Metadata", values="Meta")
    meta_data.columns.name = None
    meta_data.index.name = "Sample"

    opts = assemble_options(lum_data)
    params, _ = optimize_loss(lum_data, **opts, return_reshaped_params=True)

    df = Rtot_to_df(params["Rtot"], lum_data, list(opts["rcps"])).reset_index(level="Antigen")
    df["fucose"] = (df["IgG1f"] + df["IgG3f"]) / (df["IgG1"] + df["IgG1f"] + df["IgG3"] + df["IgG3f"])
    df = df.merge(meta_data, on="Sample", how="inner")

    fig = plt.figure(figsize=(13, 5))
    ag_exclude = ["Ebola", "HKU1.Spike", "Influenza", "CMV", "OC43.Spike", "P1.RBD", "P1.S", "RSV", "CMV"]
    df_sub = df[~df["Antigen"].isin(ag_exclude)]
    df_sub = df_sub.sort_values("Antigen")[::-1]
    df_sub["infection.status"].replace({"case": "Positive", "control": "Negative"}, inplace=True)
    ax = sns.boxplot(data=df_sub, x="Antigen", y="fucose", hue="infection.status")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title="Infection Status")
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_xlabel("Antigen")
    pairs = [((ag, "Negative"), (ag, "Positive")) for ag in df_sub.Antigen.unique()]
    annotator = Annotator(ax, pairs, data=df_sub, x="Antigen", y="fucose", hue="infection.status")
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()
    return fig