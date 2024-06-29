from argparse import ArgumentParser
from glob import glob
import json
import os
from typing import Iterable, Optional

import pandas as pd

from plot import plot_bar_chart

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--results-root", type=str, required=True, help="root of results")
    parser.add_argument("--output-root", type=str, help="root of place to save figures")
    return parser.parse_args()


METRIC_NAMES = {
        "neg_mean_absolute_error": "mae", 
        "neg_mean_squared_error": "mse", 
        "neg_median_absolute_error": "medae", 
        "r2": "r2", 
        "neg_mean_absolute_percentage_error": "mape",
    }

METRIC_LIMS = {
    "mae": (0, None),
    "mse": (0, None),
    "r2": (0, 1),
    "mape": (0, None),
}


def compare_norms(data, metric: str, exclude: Optional[Iterable[str]] = None, output: Optional[str] = None):
    metric_name = METRIC_NAMES[metric]
    metric_data = {"model": [], metric_name: [], "std": [], "norm": []}
    for config, result in data:
        norm_method = config["normalize"]
        if norm_method:
            norm_str = "-".join(map(str.capitalize, norm_method.split("-")))
        else:
            norm_str = "None"

        for model in result:
            mean_value = lambda x: x[metric]["mean"] * (-1 if metric.startswith("neg") else x[metric]["mean"])
            std_value = lambda x: x[metric]["std"]

            metric_data["model"].append(model["name"])
            metric_data[metric_name].append(mean_value(model))
            metric_data["std"].append(std_value(model))
            metric_data["norm"].append(norm_str)

    df = pd.DataFrame(metric_data)
    if exclude:
        df = df[~df.norm.isin(exclude)]
    ylim = METRIC_LIMS[metric_name]
    plot_bar_chart(df, "model", metric_name, hue="norm",
        xlabel="Model", ylabel=metric_name.upper(), ylim=ylim,
        title=f"{metric_name.upper()} Comparison Across Models and Normalization Methods",
        label=False, figsize=(22,8), output=output)


def main():
    args = get_args()

    # read in input
    data = []
    for subdir in glob(os.path.join(args.results_root, "*")):
        with open(os.path.join(subdir, "config.json"), "r") as fp:
            config = json.load(fp)
        with open(os.path.join(subdir, "results.json"), "r") as fp:
            results = json.load(fp)

        data.append((config, results))

    # plot a bar charts comparing each method 
    for config, results in data:
        norm_method = config["normalize"]
        if norm_method:
            norm_str = "-".join(map(str.capitalize, norm_method.split("-")))
        else:
            norm_str = "None"
        
        metrics = set(k for model in results for k in model if k not in ["model", "name"])
        for metric in metrics:
            mean_value = lambda x: x[metric]["mean"] * (-1 if metric.startswith("neg") else x[metric]["mean"])
            std_value = lambda x: x[metric]["std"]

            models = [x["name"] for x in results]
            mean_values = [mean_value(x) for x in results]
            std_values = [std_value(x) for x in results]
            metric_name = METRIC_NAMES[metric]
            df = pd.DataFrame({"model": models, metric_name: mean_values, "std": std_values})

            ylim = METRIC_LIMS[metric_name]

            output = os.path.join(args.output_root, f"norm_{norm_str}_metric_{metric_name}.png")
            plot_bar_chart(df, "model", metric_name, yerr=df["std"]/2,
                xlabel="Model", ylabel=metric_name.upper(), ylim=ylim,
                title=f"{metric_name.upper()} Comparison with Data Normalization={norm_str}",
                label=True, figsize=(20,8), output=output)


    # fix the metric to mape and mae and plot across normalization schemes
    compare_norms(data, "neg_mean_absolute_percentage_error", ["None", "Per-Child"], output=os.path.join(args.output_root, "all-norms_mape.png"))
    compare_norms(data, "neg_mean_absolute_error", ["None"], output=os.path.join(args.output_root, "all-norms_mae.png"))


if __name__ == '__main__':
    main()