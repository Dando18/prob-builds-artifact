""" Main script for evaluating methods for interpolation and extrapolation.
"""
# std imports
from argparse import ArgumentParser
import json
from typing import Iterable, Optional, Union
import os

# tpl imports
import numpy as np
from scipy.special import softmax
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# local imports
from models import AggregateRegressor, LookupRegressor, MeanRegressor, NearestPairRegressor, \
    RandomForestVersionRegressor, AdaBoostVersionRegressor, DenseRegressor, \
    GaussianProcessVersionRegressor, LinearVersionRegressor, XGBVersionRegressor, LinearWeightedMeansRegressor, \
    KNNWeightedMeansRegressor, XGBWeightedMeansRegressor, SimpleNearestPairRegressor
from plot import plot_bar_chart
from util import create_version_table, powerset

def get_args():
    parser = ArgumentParser(description="Evaluate interpolation and extrapolation methods.")
    parser.add_argument("-o", "--output-root", type=str, help="Root directory for results.")
    parser.add_argument("--no-plot", action="store_true", help="Don't plot results.")
    parser.add_argument("-d", "--data", type=str, default="data.csv", help="Path to probabilities data.")
    parser.add_argument("--drop", type=float, default=0.2, help="Percentage of pairs to drop.")
    parser.add_argument("--no-preserve-pairs", action="store_true", help="don't preserve all parent/child pairs when dropping")
    parser.add_argument("--cv", type=int, default=3, help="Number of cross validation folds.")
    parser.add_argument("--normalize", nargs="?", type=str, const="per-pair", 
        choices=["per-pair", "all", "per-parent", "per-child", "per-parent-version", "per-child-version"], 
        help="Normalize probabilities.")
    parser.add_argument("--normalization-method", type=str, default="maxabs", choices=["maxabs", "softmax", "zscore"], help="Normalization method.")
    parser.add_argument("--allow-non-roots", type=str, nargs="?", const="agg", choices=["mean", "median", "explode"], help="include not root pairs")
    parser.add_argument("--round", type=int, nargs="?", const=1, help="Round to n decimal places.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of jobs to run in parallel.")
    return parser.parse_args()

DEFAULT_METRICS = [
    "neg_mean_absolute_error", 
    "neg_mean_squared_error", 
    "r2", 
    "neg_mean_absolute_percentage_error",
    ]
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

TESTS = [
    {
        "name": "Constant",
        "model": DummyRegressor,
        "tune": {"strategy": ["constant"], "constant": np.arange(0, 1, 0.1)},
    },
    # {
    #     "name": "Global Mean",
    #     "model": DummyRegressor,
    #     "tune": {"strategy": ["mean", "median"]},
    # },
    {
        "name": "Per-Pair Mean",
        "model": MeanRegressor,
        "tune": {
            "agg_func": ["mean", "median"], 
            "grouping": list(filter(lambda x: len(x)>0, 
                    map(list, powerset(["parent", "child", "parent version", "child version"]))
                ))
            },
    },
    # {
    #     "name": "Simple Nearest Version",
    #     "model": SimpleNearestPairRegressor,
    #     "tune": {"p": [2]},
    # },
    #{
    #    "name": "Linear Weighted Means",
    #    "model": LinearWeightedMeansRegressor,
    #    "tune": {},
    #},
    # {
    #     "name": "KNN Weighted Means",
    #     "model": KNNWeightedMeansRegressor,
    #     "tune": {"n_neighbors": [1, 7, 13], "weights": ["uniform", "distance"]},
    # },
    # {
    #     "name": "XGB Weighted Means",
    #     "model": XGBWeightedMeansRegressor,
    #     "tune": {"n_estimators": list(range(50, 400, 50))},
    # },
    {
        "name": "Nearest Version",
        "model": NearestPairRegressor,
        "tune": {"no_ordinal": [False], "n_neighbors": [1, 13], "weights": ["uniform", "distance"]},
    },
    {
        "name": "Reg. Linear",
        "model": LinearVersionRegressor,
        "tune": {"no_ordinal": [False]},
    },
    # {
    #     "name": "Linear",
    #     "model": LinearVersionRegressor,
    #     "tune": {"no_ordinal": [True], "pca_dim": [None]},
    # },
    # {
    #     "name": "Nearest Version",
    #     "model": NearestPairRegressor,
    #     "tune": {"no_ordinal": [True], "n_neighbors": [1, 7], "weights": ["uniform", "distance"], "pca_dim": [None]},
    # },
    {
        "name": "Reg. AdaBoost",
        "model": AdaBoostVersionRegressor,
        "tune": {"no_ordinal": [False], "n_estimators": [50]},
    },
    # {
    #     "name": "AdaBoost",
    #     "model": AdaBoostVersionRegressor,
    #     "tune": {"no_ordinal": [True], "n_estimators": [50, 100], "pca_dim": [50, 100, 500]},
    # },
    # {
    #     "name": "Random Forest+Ordinal",
    #     "model": RandomForestVersionRegressor,
    #     "tune": {"no_ordinal": [False], "n_estimators": [50, 100]},
    # },
    {
        "name": "Reg. XGBoost",
        "model": XGBVersionRegressor,
        "tune": {"no_ordinal": [False], "n_estimators": [350]},
    },
    # {
    #     "name": "XGBoost",
    #     "model": XGBVersionRegressor,
    #     "tune": {"no_ordinal": [True], "n_estimators": [50, 100]},
    # }
]


def get_best_params(
    ModelClass, 
    params: dict, 
    X: pd.DataFrame, 
    y: pd.DataFrame,
    scoring: Iterable[str] = DEFAULT_METRICS,
    n_splits: int = 5, 
    version_table: Optional[dict] = None,
    preserve_pairs: bool = True, 
    test_frac: float = 0.2, 
    n_jobs: Optional[int] = None,
    random_state: Union[int, np.random.RandomState, np.random.Generator] = 42
):
    def splitter(data: pd.DataFrame, n_splits: int, preserve_pairs: bool):
        for _ in range(n_splits):
            if preserve_pairs:
                test = data.groupby(["parent", "child"]).sample(frac=test_frac, random_state=random_state)
                train = data.drop(test.index)
                yield train.index.values.astype(int), test.index.values.astype(int)
            else:
                train, test = train_test_split(data, test_size=test_frac, random_state=random_state)
                yield train.index.values.astype(int), test.index.values.astype(int)
    
    model = ModelClass()
    lookup = LookupRegressor()
    combined_model = AggregateRegressor(lookup, model)

    if issubclass(ModelClass, DenseRegressor) or issubclass(ModelClass, SimpleNearestPairRegressor):
        params["version_table"] = [version_table]

    search = GridSearchCV(combined_model, params, cv=splitter(X, n_splits, preserve_pairs), scoring=scoring, refit=False, n_jobs=n_jobs)
    search.fit(X, y)

    df = pd.DataFrame(search.cv_results_)
    results = {"model": ModelClass.__name__}
    for metric in scoring:
        best = df[df[f"rank_test_{metric}"] == 1].iloc[0]

        if issubclass(ModelClass, DenseRegressor) or issubclass(ModelClass, SimpleNearestPairRegressor):
            best["params"].pop("version_table", None)
        
        metric_results = {
            "mean_time": best["mean_fit_time"],
            "std_time": best["std_fit_time"],
            "params": best["params"],
            f"mean": best[f"mean_test_{metric}"],
            f"std": best[f"std_test_{metric}"],
        }
        results[metric] = metric_results
    return results


def main():
    args = get_args()

    # read in data set
    data = pd.read_csv(args.data)

    # non-roots
    if args.allow_non_roots == "median":
        data.drop(columns=["root"], inplace=True)
        data = data.groupby(["parent", "parent version", "child", "child version"]).median().reset_index()
    elif args.allow_non_roots == "mean":
        data.drop(columns=["root"], inplace=True)
        data = data.groupby(["parent", "parent version", "child", "child version"]).mean().reset_index()
    elif args.allow_non_roots == "explode":
        data["parent"] = data["root"] + "+" + data["parent"]
        data.drop(columns=["root"], inplace=True)
    else:
        data = data[data["parent"] == data["root"]].reset_index()

    # normalize
    if args.normalization_method == "softmax":
        norm_func = softmax
    elif args.normalization_method == "maxabs":
        # data is already in [0, inf) so no need to take absolute value
        assert data["expected improvement"].min() >= 0, "Data must be in [0, inf) to use maxabs normalization."
        norm_func = lambda x: x / x.max()
    elif args.normalization_method == "zscore":
        norm_func = lambda x: (x - x.mean()) / x.std()
    else:
        raise ValueError("Normalization method {} is not supported.".format(args.normalization_method))
    
    if args.normalize == "per-pair":
        data["expected improvement"] = data.groupby(["parent", "child"])["expected improvement"].transform(norm_func)
    elif args.normalize == "all":
        data["expected improvement"] = norm_func(data["expected improvement"])
    elif args.normalize == "per-parent":
        data["expected improvement"] = data.groupby(["parent"])["expected improvement"].transform(norm_func)
    elif args.normalize == "per-child":
        data["expected improvement"] = data.groupby(["child"])["expected improvement"].transform(norm_func)
    elif args.normalize == "per-parent-version":
        data["expected improvement"] = data.groupby(["parent", "parent version"])["expected improvement"].transform(norm_func)
    elif args.normalize == "per-child-version":
        data["expected improvement"] = data.groupby(["child", "child version"])["expected improvement"].transform(norm_func)
    elif args.normalize is not None:
        raise NotImplementedError(f"Normalization '{args.normalize}' not implemented.")

    # round
    if args.round is not None:
        data["expected improvement"] = data["expected improvement"].round(args.round)

    # create version table (some models need access to the full set of versions we'll be querying before training)
    version_table = create_version_table(data)

    # drop pairs
    X_COLUMNS = ["parent", "child", "parent version", "child version"]
    Y_COLUMNS = ["expected improvement"]
    X, y = data[X_COLUMNS], data[Y_COLUMNS]
    
    # conduct all experiments
    all_results = []
    for test in TESTS:
        print(f"Testing {test['model'].__name__} with args={test.get('kwargs', {})} and tune={test.get('tune', {})}:")
        random_state = np.random.default_rng(seed=args.seed)
        best = get_best_params(
            test["model"], 
            test.get("tune", {}), 
            X, y, 
            scoring=DEFAULT_METRICS,
            n_splits=args.cv,
            test_frac=args.drop,
            version_table=version_table,
            preserve_pairs=not args.no_preserve_pairs, 
            random_state=random_state,
            n_jobs=args.jobs,
        )
        best["name"] = test["name"]
        all_results.append(best)
        
        print(f"{best['model']}")
        for metric in DEFAULT_METRICS:
            metric_name = METRIC_NAMES[metric]
            mean = best[metric]["mean"] * (-1 if metric.startswith("neg") else best[metric]["mean"])
            print(f"\t{metric_name}: {mean:.6f} +/- {best[metric]['std']:3f} @ {best[metric]['params']}")

    if args.output_root:
        with open(os.path.join(args.output_root, "config.json"), "w") as fp:
            json.dump(vars(args), fp, indent=4)
        with open(os.path.join(args.output_root, "results.json"), "w") as fp:
            json.dump(all_results, fp, indent=4)

    # plot comparison of each model as a bar chart for each metric
    if not args.no_plot:
        for metric in DEFAULT_METRICS:
            mean_value = lambda x: x[metric]["mean"] * (-1 if metric.startswith("neg") else x[metric]["mean"])
            std_value = lambda x: x[metric]["std"]

            models = [x["name"] for x in all_results]
            mean_values = [mean_value(x) for x in all_results]
            std_values = [std_value(x) for x in all_results]
            metric_name = METRIC_NAMES[metric]
            df = pd.DataFrame({"model": models, metric_name: mean_values, "std": std_values})

            output = os.path.join(args.output_root, f"{metric_name}.png") if args.output_root else None
            plot_bar_chart(df, "model", metric_name, yerr=df["std"]/2,
                xlabel="", ylabel=metric_name.upper(), title=f"Version Extrapolation Method Comparison",
                label=True, output=output)



if __name__ == "__main__":
    main()