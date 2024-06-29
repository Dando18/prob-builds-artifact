""" Models for extrapolating build probabilities for unknown versions.
    notes:

    The model has to handle 4 distinct cases. Consider
    we are predicting the build probability of a parent/child
    pair P/C with parent version P_v and child version C_v.
    Assume we already have >=1 data points for builds of P/C.
    (1) We have an existing data point for P@P_v and C@C_v. 
        In this case we should return the existing value.
    (2) We have at least one data point for P@P_v, but none
        for C@C_v. In this case we need to extrapolate the
        build prediction value.
    (3) We don't have an existing data point for P@P_v, but we
        do have data points for C@C_v. In this case we need to
        extrapolate the build prediction value.
    (4) We don't have any data points for P@P_v or C@C_v. In
        this case we need to extrapolate the build prediction
        value.
    
    Cases (2-4) require extrapolation, however, they might
    be handled very differently. Also note that I ignore 
    interpolation for the time-being, since it seems unlikely
    that we would have data points for two versions, but not
    those between them. This, of course, could be studied
    separately. 

    There are a number of meaningful ways to extrapolate for
    unseen versions.
    * Constant value: return a constant value for all unseen
        versions.
    * Mean value: return the mean of known values. This
        could be the mean over child versions for a fixed
        parent version (2) or just the mean for all
        versions in a parent/child pair (3-4).
    * Nearest pair: return the value of the nearest known
        version. Version distance can be the distance
        between the concatenated version tuples for p/c.
    * Bayes: use Bayes and the priors to compute the 
        probability of P/C conditioned on previous 
        builds. This has the ability to make use of 
        information from builds using P or C in other
        parent/child pairs.
    * Gaussian Mixture: similar to above, but fit a 
        Gaussian mixture model to the data and use
        that to compute the probability of P/C.
    * Other ML model: fit any regression model to 
        encodings of P/C pairs and use them to
        extrapolate.
"""
# std imports
from collections import defaultdict
from distutils.version import LooseVersion, StrictVersion
import math
from typing import Optional

# tpl imports
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

# local imports
from util import filter_dict, encode_versions


class LookupRegressor(BaseEstimator, RegressorMixin):
    """ Lookup value in data set and if it exists return that value.
        Otherwise returns None (or a user-specified value).
    """

    def __init__(self, missing_value=None):
        self.missing_value = missing_value
    
    def fit(self, X, y=None):
        assert y is not None, "y must be provided."
        self.data = X.join(y, validate="1:1")
        self.output_columns = y.columns.tolist()
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        merge_df = pd.merge(X, self.data, how="left", on=X.columns.tolist())
        if self.missing_value:
            merge_df.fillna(value=self.missing_value, inplace=True)
        return merge_df[self.output_columns]


class MeanRegressor(BaseEstimator, RegressorMixin):
    """ Return the mean of the known values for a parent/child pair.
    """

    def __init__(self, agg_func: str = "mean", per_parent_version=False, grouping=["parent", "child"]):
        self.per_parent_version = per_parent_version
        self.agg_func = agg_func
        self.grouping = grouping

    def set_params(self, **params):
        self.agg_func = params.get("agg_func", self.agg_func)
        self.per_parent_version = params.get("per_parent_version", self.per_parent_version)
        self.grouping = params.get("grouping", self.grouping)
        return self

    def fit(self, X, y=None):
        assert y is not None, "y must be provided."

        data = X.join(y, validate="1:1")
        
        # compute mean for each group
        self.means = data.groupby(self.grouping)["expected improvement"].agg(self.agg_func)
        self.means = self.means.fillna(value=0.0)

        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        result = X.merge(self.means, how="left", left_on=self.grouping, right_index=True, validate="m:1")
        result.fillna(value=0.0, inplace=True)  # todo -- see why some values are nan
        return result["expected improvement"].to_numpy()
    

class WeightedMeansRegressor(BaseEstimator, RegressorMixin):
    """ Trains a tabular model on the mean, std, and percentiles of the known values for
        each parent, child, parent-child, parent-child-parent_version, and parent-child-child_version.
    """

    def __init__(self, BaseClass, **kwargs):

        # estimators based on the groupings you know
        self.estimators = {
            ("parent", "child", "parent-child"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "parent-parent_version"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "child-child_version"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "parent-parent_version", "child-child_version"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "parent-parent_version", "parent-child-parent_version"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "child-child_version", "parent-child-child_version"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "parent-parent_version", "child-child_version", "parent-child-parent_version"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "parent-parent_version", "child-child_version", "parent-child-child_version"): BaseClass(**kwargs),
            ("parent", "child", "parent-child", "parent-parent_version", "child-child_version", "parent-child-parent_version", "parent-child-child_version"): BaseClass(**kwargs),
        }

        self.grouping_columns = {
            "parent": ["parent"],
            "child": ["child"],
            "parent-child": ["parent", "child"],
            "parent-parent_version": ["parent", "parent version"],
            "child-child_version": ["child", "child version"],
            "parent-child-parent_version": ["parent", "child", "parent version"],
            "parent-child-child_version": ["parent", "child", "child version"]
        }
    
    def set_params(self, **params):
        for estimator in self.estimators.values():
            estimator.set_params(**params)
        return self

    def fit(self, X, y=None):
        assert y is not None, "y must be provided."
        data = X.join(y, validate="1:1")
        
        self.groupings = {}
        merged_dfs = X.copy()
        for name, group in self.grouping_columns.items():
            grouped_df = data.groupby(group)["expected improvement"].describe()

            if grouped_df["std"].isna().sum() > 0:  # describe::std returns NaN if group size is 1
                grouped_df["std"] = data.groupby(group)["expected improvement"].std(ddof=0)
        
            grouped_df.drop(columns=["count", "min", "max"], inplace=True)
            grouped_df = grouped_df.add_prefix(name + "_")
            self.groupings[name] = grouped_df

            merged_dfs = pd.merge(merged_dfs, grouped_df, how="left", right_index=True, left_on=group, validate="m:1")
        
        train_data = merged_dfs.drop(columns=["parent", "child", "parent version", "child version"])

        for key, estimator in self.estimators.items():
            columns = [f"{col}_{stat}" for col in key for stat in ["mean", "std", "25%", "50%", "75%"]]
            estimator.fit(train_data[columns], y.values.ravel())

        return self
    
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        out = X.copy()
        for name, group in self.grouping_columns.items():
            out = pd.merge(out, self.groupings[name], how="left", right_index=True, left_on=group, validate="m:1")

        out.drop(columns=["parent", "child", "parent version", "child version"], inplace=True)

        all_y = []
        for key, estimator in self.estimators.items():
            columns = [f"{col}_{stat}" for col in key for stat in ["mean", "std", "25%", "50%", "75%"]]
            inv_columns = [c for c in out.columns if c not in columns]

            # get rows where columns are not NaN and the rest are not
            df = out[out[columns].notna().all(axis=1) & out[inv_columns].isna().all(axis=1)]
            y = estimator.predict(df[columns]) if len(df) > 0 else np.array([])
            y = pd.DataFrame(index=df.index, data=y, columns=["expected improvement"])
            all_y.append(y)
        
        assert sum(len(a) for a in all_y) == len(X), "Not all rows were predicted."
        result = pd.concat(all_y).reindex(X.index)
        return result["expected improvement"].to_numpy()
    

class LinearWeightedMeansRegressor(WeightedMeansRegressor):
    def __init__(self, **kwargs):
        super().__init__(LinearRegression, **kwargs)

class KNNWeightedMeansRegressor(WeightedMeansRegressor):
    def __init__(self, **kwargs):
        super().__init__(KNeighborsRegressor, **kwargs)

class XGBWeightedMeansRegressor(WeightedMeansRegressor):
    def __init__(self, **kwargs):
        from xgboost import XGBRegressor
        super().__init__(XGBRegressor, **kwargs)


class SimpleNearestPairRegressor(BaseEstimator, RegressorMixin):
    """ Find the nearest parent-child pair and return it's expected improvement.
    """
    
    def __init__(self, p: int = 2, ordinal_offset: int = 0, version_table : Optional[dict] = None):
        self.p = p
        self.ordinal_offset = ordinal_offset
        self.version_table = version_table
    
    def fit(self, X, y=None, known_versions: Optional[dict] = None):
        assert y is not None, "y must be provided."
        if known_versions is None:
            assert self.version_table is not None, "version_table must be provided if known_versions is None."
        data = X.join(y, validate="1:1")

        # get all known versions for a package
        if self.version_table is None:
            self.version_table = defaultdict(set) # mapping from package name to set of versions
            for (parent, child), parent_versions in known_versions.items():
                for parent_version, child_versions in parent_versions.items():
                    self.version_table[parent].add(parent_version)
                    self.version_table[child].update(child_versions)

        # build a dict that maps (package, version) to its encoding
        self.version_encodings = {}
        for package, versions in self.version_table.items():
            encoded_versions = encode_versions(versions, offset=self.ordinal_offset)
            for version, encoding in encoded_versions.items():
                self.version_encodings[(package, version)] = encoding

        # build a dict keyed on (parent, child) where each value is another dict
        # that maps (parent version, child version) to expected improvement
        self.expected_improvements = defaultdict(dict)
        for idx, parent, child, parent_version, child_version, exp_imp in data.itertuples():
            parent_encoding = self.version_encodings[(parent, parent_version)]
            child_encoding = self.version_encodings[(child, child_version)]
            self.expected_improvements[(parent, child)][(parent_encoding, child_encoding)] = exp_imp

        return self

    def _get_nearest_pair(self, parent, child, parent_encoding, child_encoding):
        nearest_exp_imp = np.inf
        nearest_pair_encodings = None

        fixed_point = np.array([parent_encoding, child_encoding])
        for (parent_enc, child_enc), exp_imp in self.expected_improvements[(parent, child)].items():
            # calculate distance between parent and child encodings
            point = np.array([parent_enc, child_enc])
            dist = np.linalg.norm(point - fixed_point, ord=self.p)
            if dist < nearest_exp_imp:
                nearest_exp_imp = dist
                nearest_pair_encodings = (parent_enc, child_enc)
        
        return nearest_pair_encodings, nearest_exp_imp

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        result = []
        for idx, parent, child, parent_version, child_version in X.itertuples():
            parent_encoding = self.version_encodings[(parent, parent_version)]
            child_encoding = self.version_encodings[(child, child_version)]

            if (parent_encoding, child_encoding) in self.expected_improvements[(parent, child)]:
                # return expected improvement if we already have it
                nearest_exp_imp = self.expected_improvements[(parent, child)][(parent_encoding, child_encoding)]
            else:
                # otherwise get nearest parent-child pair
                _, nearest_exp_imp = self._get_nearest_pair(parent, child, parent_encoding, child_encoding)

            result.append(nearest_exp_imp)            

        return np.array(result)
    

class DenseRegressor(BaseEstimator, RegressorMixin):
    """ Map data to a dense representation and then use a standard regressor
        to predict the expected improvement.
        The dense representation is feature columns parent_p and child_p for 
        every package p. For a pair p@v1 and c@v2, the columns parent_p and 
        child_c will be set to ordinal encodings for their respective package
        versions. All other features are set to `fill_na` (default -100.0).

        This dense format is used to train a regressor to predict the expected
        improvement.
    """

    def __init__(self, 
        RgrClass, 
        pca_dim: Optional[int] = None,
        no_ordinal: bool = False,
        version_table : Optional[dict] = None,
        na_val: float = -100.0, 
        ordinal_offset: int = 1, 
        **kwargs
    ):
        self.na_val = na_val
        self.no_ordinal = no_ordinal
        self.ordinal_offset = ordinal_offset
        self.version_table = version_table
        self.pca_dim = pca_dim
        self.pca = None
        self.regressor_ = RgrClass(**kwargs)

    def set_params(self, **params):
        if "pca_dim" in params:
            self.pca_dim = params.pop("pca_dim")
        if "no_ordinal" in params:
            self.no_ordinal = params.pop("no_ordinal")
        if "version_table" in params:
            self.version_table = params.pop("version_table")
        if "ordinal_offset" in params:
            self.ordinal_offset = params.pop("ordinal_offset")
        if "na_val" in params:
            self.na_val = params.pop("na_val")
        self.regressor_.set_params(**params)
        return self

    def _compute_version_table(self, known_versions: dict):
        """ Compute an encoding for each version of each package.
            This encoding will be used to compute the distance
            between versions.
            Each version is encoded as an integer, where the
            integer is the index of the version in the sorted
            list of versions for that package.
        """
        if self.version_table is None:
            self.version_table = defaultdict(set) # mapping from package name to set of versions
            for (parent, child), parent_versions in known_versions.items():
                for parent_version, child_versions in parent_versions.items():
                    self.version_table[parent].add(parent_version)
                    self.version_table[child].update(child_versions)
        
        self.version_encodings = {} # mapping from (package, version) to encoding
        for package, versions in self.version_table.items():
            encoded_versions = encode_versions(versions, offset=self.ordinal_offset)
            for version, encoding in encoded_versions.items():
                self.version_encodings[(package, version)] = encoding


    def fit(self, X, y=None, known_versions: Optional[dict] = None):
        assert y is not None, "y must be provided."
        assert known_versions is not None, "known_versions must be provided."
        self._compute_version_table(known_versions)

        data = X.copy()
        if self.no_ordinal:
            all_pairs = [f"{p}@{v}" for p, v in self.version_encodings.keys()]
            columns = [f"parent_{pair}" for pair in all_pairs] + [f"child_{pair}" for pair in all_pairs]
            empty_df = pd.DataFrame(data=0.0, index=data.index,columns=columns)

            data["parent"] = data["parent"] + "@" + data["parent version"]
            data["child"] = data["child"] + "@" + data["child version"]
            data.drop(columns=["parent version", "child version"], inplace=True)
            data = pd.get_dummies(data, dtype=float)
            missing_cols = list(set(empty_df.columns) - set(data.columns))
            data = pd.concat([data, empty_df[missing_cols]], axis=1)
        else:
            data["parent version"] = data[["parent", "parent version"]].apply(lambda x: self.version_encodings[(x["parent"], x["parent version"])], axis=1)
            data["child version"] = data[["child", "child version"]].apply(lambda x: self.version_encodings[(x["child"], x["child version"])], axis=1)

            parent_pivot = data[["parent", "parent version"]].pivot(columns="parent", values="parent version").fillna(value=self.na_val)
            child_pivot = data[["child", "child version"]].pivot(columns="child", values="child version").fillna(value=self.na_val)
            parent_pivot = parent_pivot.add_prefix("parent_")
            child_pivot = child_pivot.add_prefix("child_")
            data = pd.concat([parent_pivot, child_pivot], axis=1)

        self.sample_columns_ = data.columns.tolist()

        if self.pca_dim is not None and self.pca_dim < data.shape[1]:
            self.pca = PCA(n_components=self.pca_dim)
            data = self.pca.fit_transform(data)
        
        self.regressor_.fit(data, y.values.ravel())
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        parents, children = X.parent.values, X.child.values
        parent_versions, child_versions = X["parent version"].values, X["child version"].values

        fill_val = 0.0 if self.no_ordinal else self.na_val
        out = pd.DataFrame(data=fill_val, index=np.arange(X.shape[0]), columns=self.sample_columns_)
        for idx, (parent, child, parent_version, child_version) in enumerate(zip(parents, children, parent_versions, child_versions)):
            if self.no_ordinal:
                out.loc[idx, f"parent_{parent}@{parent_version}"] = 1.0
                out.loc[idx, f"child_{child}@{child_version}"] = 1.0
            else:
                out.loc[idx, f"parent_{parent}"] = self.version_encodings[(parent, parent_version)]
                out.loc[idx, f"child_{child}"] = self.version_encodings[(child, child_version)]
        
        if self.pca:
            out = self.pca.transform(out)
        
        return self.regressor_.predict(out)


class NearestPairRegressor(DenseRegressor):
    def __init__(self, **kwargs):
        super().__init__(KNeighborsRegressor, **kwargs)

class RandomForestVersionRegressor(DenseRegressor):
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor, **kwargs)

class AdaBoostVersionRegressor(DenseRegressor):
    def __init__(self, **kwargs):
        super().__init__(AdaBoostRegressor, **kwargs)

class LinearVersionRegressor(DenseRegressor):
    def __init__(self, **kwargs):
        super().__init__(LinearRegression, **kwargs)

class GaussianProcessVersionRegressor(DenseRegressor):
    def __init__(self, **kwargs):
        super().__init__(GaussianProcessRegressor, **kwargs)

class XGBVersionRegressor(DenseRegressor):
    def __init__(self, **kwargs):
        from xgboost import XGBRegressor
        super().__init__(XGBRegressor, **kwargs)


class AggregateRegressor(BaseEstimator, RegressorMixin):
    """ Combine a base model and extrapolater into a single model.
        The base model is used to predict the expected improvement
        for known parent/child pairs. The extrapolater is used to
        predict the expected improvement for unknown parent/child
        pairs.
    """

    def __init__(self, base: BaseEstimator, extrapolater: BaseEstimator, include_branches: bool = True):
        self.base = base
        self.extrapolater = extrapolater
        self.include_branches = include_branches

    def fit(self, X, y=None):
        self.data = X
        self._compute_known_versions()
        self.base.fit(X, y)

        extrapolater_kwargs = {"known_versions": self.known_versions}
        extrapolater_kwargs = filter_dict(extrapolater_kwargs, self.extrapolater.fit)
        self.extrapolater.fit(X, y, **extrapolater_kwargs)
        return self

    def _compute_known_versions(self):
        """ Compute the known versions for each parent/child pair. Stored
            in self.known_versions as a dict of dicts. The outer dict is
            keyed by (parent, child) and the inner dict is keyed by parent
            version and contains a set of child versions.
            For example:
            {
                ("parent1", "child1"): {
                    "parent1_v1": {"child1_v1", "child1_v2"},
                    "parent1_v2": {"child1_v1", "child1_v2", "child1_v3"}
                },
                ("parent2", "child2"): {
                    "parent2_v1": {"child2_v1", "child2_v2"},
                    "parent2_v2": {"child2_v1", "child2_v2", "child2_v3"}
                }
            }
        """
        versions = self.data.groupby(["parent", "parent version", "child"]).agg(set)
        versions = versions.to_dict(orient="index")
        self.known_versions = {}
        for (parent, parent_version, child), child_versions in versions.items():
            key = (parent, child)
            if key not in self.known_versions:
                self.known_versions[key] = {}

            self.known_versions[key][parent_version] = child_versions["child version"]
            if not self.include_branches:
                for branch_name in ["develop", "main", "master"]:
                    self.known_versions[key][parent_version].discard(branch_name)

    def is_parent_version_known(self, parent: str, parent_version: str, child: str) -> bool:
        """ Do we have any examples of this parent version for this parent/child pair """
        key = (parent, child)
        return parent_version in self.known_versions.get(key, {})
    
    def is_child_version_known(self, parent: str, child: str, child_version: str) -> bool:
        """ Do we have any examples of this child version for this parent/child pair """
        key = (parent, child)
        return any(child_version in cvs  for vals in self.known_versions.get(key, {}).values() for cvs in vals)

    def _predict_unknown(self, x):
        # do nothing if we already have a value
        expected_improvement = x["expected improvement"]
        if not math.isnan(expected_improvement):
            return expected_improvement
        
        # if this parent/child pair is not present in the training data, return None
        # todo -- in the future we should provide a model for when the parent and/or child
        #         are present in the training data, but not the pair
        if (x["parent"], x["child"]) not in self.known_versions:
            return None
        
        row_df = pd.DataFrame(x).transpose()
        return self.extrapolater.predict(row_df)[0]

    def set_params(self, **params):
        """ only pass params to extrapolater """
        self.extrapolater.set_params(**params)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # get the predictions for known samples
        base_preds = self.base.predict(X)

        # get the predictions for unknown samples; apply _predict_unknown to each row
        extrap_preds = self.extrapolater.predict(X)
        extrap_preds = pd.DataFrame(data=extrap_preds, index=base_preds.index, columns=["expected improvement"])
        preds = base_preds.fillna(extrap_preds)
        
        preds.index = X.index

        return preds["expected improvement"]


