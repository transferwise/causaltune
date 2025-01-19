import copy
import logging
import math
from typing import Optional, Dict, Union, Any, List, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from econml.cate_interpreter import SingleTreeCateInterpreter  # noqa F401
from dowhy.causal_estimator import CausalEstimate
from dowhy import CausalModel

from causaltune.score.thompson import thompson_policy, extract_means_stds
from causaltune.thirdparty.causalml import metrics
from causaltune.score.erupt import ERUPT
from .bite import bite
from causaltune.utils import treatment_values, psw_joint_weights

import dcor

from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DummyEstimator:
    def __init__(self, cate_estimate: np.ndarray, effect_intervals: Optional[np.ndarray] = None):
        self.cate_estimate = cate_estimate
        self.effect_intervals = effect_intervals

    def const_marginal_effect(self, X):
        return self.cate_estimate


def supported_metrics(
    problem: str, multivalue: bool, scores_only: bool, constant_ptt: bool = False
) -> List[str]:
    if problem == "iv":
        metrics = ["energy_distance", "frobenius_norm", "codec"]
        if not scores_only:
            metrics.append("ate")
        return metrics
    elif problem == "backdoor":
        # print("backdoor")
        if multivalue:
            # TODO: support other metrics for the multivalue case
            return ["psw_energy_distance", "energy_distance"]  # TODO: add erupt
        else:
            metrics = [
                "erupt",
                "norm_erupt",
                # "greedy_erupt",  # regular erupt was made probabilistic, no need for a separate one
                "policy_risk",  # NEW
                "qini",
                "auc",
                # "r_scorer",
                "energy_distance",  # should only be used in iv problems
                "psw_energy_distance",
                "frobenius_norm",  # NEW
                "codec",  # NEW
                "bite",  # NEW
            ]
            if not scores_only:
                metrics.append("ate")
            return metrics


def metrics_to_minimize():
    return [
        "energy_distance",
        "psw_energy_distance",
        "codec",
        "frobenius_norm",
        "psw_frobenius_norm",
        "policy_risk",
    ]


class Scorer:
    def __init__(
        self,
        causal_model: CausalModel,
        propensity_model: Any,
        problem: str,
        multivalue: bool,
    ):
        """
        Contains scoring logic for CausalTune.

        Access methods and attributes via `CausalTune.scorer`.

        """
        logger.info("Initializing Scorer")
        self.problem = problem
        self.multivalue = multivalue
        self.causal_model = copy.deepcopy(causal_model)

        self.identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        if "Dummy" in propensity_model.__class__.__name__:
            self.constant_ptt = True
        else:
            self.constant_ptt = False

        if problem == "backdoor":
            print("Fitting a Propensity-Weighted scoring estimator " "to be used in scoring tasks")
            treatment_series = causal_model._data[causal_model._treatment[0]]
            # this will also fit self.propensity_model, which we'll also use in
            # self.erupt
            self.psw_estimator = self.causal_model.estimate_effect(
                self.identified_estimand,
                method_name="backdoor.causaltune.models.MultivaluePSW",
                control_value=0,
                treatment_value=treatment_values(treatment_series, 0),
                target_units="ate",  # condition used for CATE
                confidence_intervals=False,
                method_params={
                    "init_params": {"propensity_model": propensity_model},
                },
            ).estimator

            if not hasattr(self.psw_estimator, "estimator") or not hasattr(
                self.psw_estimator.estimator, "propensity_model"
            ):
                raise ValueError("Propensity model fitting failed. Please check the setup.")
            else:
                print("Propensity Model Fitted Successfully")

            treatment_name = self.psw_estimator._treatment_name
            if not isinstance(treatment_name, str):
                treatment_name = treatment_name[0]

            # No need to call self.erupt.fit()
            # as propensity model is already fitted
            # self.propensity_model = est.propensity_model
            self.erupt = ERUPT(
                treatment_name=treatment_name,
                propensity_model=self.psw_estimator.estimator.propensity_model,
                X_names=self.psw_estimator._effect_modifier_names
                + self.psw_estimator._observed_common_causes_names,
            )

    def inverse_propensity_score(self, df: pd.DataFrame, clip: float = 0.05) -> np.ndarray:
        """
        Calculate the inverse propensity score weights for the given dataframe.

        Args:
            df (pandas.DataFrame): input dataframe
            clip (float): clipping value for propensity scores
        """

        propensity_model = self.psw_estimator.estimator.propensity_model
        p = propensity_model.predict_proba(
            df[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
        )
        treatment = df[self.psw_estimator._treatment_name].values
        ex_ante_p = p[np.arange(p.shape[0]), treatment]

        psw = 1.0 / np.clip(ex_ante_p, clip, 1 - clip)

        return psw

    def ate(self, df: pd.DataFrame) -> tuple:
        """
        Calculate the Average Treatment Effect. Provide naive std estimates in
        single-treatment cases.

        Args:
            df (pandas.DataFrame): input dataframe

        Returns:
            tuple: tuple containing the ATE, standard deviation of the estimate
            (or None if multi-treatment), and sample size (or None if estimate
            has more than one dimension)
        """

        estimate = self.psw_estimator.estimator.effect(df).mean(axis=0)

        if len(estimate) == 1:
            # for now, let's cheat on the std estimation, take that from the
            # naive ate
            treatment_name = self.causal_model._treatment[0]
            outcome_name = self.causal_model._outcome[0]
            naive_est = Scorer.naive_ate(df[treatment_name], df[outcome_name])
            return estimate[0], naive_est[1], naive_est[2]
        else:
            return estimate, None, None

    def resolve_metric(self, metric: str) -> str:
        """Check if supplied metric is supported.
            If not, default to 'energy_distance'.

        Args:
            metric (str): evaluation metric

        Returns:
            str: metric/'energy_distance'

        """

        metrics = supported_metrics(self.problem, self.multivalue, scores_only=True)

        if metric not in metrics:
            logging.warning(
                f"Using energy_distance metric as {metric} is not in the list "
                f"of supported metrics for this usecase ({str(metrics)})"
            )
            return "energy_distance"
        else:
            return metric

    def resolve_reported_metrics(
        self, metrics_to_report: Union[List[str], None], scoring_metric: str
    ) -> List[str]:
        """
        Check if supplied reporting metrics are valid.

        Args:
            metrics_to_report (Union[List[str], None]): list of strings
            specifying the evaluation metrics to compute. Possible options
            include 'ate', 'erupt', 'norm_erupt', 'qini', 'auc',
            'energy_distance', and 'psw_energy_distance'.
            scoring_metric (str): specified metric.

        Returns:
            List[str]: list of valid metrics.
        """

        metrics = supported_metrics(self.problem, self.multivalue, scores_only=False)

        if metrics_to_report is None:
            return metrics
        else:
            metrics_to_report = sorted(list(set(metrics_to_report + [scoring_metric])))
            for m in metrics_to_report.copy():
                if m not in metrics:
                    logging.warning(
                        f"Dropping the metric {m} for problem: {self.problem} \
                        : must be one of {metrics}"
                    )
                    metrics_to_report.remove(m)
        return metrics_to_report

    @staticmethod
    def energy_distance_score(
        estimate: CausalEstimate,
        df: pd.DataFrame,
    ) -> float:
        """
        Calculate energy distance score between treated and controls. For
        theoretical details, see Ramos-Carreño and Torrecilla (2023).

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate
            to evaluate
            df (pandas.DataFrame): input dataframe

        Returns:
            float: energy distance score
        """

        Y0X, _, split_test_by = Scorer._Y0_X_potential_outcomes(estimate, df)

        YX_1 = Y0X[Y0X[split_test_by] == 1]
        YX_0 = Y0X[Y0X[split_test_by] == 0]
        select_cols = estimate.estimator._effect_modifier_names + ["yhat"]

        energy_distance_score = dcor.energy_distance(YX_1[select_cols], YX_0[select_cols])

        return energy_distance_score

    @staticmethod
    def _Y0_X_potential_outcomes(estimate: CausalEstimate, df: pd.DataFrame):
        est = estimate.estimator
        # assert est.identifier_method in ["iv", "backdoor"]
        treatment_name = (
            est._treatment_name if isinstance(est._treatment_name, str) else est._treatment_name[0]
        )
        df["dy"] = estimate.estimator.effect_tt(df)
        df["yhat"] = df[est._outcome_name] - df["dy"]

        split_test_by = (
            est.estimating_instrument_names[0] if est.identifier_method == "iv" else treatment_name
        )
        Y0X = copy.deepcopy(df)

        return Y0X, treatment_name, split_test_by

    # NEW:
    def frobenius_norm_score(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        sd_threshold: float = 1e-2,
        epsilon: float = 1e-5,
        alpha: float = 0.5,
    ) -> float:
        """
        Calculate adaptive Frobenius norm-based score between treated and controls.
        Automatically determines whether to use propensity score weighting based on:
        1. Problem type (IV vs backdoor)
        2. Data characteristics (presence of propensity modifiers/instruments)
        3. Estimator properties

        Args:
            estimate (CausalEstimate): causal estimate to evaluate
            df (pandas.DataFrame): input dataframe
            sd_threshold (float): threshold for standard deviation of CATE estimates
            epsilon (float): small regularization constant
            alpha (float): weight between Frobenius norm and variance component

        Returns:
            float: Frobenius norm-based score, with propensity weighting if applicable
        """
        # Get CATE estimates
        try:
            cate_estimates = estimate.estimator.effect(df)
        except AttributeError:
            try:
                cate_estimates = estimate.estimator.effect_tt(df)
            except AttributeError:
                return np.inf

        if np.std(cate_estimates) <= sd_threshold:
            return np.inf

        # Get data splits and check validity
        Y0X, treatment_name, split_test_by = self._Y0_X_potential_outcomes(estimate, df)

        Y0X_1 = Y0X[Y0X[split_test_by] == 1]
        Y0X_0 = Y0X[Y0X[split_test_by] == 0]

        if len(Y0X_1) == 0 or len(Y0X_0) == 0:
            return np.inf

        # Determine if propensity weighting should be used
        use_propensity = self._should_use_propensity(estimate)

        # Normalize features
        select_cols = estimate.estimator._effect_modifier_names + ["yhat"]
        scaler = StandardScaler()
        Y0X_0_normalized = scaler.fit_transform(Y0X_0[select_cols])
        Y0X_1_normalized = scaler.transform(Y0X_1[select_cols])

        # Calculate pairwise differences
        differences_xy = Y0X_1_normalized[:, np.newaxis, :] - Y0X_0_normalized[np.newaxis, :, :]

        if use_propensity:
            try:
                # Calculate and apply propensity weights
                propensitymodel = self.psw_estimator.estimator.propensity_model
                YX_1_all_psw = propensitymodel.predict_proba(
                    Y0X_1[
                        self.causal_model.get_effect_modifiers()
                        + self.causal_model.get_common_causes()
                    ]
                )
                treatment_series = Y0X_1[treatment_name]
                YX_1_psw = np.zeros(YX_1_all_psw.shape[0])
                for i in treatment_series.unique():
                    YX_1_psw[treatment_series == i] = YX_1_all_psw[:, i][treatment_series == i]

                YX_0_psw = propensitymodel.predict_proba(
                    Y0X_0[
                        self.causal_model.get_effect_modifiers()
                        + self.causal_model.get_common_causes()
                    ]
                )[:, 0]

                # Trim propensity scores
                YX_1_psw = np.clip(YX_1_psw, 0.01, 0.99)
                YX_0_psw = np.clip(YX_0_psw, 0.01, 0.99)

                # Calculate joint weights and apply them
                xy_psw = psw_joint_weights(YX_1_psw, YX_0_psw)
                xy_mean_weights = np.mean(xy_psw)
                weighted_differences_xy = np.reciprocal(xy_mean_weights) * np.multiply(
                    xy_psw[:, :, np.newaxis], differences_xy
                )
            except (AttributeError, KeyError):
                # Fallback to unweighted if propensity weighting fails
                weighted_differences_xy = differences_xy
        else:
            weighted_differences_xy = differences_xy

        # Compute Frobenius norm
        frobenius_norm = np.sqrt(np.sum(weighted_differences_xy**2))

        # Normalize
        n_1, n_0 = len(Y0X_1), len(Y0X_0)
        p = differences_xy.shape[-1]
        normalized_score = frobenius_norm / np.sqrt(n_1 * n_0 * p)

        # Add regularization and variance component
        cate_variance = np.var(cate_estimates)
        inverse_variance_component = 1 / (cate_variance + epsilon)

        composite_score = alpha * normalized_score + (1 - alpha) * inverse_variance_component

        return composite_score if np.isfinite(composite_score) else np.inf

    def _should_use_propensity(self, estimate: CausalEstimate) -> bool:
        """
        Determine if propensity score weighting should be used based on:
        1. Problem type
        2. Data characteristics
        3. Estimator properties

        Args:
            estimate (CausalEstimate): causal estimate being evaluated

        Returns:
            bool: True if propensity weighting should be used
        """
        # Don't use propensity for IV problems
        if self.problem == "iv":
            return False

        # Check if we have a backdoor problem with propensity modifiers
        if self.problem == "backdoor":
            data = self.causal_model
            has_propensity = (
                hasattr(data, "get_propensity_modifiers")
                and len(data.get_propensity_modifiers()) > 0
            )
            has_confounders = len(data.get_common_causes()) > 0

            # Use propensity if we have modifiers or confounders
            return has_propensity or has_confounders

        # Default to no propensity weighting
        return False

    def psw_energy_distance(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        normalise_features=False,
    ) -> float:
        """
        Calculate propensity score adjusted energy distance score between
        treated and controls.

        Features are normalized using the
        `sklearn.preprocessing.QuantileTransformer`.

        For theoretical details, see Ramos-Carreño and Torrecilla (2023).

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate
            to evaluate.
            df (pandas.DataFrame): input dataframe.
            normalise_features (bool): whether to normalize features with
            `QuantileTransformer`.

        Returns:
            float: propensity-score weighted energy distance score.
        """

        Y0X, treatment_name, split_test_by = Scorer._Y0_X_potential_outcomes(estimate, df)

        Y0X_1 = Y0X[Y0X[split_test_by] == 1]
        Y0X_0 = Y0X[Y0X[split_test_by] == 0]

        propensitymodel = self.psw_estimator.estimator.propensity_model
        YX_1_all_psw = propensitymodel.predict_proba(
            Y0X_1[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
        )
        treatment_series = Y0X_1[treatment_name]

        YX_1_psw = np.zeros(YX_1_all_psw.shape[0])
        for i in treatment_series.unique():
            YX_1_psw[treatment_series == i] = YX_1_all_psw[:, i][treatment_series == i]

        propensitymodel = self.psw_estimator.estimator.propensity_model
        YX_0_psw = propensitymodel.predict_proba(
            Y0X_0[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
        )[:, 0]

        select_cols = estimate.estimator._effect_modifier_names + ["yhat"]
        features = estimate.estimator._effect_modifier_names

        xy_psw = psw_joint_weights(YX_1_psw, YX_0_psw)
        xx_psw = psw_joint_weights(YX_0_psw)
        yy_psw = psw_joint_weights(YX_1_psw)

        xy_mean_weights = np.mean(xy_psw)
        xx_mean_weights = np.mean(xx_psw)
        yy_mean_weights = np.mean(yy_psw)

        if normalise_features:
            qt = QuantileTransformer(n_quantiles=200)
            X_quantiles = qt.fit_transform(Y0X[features])

            Y0X_transformed = pd.DataFrame(X_quantiles, columns=features, index=Y0X.index)
            Y0X_transformed.loc[:, ["yhat", split_test_by]] = Y0X[["yhat", split_test_by]]

            Y0X_1 = Y0X_transformed[Y0X_transformed[split_test_by] == 1]
            Y0X_0 = Y0X_transformed[Y0X_transformed[split_test_by] == 0]

        exponent = 1
        distance_xy = np.reciprocal(xy_mean_weights) * np.multiply(
            xy_psw,
            dcor.distances.pairwise_distances(
                Y0X_1[select_cols], Y0X_0[select_cols], exponent=exponent
            ),
        )
        distance_yy = np.reciprocal(yy_mean_weights) * np.multiply(
            yy_psw,
            dcor.distances.pairwise_distances(Y0X_1[select_cols], exponent=exponent),
        )
        distance_xx = np.reciprocal(xx_mean_weights) * np.multiply(
            xx_psw,
            dcor.distances.pairwise_distances(Y0X_0[select_cols], exponent=exponent),
        )
        psw_energy_distance = 2 * np.mean(distance_xy) - np.mean(distance_xx) - np.mean(distance_yy)
        return psw_energy_distance

    @staticmethod
    def default_policy(cate: np.ndarray) -> np.ndarray:
        """Default policy that assigns treatment if CATE > 0."""
        return (cate > 0).astype(int)

    def policy_risk_score(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        cate_estimate: np.ndarray,
        outcome_name: str,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        rct_indices: Optional[pd.Index] = None,
        sd_threshold: float = 1e-4,
        clip: float = 0.05,
    ) -> float:
        # Ensure cate_estimate is a 1D array
        cate_estimate = np.squeeze(cate_estimate)

        # Handle constant or near-constant CATE estimates
        if np.std(cate_estimate) <= sd_threshold:
            return np.inf  # Return infinity for constant estimates

        # Use default_policy if no policy is provided
        if policy is None:
            policy = self.default_policy

        # Apply the policy to get treatment assignments
        policy_treatment = policy(cate_estimate)

        # Calculate propensity scores
        if not hasattr(self.psw_estimator, "estimator") or not hasattr(
            self.psw_estimator.estimator, "propensity_model"
        ):
            raise ValueError("Propensity model fitting failed. Please check the setup.")

        propensity_scores = self.psw_estimator.estimator.propensity_model.predict_proba(
            df[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
        )
        if propensity_scores.ndim == 2:
            propensity_scores = propensity_scores[:, 1]
        propensity_scores = np.clip(propensity_scores, clip, 1 - clip)

        treatment_name = self.psw_estimator._treatment_name

        # Calculate weights
        weights = np.where(
            df[treatment_name] == 1, 1 / propensity_scores, 1 / (1 - propensity_scores)
        )

        # Prepare RCT subset
        rct_df = df.loc[rct_indices].copy() if rct_indices is not None else df.copy()
        rct_df["weight"] = weights
        rct_df["policy_treatment"] = policy_treatment

        # Compute policy value
        value_policy = (
            rct_df[outcome_name]
            * (rct_df[treatment_name] == 1)
            * (rct_df["policy_treatment"] == 1)
            * rct_df["weight"]
        ).sum() / rct_df["weight"].sum() * (rct_df["policy_treatment"] == 1).mean() + (
            rct_df[outcome_name]
            * (rct_df[treatment_name] == 0)
            * (rct_df["policy_treatment"] == 0)
            * rct_df["weight"]
        ).sum() / rct_df[
            "weight"
        ].sum() * (
            rct_df["policy_treatment"] == 0
        ).mean()

        # Compute naive policy value (treating everyone)
        naive_value = rct_df[outcome_name].mean()

        # Compute normalized policy risk
        policy_risk = max(0, (naive_value - value_policy) / abs(naive_value))

        return policy_risk

    @staticmethod
    def qini_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
        """
        Calculate the Qini score, defined as the area between the Qini curves
        of a model and random.

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate
            to evaluate
            df (pandas.DataFrame): input dataframe
            cate_estimate (np.ndarray): array with CATE estimates

        Returns:
            float: Qini score
        """

        est = estimate.estimator
        new_df = pd.DataFrame()
        new_df["y"] = df[est._outcome_name]
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        new_df["w"] = df[treatment_name]
        new_df["model"] = cate_estimate

        qini_score = metrics.qini_score(new_df)

        return qini_score["model"]

    # NEW
    @staticmethod
    def randomNN(ids):
        """
        Generate a list of random nearest neighbors.

        Parameters:
        ids (array-like): List of indices to sample from.

        Returns:
        numpy.ndarray: Array of sampled indices with
        no position i having x[i] == i.
        """

        m = len(ids)
        # Sample random integers from 0 to m-2, size m, with replacement
        x = np.random.choice(m - 1, m, replace=True)
        # Adjust x to ensure no position i has x[i] == i
        x = x + (x >= np.arange(m))
        return np.array(ids)[x]

    # NEW
    @staticmethod
    def estimateConditionalQ(Y, X, Z):
        """
        Estimate Q(Y, Z | X), the numerator of the measure of
        conditional dependence of Y on Z given X.

        Parameters:
        Y (array-like): Vector of responses (length n).
        X (array-like): Matrix of predictors (n by p).
        Z (array-like): Matrix of predictors (n by q).

        Returns:
        float: Estimation of Q(Y, Z | X).
        """

        # Ensure X and Z are numpy arrays
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Z, np.ndarray):
            Z = np.array(Z)

        # To turn Z from shape (n,) to (n,1)
        Z = Z.reshape(-1, 1)

        n = len(Y)
        W = np.hstack((X, Z))

        # Compute the nearest neighbor of X
        nn_X = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
        nn_dists_X, nn_indices_X = nn_X.kneighbors(X)
        nn_index_X = nn_indices_X[:, 1]

        # Handle repeated data
        repeat_data = np.where(nn_dists_X[:, 1] == 0)[0]
        df_X = pd.DataFrame({"id": repeat_data, "group": nn_indices_X[repeat_data, 0]})
        df_X["rnn"] = df_X.groupby("group")["id"].transform(Scorer.randomNN)
        nn_index_X[repeat_data] = df_X["rnn"].values

        # Nearest neighbors with ties
        ties = np.where(nn_dists_X[:, 1] == nn_dists_X[:, 2])[0]
        ties = np.setdiff1d(ties, repeat_data)

        if len(ties) > 0:

            def helper_ties(a):
                distances = distance.cdist(X[a].reshape(1, -1), np.delete(X, a, axis=0)).flatten()
                ids = np.where(distances == distances.min())[0]
                x = np.random.choice(ids)
                return x + (x >= a)

            nn_index_X[ties] = [helper_ties(a) for a in ties]

        # Compute the nearest neighbor of W
        nn_W = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(W)
        nn_dists_W, nn_indices_W = nn_W.kneighbors(W)
        nn_index_W = nn_indices_W[:, 1]

        repeat_data = np.where(nn_dists_W[:, 1] == 0)[0]
        df_W = pd.DataFrame({"id": repeat_data, "group": nn_indices_W[repeat_data, 0]})
        df_W["rnn"] = df_W.groupby("group")["id"].transform(Scorer.randomNN)
        nn_index_W[repeat_data] = df_W["rnn"].values

        # Nearest neighbors with ties
        ties = np.where(nn_dists_W[:, 1] == nn_dists_W[:, 2])[0]
        ties = np.setdiff1d(ties, repeat_data)

        if len(ties) > 0:
            nn_index_W[ties] = [helper_ties(a) for a in ties]

        # Estimate Q
        R_Y = np.argsort(np.argsort(Y))  # Rank Y with ties method 'max'
        Q_n = (
            np.sum(np.minimum(R_Y, R_Y[nn_index_W])) - np.sum(np.minimum(R_Y, R_Y[nn_index_X]))
        ) / (n**2)

        return Q_n

    # NEW
    @staticmethod
    def estimateConditionalS(Y, X):
        """
        Estimate S(Y, X), the denominator of the
        measure of dependence of Y on Z given X.

        Parameters:
        Y (array-like): Vector of responses (length n).
        X (array-like): Matrix of predictors (n by p).

        Returns:
        float: Estimation of S(Y, X).
        """

        # Ensure X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n = len(Y)

        # Compute the nearest neighbor of X
        nn_X = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
        nn_dists_X, nn_indices_X = nn_X.kneighbors(X)
        nn_index_X = nn_indices_X[:, 1]

        # Handle repeated data
        repeat_data = np.where(nn_dists_X[:, 1] == 0)[0]
        df_X = pd.DataFrame({"id": repeat_data, "group": nn_indices_X[repeat_data, 0]})
        df_X["rnn"] = df_X.groupby("group")["id"].transform(Scorer.randomNN)
        nn_index_X[repeat_data] = df_X["rnn"].values

        # Nearest neighbors with ties
        ties = np.where(nn_dists_X[:, 1] == nn_dists_X[:, 2])[0]
        ties = np.setdiff1d(ties, repeat_data)

        if len(ties) > 0:

            def helper_ties(a):
                distances = distance.cdist(X[a].reshape(1, -1), np.delete(X, a, axis=0)).flatten()
                ids = np.where(distances == distances.min())[0]
                x = np.random.choice(ids)
                return x + (x >= a)

            nn_index_X[ties] = [helper_ties(a) for a in ties]

        # Estimate S
        R_Y = np.argsort(np.argsort(Y))  # Rank Y with ties method 'max'
        S_n = np.sum(R_Y - np.minimum(R_Y, R_Y[nn_index_X])) / (n**2)

        return S_n

    # NEW
    @staticmethod
    def estimateConditionalT(Y, Z, X):
        """
        Estimate T(Y, Z | X), the measure of dependence of Y on Z given X.

        Parameters:
        Y (array-like): Vector of responses (length n).
        Z (array-like): Matrix of predictors (n by q).
        X (array-like): Matrix of predictors (n by p).

        Returns:
        float: Estimation of T(Y, Z | X).
        """

        S = Scorer.estimateConditionalS(Y, X)

        # Happens only if Y is constant
        if S == 0:
            return 1
        else:
            return Scorer.estimateConditionalQ(Y, X, Z) / S

    # NEW
    @staticmethod
    def codec(Y, Z, X=None, na_rm=True):
        """
        Estimate the conditional dependence coefficient (CODEC).

        The conditional dependence coefficient (CODEC) is a measure of the
        amount of conditional dependence between a random variable Y and a
        random vector Z given a random vector X, based on an i.i.d. sample of
        (Y, Z, X). The coefficient is asymptotically guaranteed to be between
        0 and 1.

        Parameters:
            Y (array-like): Vector of responses (length n).
            Z (array-like): Matrix of predictors (n by q).
            X (array-like, optional): Matrix of predictors (n by p). Default
            is None.
            na_rm (bool): If True, remove NAs.

        Returns:
            float: The conditional dependence coefficient (CODEC) of Y and Z
            given X. If X is None, this is just a measure of the dependence
            between Y and Z.

        References:
            Azadkia, M. and Chatterjee, S. (2019). A simple measure of
            conditional dependence. https://arxiv.org/pdf/1910.12327.pdf
        """

        if X is None:
            # Ensure inputs are in proper format
            if not isinstance(Y, np.ndarray):
                Y = np.array(Y)
            if not isinstance(Z, np.ndarray):
                Z = np.array(Z)
                # print(f"Shape of Z: {Z.shape}")
                # print(f"Z is: {Z}")

            if len(Y) != Z.shape[0]:
                raise ValueError("Number of rows of Y and Z should be equal.")
            if na_rm:
                # Remove NAs
                mask = np.isfinite(Y) & np.all(np.isfinite(Z), axis=1)
                Z = Z[mask]
                Y = Y[mask]

            n = len(Y)
            if n < 2:
                raise ValueError("Number of rows with no NAs should be greater than 1.")

            return Scorer.estimateConditionalQ(Y, Z, np.zeros((n, 0)))

        # Ensure inputs are in proper format
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Z, np.ndarray):
            Z = np.array(Z)
        if len(Y) != X.shape[0] or len(Y) != Z.shape[0] or X.shape[0] != Z.shape[0]:
            raise ValueError("Number of rows of Y, X, and Z should be equal.")

        n = len(Y)
        if n < 2:
            raise ValueError("Number of rows with no NAs should be greater than 1.")

        return Scorer.estimateConditionalT(Y, Z, X)

    # NEW
    @staticmethod
    def identify_confounders(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> list:
        """
        Identify confounders in a DataFrame.

        Args:
            df (pd.DataFrame): Input dataframe
            treatment_col (str): Name of the treatment column
            outcome_col (str): Name of the outcome column

        Returns:
            list: List of confounders' column names
        """

        confounders = [
            col for col in df.columns if col not in [treatment_col, outcome_col, "random", "index"]
        ]
        return confounders

    # NEW
    @staticmethod
    def codec_score(estimate: CausalEstimate, df: pd.DataFrame) -> float:
        """Calculate the CODEC score for the effect of treatment on y_factual.

        Args:
            estimate (CausalEstimate): causal estimate to evaluate
            df (pd.DataFrame): input dataframe

        Returns:
            float: CODEC score
        """
        est = estimate.estimator
        treatment_name = (
            est._treatment_name if isinstance(est._treatment_name, str) else est._treatment_name[0]
        )
        outcome_name = est._outcome_name
        confounders = Scorer.identify_confounders(df, treatment_name, outcome_name)

        ########
        cate_est = est.effect(df)
        standard_deviations = np.std(cate_est)

        df["dy"] = est.effect_tt(df)

        df["yhat"] = df[est._outcome_name] - df["dy"]

        # have to use corrected y, not y factual to get the estimators
        # contribution in
        Y = df["yhat"]
        Z = df[treatment_name]
        X = df[confounders]

        if standard_deviations < 0.01:
            return np.inf

        return abs(Scorer.codec(Y, Z, X))

    @staticmethod
    def auc_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
        """Calculate the area under the uplift curve.

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate
            to evaluate
            df (pandas.DataFrame): input dataframe
            cate_estimate (np.ndarray): array with cate estimates

        Returns:
            float: area under the uplift curve

        """
        print("running auuc_score")
        est = estimate.estimator
        new_df = pd.DataFrame()
        new_df["y"] = df[est._outcome_name]
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        new_df["w"] = df[treatment_name]
        new_df["model"] = cate_estimate

        auc_score = metrics.auuc_score(new_df)

        return auc_score["model"]

    @staticmethod
    def real_qini_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
        # TODO  To calculate the 'real' qini score for synthetic datasets, to
        # be done

        # est = estimate.estimator
        new_df = pd.DataFrame()

        # new_df['tau'] = [df['y_factual'] - df['y_cfactual']]
        new_df["model"] = cate_estimate

        qini_score = metrics.qini_score(new_df)

        return qini_score["model"]

    @staticmethod
    def r_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray, r_scorer
    ) -> float:
        """
        Calculate r_score.

        For details, refer to Nie and Wager (2017) and Schuler et al. (2018).
        Adapted from the EconML implementation.

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate
            to evaluate
            df (pandas.DataFrame): input dataframe
            cate_estimate (np.ndarray): array with CATE estimates
            r_scorer: callable object used to compute the R-score

        Returns:
            float: r_score
        """

        # TODO
        return r_scorer.score(cate_estimate)

    @staticmethod
    def naive_ate(treatment: pd.Series, outcome: pd.Series):
        """Calculate simple ATE.

        Args:
            treatment (pandas.Series): series of treatments
            outcome (pandas.Series): series of outcomes

        Returns:
            tuple: tuple of simple ATE, standard deviation, and sample size

        """

        treated = (treatment == 1).sum()

        mean_ = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
        std1 = outcome[treatment == 1].std() / (math.sqrt(treated) + 1e-3)
        std2 = outcome[treatment == 0].std() / (math.sqrt(len(outcome) - treated) + 1e-3)
        std_ = math.sqrt(std1 * std1 + std2 * std2)
        return (mean_, std_, len(treatment))

    def group_ate(self, df: pd.DataFrame, policy: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Compute the average treatment effect (ATE) for different groups
        specified by a policy.

        Args:
            df (pandas.DataFrame): input dataframe, should contain columns
            for the treatment, outcome, and policy.
            policy (Union[pd.DataFrame, np.ndarray]): policy column in df or
            an array of the policy values, used to group the data.

        Returns:
            pandas.DataFrame: ATE, std, and size per policy.
        """

        tmp = {"all": self.ate(df)}
        for p in sorted(list(policy.unique())):
            tmp[p] = self.ate(df[policy == p])

        tmp2 = [{"policy": str(p), "mean": m, "std": s, "count": c} for p, (m, s, c) in tmp.items()]

        return pd.DataFrame(tmp2)

    # NEW:
    def bite_score(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        N_values: Optional[List[int]] = None,
    ) -> float:
        """
        Calculate the BITE (Bins-induced Kendall's Tau Evaluation) score.

        Args:
            estimate (CausalEstimate): The causal estimate to evaluate.
            df (pd.DataFrame): The test dataframe.
            N_values (Optional[List[int]]): List of bin counts to evaluate.

        Returns:
            float: The BITE score. Higher values indicate better model performance.
        """

        est = estimate.estimator
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        outcome_name = est._outcome_name

        # Create a copy of df to avoid modifying original
        working_df = df.copy()

        # Estimated ITEs on test data
        cate_estimate = est.effect(df)
        if len(cate_estimate.shape) > 1 and cate_estimate.shape[1] == 1:
            cate_estimate = cate_estimate.reshape(-1)
        working_df["estimated_ITE"] = cate_estimate

        # Get propensity scores
        if hasattr(self.psw_estimator.estimator, "propensity_model"):
            propensity_model = self.psw_estimator.estimator.propensity_model
            working_df["propensity"] = propensity_model.predict_proba(
                df[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
            )[:, 1]
        else:
            raise ValueError("Propensity model is not available.")

        # Calculate the BITE score
        bite_score = bite(working_df, treatment_name, outcome_name)
        return bite_score

    def make_scores(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        metrics_to_report: List[str],
        r_scorer=None,
    ) -> dict:
        """
        Calculate various performance metrics for a given causal estimate using
        a given DataFrame.

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate
            to evaluate.
            df (pandas.DataFrame): input dataframe.
            metrics_to_report (List[str]): list of strings specifying the
            evaluation metrics to compute. Possible options include 'ate',
            'erupt', 'norm_erupt', 'qini', 'auc', 'energy_distance' and
            'psw_energy_distance'.
            r_scorer (Optional): callable object used to compute the R-score,
            default is None.

        Returns:
            dict: dictionary containing the evaluation metrics specified in
            metrics_to_report. The values key in the dictionary contains the
            input DataFrame with additional columns for the propensity scores,
            the policy, the normalized policy, and the weights, if applicable.
        """

        out = dict()
        df = df.copy().reset_index()

        est = estimate.estimator
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        outcome_name = est._outcome_name

        cate_estimate = est.effect(df)

        # TODO: fix this hack with proper treatment of multivalues
        if len(cate_estimate.shape) > 1 and cate_estimate.shape[1] == 1:
            cate_estimate = cate_estimate.reshape(-1)

        # TODO: fix this, currently broken
        # covariates = est._effect_modifier_names
        # Include CATE Interpereter for both IV and CATE models
        # intrp = SingleTreeCateInterpreter(
        #     include_model_uncertainty=False, max_depth=2, min_samples_leaf=10
        # )
        # intrp.interpret(DummyEstimator(cate_estimate), df[covariates])
        # intrp.feature_names = covariates
        # out["intrp"] = intrp

        if self.problem == "backdoor":
            values = df[[treatment_name, outcome_name]].copy()
            simple_ate = self.ate(df)[0]

            if isinstance(simple_ate, float):
                # simple_ate = simple_ate[0]
                # .reset_index(drop=True)
                propensitymodel = self.psw_estimator.estimator.propensity_model
                values["p"] = propensitymodel.predict_proba(
                    df[
                        self.causal_model.get_effect_modifiers()
                        + self.causal_model.get_common_causes()
                    ]
                )[:, 1]
                values["policy"] = cate_estimate > 0
                values["norm_policy"] = cate_estimate > simple_ate
                # values["weights"] = self.erupt.weights(df, lambda x: cate_estimate > 0)
            else:
                pass
                # TODO: what do we do here if multiple treatments?

            if "erupt" in metrics_to_report:
                # create standard deviations for thompson sampling
                # what is this for?
                if len(cate_estimate.shape) > 1 and cate_estimate.shape[1] == 1:
                    cate_estimate = cate_estimate.reshape(-1)

                # Get standard errors using established methods if available
                # TODO: pass num-treatments around cleanly
                num_treatments = df[treatment_name].nunique()
                # TODO: can I not get the values in one fell swoop?
                effect_means, effect_stds = extract_means_stds(
                    est, df, treatment_name, num_treatments
                )
                policy = thompson_policy(means=effect_means, stds=effect_stds)

                erupt_score = self.erupt.score(df, df[outcome_name], policy)
                out["erupt"] = erupt_score

            if "norm_erupt" in metrics_to_report:
                norm_erupt_score = (
                    self.erupt.score(df, df[outcome_name], cate_estimate > simple_ate)
                    - simple_ate * values["norm_policy"].mean()
                )
                out["norm_erupt"] = norm_erupt_score

            # if "frobenius_norm" in metrics_to_report:
            #     out["frobenius_norm"] = self.frobenius_norm_score(estimate, df)

            if "policy_risk" in metrics_to_report:
                out["policy_risk"] = self.policy_risk_score(
                    estimate=estimate,
                    df=df,
                    cate_estimate=cate_estimate,
                    outcome_name=outcome_name,
                    policy=None,
                )

            if "qini" in metrics_to_report:
                out["qini"] = Scorer.qini_make_score(estimate, df, cate_estimate)

            if "auc" in metrics_to_report:
                out["auc"] = Scorer.auc_make_score(estimate, df, cate_estimate)

            if "bite" in metrics_to_report:
                bite_score = self.bite_score(estimate, df)
                out["bite"] = bite_score

            if r_scorer is not None:
                out["r_score"] = Scorer.r_make_score(estimate, df, cate_estimate, r_scorer)

            # values = values.rename(columns={treatment_name: "treated"})
            assert len(values) == len(df), "Index weirdness when adding columns!"
            values = values.copy()
            out["values"] = values

        if "ate" in metrics_to_report:
            out["ate"] = cate_estimate.mean()
            out["ate_std"] = cate_estimate.std()

        if "energy_distance" in metrics_to_report:
            out["energy_distance"] = Scorer.energy_distance_score(estimate, df)

        if "psw_energy_distance" in metrics_to_report:
            out["psw_energy_distance"] = self.psw_energy_distance(
                estimate,
                df,
            )
        if "codec" in metrics_to_report:
            temp = self.codec_score(estimate, df)
            out["codec"] = temp

        if "frobenius_norm" in metrics_to_report:
            out["frobenius_norm"] = self.frobenius_norm_score(estimate, df)

        # if "psw_frobenius_norm" in metrics_to_report:
        #     out["psw_frobenius_norm"] = self.psw_frobenius_norm_score(estimate, df)

        del df
        return out

    @staticmethod
    def best_score_by_estimator(scores: Dict[str, dict], metric: str) -> Dict[str, dict]:
        """Obtain best score for each estimator.

        Args:
            scores (Dict[str, dict]): CausalTune.scores dictionary
            metric (str): metric of interest

        Returns:
            Dict[str, dict]: dictionary containing best score by estimator

        """

        for k, v in scores.items():
            if "estimator_name" not in v:
                raise ValueError(
                    f"Malformed scores dict, 'estimator_name' field missing " f"in{k}, {v}"
                )

        estimator_names = sorted(
            list(set([v["estimator_name"] for v in scores.values() if "estimator_name" in v]))
        )
        best = {}
        for name in estimator_names:
            est_scores = [
                v for v in scores.values() if "estimator_name" in v and v["estimator_name"] == name
            ]
            best[name] = (
                min(est_scores, key=lambda x: x[metric])
                if metric
                in [
                    "energy_distance",
                    "psw_energy_distance",
                    "frobenius_norm",
                    "codec",
                    "policy_risk",
                ]
                else max(est_scores, key=lambda x: x[metric])
            )

        return best
