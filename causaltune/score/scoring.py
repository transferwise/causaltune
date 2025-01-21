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
from causaltune.score.codec import codec_score
from causaltune.utils import treatment_values, psw_joint_weights

import dcor

from scipy.stats import kendalltau

from sklearn.preprocessing import StandardScaler


class DummyEstimator:
    def __init__(self, cate_estimate: np.ndarray, effect_intervals: Optional[np.ndarray] = None):
        self.cate_estimate = cate_estimate
        self.effect_intervals = effect_intervals

    def const_marginal_effect(self, X):
        return self.cate_estimate


def supported_metrics(problem: str, multivalue: bool, scores_only: bool, constant_ptt: bool = False) -> List[str]:
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
                X_names=self.psw_estimator._effect_modifier_names + self.psw_estimator._observed_common_causes_names,
            )

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

    def resolve_reported_metrics(self, metrics_to_report: Union[List[str], None], scoring_metric: str) -> List[str]:
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
        treatment_name = est._treatment_name if isinstance(est._treatment_name, str) else est._treatment_name[0]
        df["dy"] = estimate.estimator.effect_tt(df)
        df["yhat"] = df[est._outcome_name] - df["dy"]

        split_test_by = est.estimating_instrument_names[0] if est.identifier_method == "iv" else treatment_name
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
        Y0X_1_normalized = scaler.fit_transform(Y0X_1[select_cols])
        Y0X_0_normalized = scaler.transform(Y0X_0[select_cols])

        # Calculate pairwise differences
        differences_xy = Y0X_1_normalized[:, np.newaxis, :] - Y0X_0_normalized[np.newaxis, :, :]

        if use_propensity:
            try:
                # Calculate and apply propensity weights
                propensitymodel = self.psw_estimator.estimator.propensity_model
                YX_1_all_psw = propensitymodel.predict_proba(
                    Y0X_1[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
                )
                treatment_series = Y0X_1[treatment_name]
                YX_1_psw = np.zeros(YX_1_all_psw.shape[0])
                for i in treatment_series.unique():
                    YX_1_psw[treatment_series == i] = YX_1_all_psw[:, i][treatment_series == i]

                YX_0_psw = propensitymodel.predict_proba(
                    Y0X_0[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
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
            has_propensity = hasattr(data, "get_propensity_modifiers") and len(data.get_propensity_modifiers()) > 0
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
            dcor.distances.pairwise_distances(Y0X_1[select_cols], Y0X_0[select_cols], exponent=exponent),
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

    def default_policy(self, cate: np.ndarray) -> np.ndarray:
        return (cate > 0).astype(int)

    def policy_risk_score(
        self,
        estimate,
        df: pd.DataFrame,
        cate_estimate: np.ndarray,
        outcome_name: str,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        rct_indices: Optional[pd.Index] = None,
        sd_threshold: float = 1e-4,
        clip: float = 0.05,
    ) -> float:
        """
        Computes a 'policy risk' in the sense of:
            PolicyRisk = 1 - [ IPW average outcome under the policy ].
        This assumes your outcome is scaled to [0,1].

        If your outcome is not in [0,1], you may want to transform it or use
        a different final risk formula.
        """
        # Ensure cate_estimate is a 1D array
        cate_estimate = np.squeeze(cate_estimate)

        # Handle constant or near-constant CATE estimates
        if np.std(cate_estimate) <= sd_threshold:
            return np.inf  # Return infinity for constant estimates

        # Use default_policy if no policy is provided
        if policy is None:
            policy = self.default_policy

        # Apply the policy to get recommended treatment (pi_i)
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
        weights = np.where(df[treatment_name] == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))

        # Restrict to RCT subset if provided
        rct_df = df.loc[rct_indices].copy() if rct_indices is not None else df.copy()
        rct_df["weight"] = weights
        rct_df["policy_treatment"] = policy_treatment

        # -- 3) Compute the standard IPW estimate of the policy's value --
        #   V_hat(pi) = (1/N) * sum_{i=1 to N} [ I(T_i = pi_i) * Y_i * weight_i ]
        #   where N = number of rows in rct_df
        #   and I(T_i = pi_i) is 1 if observed treatment matches the policy's recommendation.

        N = len(rct_df)
        if N == 0:
            # Avoid zero-division if the RCT subset is empty
            return np.inf

        # Indicator that the actual treatment matches the policy
        treat_matches_policy = (rct_df[treatment_name] == rct_df["policy_treatment"]).astype(float)

        # Weighted sum
        weighted_sum = (treat_matches_policy * rct_df[outcome_name] * rct_df["weight"]).sum()

        # Average over N
        policy_value_ipw = weighted_sum / N  # This is our V_hat(pi).

        # -- 4) Compute the "policy risk" as 1 - policy_value_ipw

        policy_risk = 1.0 - policy_value_ipw

        return policy_risk

    @staticmethod
    def qini_make_score(estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray) -> float:
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

    @staticmethod
    def codec_score(estimate: CausalEstimate, df: pd.DataFrame) -> float:
        """Calculate the CODEC score for the effect of treatment on y_factual.

        Args:
            estimate (CausalEstimate): causal estimate to evaluate
            df (pd.DataFrame): input dataframe

        Returns:
            float: CODEC score
        """
        return codec_score(estimate, df)

    @staticmethod
    def auc_make_score(estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray) -> float:
        """Calculate the area under the uplift curve.

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate
            to evaluate
            df (pandas.DataFrame): input dataframe
            cate_estimate (np.ndarray): array with cate estimates

        Returns:
            float: area under the uplift curve

        """

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
    def real_qini_make_score(estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray) -> float:
        # TODO  To calculate the 'real' qini score for synthetic datasets, to
        # be done

        # est = estimate.estimator
        new_df = pd.DataFrame()

        # new_df['tau'] = [df['y_factual'] - df['y_cfactual']]
        new_df["model"] = cate_estimate

        qini_score = metrics.qini_score(new_df)

        return qini_score["model"]

    @staticmethod
    def r_make_score(estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray, r_scorer) -> float:
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
        if N_values is None:
            N_values = list(range(10, 21)) + list(range(25, 51, 5)) + list(range(60, 101, 10))

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

        # Calculate weights with clipping to avoid extremes
        working_df["weights"] = np.where(
            working_df[treatment_name] == 1,
            1 / np.clip(working_df["propensity"], 0.05, 0.95),
            1 / np.clip(1 - working_df["propensity"], 0.05, 0.95),
        )

        kendall_tau_values = []

        def compute_naive_estimate(group_data):
            """Compute naive estimate for a group with safeguards against edge cases."""
            treated = group_data[group_data[treatment_name] == 1]
            control = group_data[group_data[treatment_name] == 0]

            if len(treated) == 0 or len(control) == 0:
                return np.nan

            treated_weights = treated["weights"].values
            control_weights = control["weights"].values

            # Check if weights sum to 0 or if all weights are 0
            if (
                treated_weights.sum() == 0
                or control_weights.sum() == 0
                or not (treated_weights > 0).any()
                or not (control_weights > 0).any()
            ):
                return np.nan

            # Weighted averages with explicit handling of edge cases
            try:
                y1 = np.average(treated[outcome_name], weights=treated_weights)
                y0 = np.average(control[outcome_name], weights=control_weights)
                return y1 - y0
            except ZeroDivisionError:
                return np.nan

        for N in N_values:
            iter_df = working_df.copy()

            try:
                # Ensure enough unique values for binning
                unique_ites = np.unique(iter_df["estimated_ITE"])
                if len(unique_ites) < N:
                    continue

                # Create bins
                iter_df["ITE_bin"] = pd.qcut(iter_df["estimated_ITE"], q=N, labels=False, duplicates="drop")

                # Compute bin statistics
                bin_stats = []
                for bin_idx in iter_df["ITE_bin"].unique():
                    bin_data = iter_df[iter_df["ITE_bin"] == bin_idx]

                    # Skip if bin is too small
                    if len(bin_data) < 2:
                        continue

                    naive_est = compute_naive_estimate(bin_data)

                    # Only compute average ITE if weights are valid
                    bin_weights = bin_data["weights"].values
                    if bin_weights.sum() > 0 and not np.isnan(naive_est):
                        try:
                            avg_est_ite = np.average(bin_data["estimated_ITE"], weights=bin_weights)
                            bin_stats.append(
                                {
                                    "ITE_bin": bin_idx,
                                    "naive_estimate": naive_est,
                                    "average_estimated_ITE": avg_est_ite,
                                }
                            )
                        except ZeroDivisionError:
                            continue

                # Calculate Kendall's Tau if we have enough valid bins
                bin_stats_df = pd.DataFrame(bin_stats)
                if len(bin_stats_df) >= 2:
                    tau, _ = kendalltau(
                        bin_stats_df["naive_estimate"],
                        bin_stats_df["average_estimated_ITE"],
                    )
                    if not np.isnan(tau):
                        kendall_tau_values.append(tau)

            except (ValueError, ZeroDivisionError):
                continue

        # Return final score
        if len(kendall_tau_values) == 0:
            return -np.inf  # Return -inf for failed computations

        top_3_taus = sorted(kendall_tau_values, reverse=True)[:3]
        return np.mean(top_3_taus)

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
                    df[self.causal_model.get_effect_modifiers() + self.causal_model.get_common_causes()]
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
                effect_means, effect_stds = extract_means_stds(est, df, treatment_name, num_treatments)
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
                raise ValueError(f"Malformed scores dict, 'estimator_name' field missing " f"in{k}, {v}")

        estimator_names = sorted(list(set([v["estimator_name"] for v in scores.values() if "estimator_name" in v])))
        best = {}
        for name in estimator_names:
            est_scores = [v for v in scores.values() if "estimator_name" in v and v["estimator_name"] == name]
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
