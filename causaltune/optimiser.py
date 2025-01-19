import copy
import warnings
from typing import List, Optional, Union
from collections import defaultdict
import time

import traceback
import pandas as pd
import numpy as np
from sklearn.linear_model import _base
from flaml import tune

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from dowhy import CausalModel
from dowhy.causal_identifier import IdentifiedEstimand

from econml.inference import BootstrapInference

from causaltune.search.params import SimpleParamService
from causaltune.score.scoring import Scorer, metrics_to_minimize
from causaltune.utils import treatment_is_multivalue
from causaltune.models.monkey_patches import (
    AutoML,
    apply_multitreatment,
    effect_stderr,
)

# from causaltune.models.monkey_patch_flaml import run

from causaltune.data_utils import CausalityDataset
from causaltune.dataset_processor import CausalityDatasetProcessor
from causaltune.models.passthrough import feature_filter


# Patched from sklearn.linear_model._base to adjust rtol and atol values
def _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale, rtol=1e-4, atol=1e-2):
    n_features = X.shape[1]
    f1 = n_features // 2
    f2 = min(f1 + 1, n_features - 1)

    v1 = (X[:, f1] - X_offset[f1]) * X_scale[f1]
    v2 = (X[:, f2] - X_offset[f2]) * X_scale[f2]

    expected = np.dot(v1, v2)
    actual = precompute[f1, f2]

    if not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise ValueError(
            "Gram matrix passed in via 'precompute' parameter "
            "did not pass validation when a single element was "
            "checked - please check that it was computed "
            f"properly. For element ({f1},{f2}) we computed "
            f"{expected} but the user-supplied value was "
            f"{actual}."
        )


_base._check_precomputed_gram_matrix = _check_precomputed_gram_matrix


class CausalTune:
    """Performs AutoML to find best EconML estimator.
    Optimises hyperparams of component models of each estimator
    and hyperparams of the estimators themselves. Uses the ERUPT
    metric for estimator selection.

    Example:

    .. code-block:: shell

        cd = CausalityDataset(data=df, treatment='T', outcomes=['Y'])
        cd.preprocess_dataset()

        estimator_list = [".LinearDML","LinearDRLearner","metalearners"]
        ct = CausalTune(time_budget=10, estimator_list=estimator_list)

        ct.fit(cd)

        print(f"Best estimator: {ct.best_estimator}")

    """

    def __init__(
        self,
        data_df=None,
        metric="energy_distance",
        metrics_to_report=None,
        time_budget=None,
        verbose=0,
        use_ray=False,
        estimator_list="auto",
        train_size=0.8,
        test_size=None,
        num_samples=-1,
        propensity_model="dummy",
        propensity_automl_estimators: Optional[List[str]] = None,
        outcome_model="nested",
        components_task="regression",
        components_verbose=0,
        components_pred_time_limit=10 / 1e6,
        components_njobs=-1,
        components_time_budget=None,
        try_init_configs=True,
        resources_per_trial=None,
        include_experimental_estimators=False,
        store_all_estimators: Optional[bool] = False,
    ):
        """Constructor.

        Args:
            data_df (pandas.DataFrame): dataset to perform causal inference on
            metric (str): metric to optimise.
            data_df (pandas.DataFrame): dataset to perform causal inference on
            metric (str): metric to optimise.
                Defaults to "erupt" for CATE, "energy_distance" for IV
            metrics_to_report (list): additional metrics to compute and report.
                Defaults to ["qini","auc","ate","erupt", "norm_erupt"] for CATE
                or ["energy_distance"] for IV
            time_budget (float): a number of the time budget in seconds. -1 if no limit.
            num_samples (int): max number of iterations.
            verbose (int):  controls verbosity, higher means more messages. range (0,3). Defaults to 0.
            use_ray (bool): use Ray backend (requires ray to be installed).
            estimator_list (list): a list of strings for estimator names,
            time_budget (float): a number of the time budget in seconds. -1 if no limit.
            num_samples (int): max number of iterations.
            verbose (int):  controls verbosity, higher means more messages. range (0,3). Defaults to 0.
            use_ray (bool): use Ray backend (requires ray to be installed).
            estimator_list (list): a list of strings for estimator names,
             or "auto" for a recommended subset, "all" for all, or a list of substrings of estimator names
               e.g. ```['dml', 'CausalForest']```
            train_size (float): Fraction of data used for training set. Defaults to 0.8.
            test_size (float): Optional size of test dataset. Defaults to None.
            propensity_model (Union[str, Any]): 'dummy' for dummy classifier, 'auto' for AutoML, or an
            train_size (float): Fraction of data used for training set. Defaults to 0.8.
            test_size (float): Optional size of test dataset. Defaults to None.
            propensity_model (Union[str, Any]): 'dummy' for dummy classifier, 'auto' for AutoML, or an
                sklearn-style classifier
            components_task (str): task for component models. Defaults to "regression".
            components_verbose (int): verbosity of component model HPO (hyper parameter optimisation).
            components_task (str): task for component models. Defaults to "regression".
            components_verbose (int): verbosity of component model HPO (hyper parameter optimisation).
                range (0,3). Defaults to 0.
            components_pred_time_limit (float): prediction time limit for component models
            components_njobs (int): number of concurrent jobs for component model optimisation.
            components_pred_time_limit (float): prediction time limit for component models
            components_njobs (int): number of concurrent jobs for component model optimisation.
                Defaults to -1 (all available cores).
            components_time_budget (float): time budget for HPO of component models in seconds.
            components_time_budget (float): time budget for HPO of component models in seconds.
                Defaults to overall time budget / 2.
            try_init_configs (bool): try list of good performing estimators before continuing with HPO.
            try_init_configs (bool): try list of good performing estimators before continuing with HPO.
                Defaults to False.
            resources_per_trial: computational resources per trial, defaults in constructor to {"cpu": 0.5}
            include_experimental_estimators (bool): Include experimental causal estimators. Whether an estimator
            resources_per_trial: computational resources per trial, defaults in constructor to {"cpu": 0.5}
            include_experimental_estimators (bool): Include experimental causal estimators. Whether an estimator
                is experimental can be seen in SimpleParamsService in scoring.py
            store_all_estimators (Optional[bool]). store estimator objects for interim trials. Defaults to False
            store_all_estimators (Optional[bool]). store estimator objects for interim trials. Defaults to False

            Returns:
                None
        """
        assert (
            time_budget is not None or components_time_budget is not None
        ), "Either time_budget or components_time_budget must be specified"

        self._settings = {}
        self._settings["tuner"] = {}
        self._settings["tuner"]["time_budget_s"] = time_budget
        self._settings["tuner"]["num_samples"] = num_samples
        self._settings["tuner"]["verbose"] = verbose
        self._settings["tuner"]["resources_per_trial"] = (
            resources_per_trial if resources_per_trial is not None else {"cpu": 0.5}
        )
        self._settings["try_init_configs"] = try_init_configs
        self._settings["include_experimental_estimators"] = include_experimental_estimators

        # params for FLAML on component models:
        self._settings["component_models"] = {}
        self._settings["component_models"]["task"] = components_task
        self._settings["component_models"]["verbose"] = components_verbose
        self._settings["component_models"]["pred_time_limit"] = components_pred_time_limit
        self._settings["component_models"]["n_jobs"] = components_njobs
        self._settings["component_models"]["time_budget"] = components_time_budget
        self._settings["component_models"]["eval_method"] = "holdout"
        self._settings["propensity_automl_estimators"] = propensity_automl_estimators

        if 0 < train_size < 1:
            component_test_size = 1 - train_size
        else:
            # TODO: convert train_size to fraction based on data size, in fit()
            component_test_size = 0.2
        self._settings["component_models"]["split_ratio"] = component_test_size
        self._settings["train_size"] = train_size
        self._settings["test_size"] = test_size
        self._settings["store_all"] = store_all_estimators
        self._settings["metric"] = metric
        self._settings["metrics_to_report"] = metrics_to_report
        self._settings["propensity_model"] = propensity_model
        self._settings["outcome_model"] = outcome_model

        self.results = None
        self._best_estimators = defaultdict(lambda: (float("-inf"), None))

        self.original_estimator_list = estimator_list
        self.data_df = data_df or pd.DataFrame()
        self.causal_model = None
        self.identified_estimand = None
        self.problem = None
        self.use_ray = use_ray
        # properties that are used to resume fits (warm start)
        self.resume_scores = []
        self.resume_cfg = []

    def get_params(self, deep=False):
        return self._settings.copy()

    def get_estimators(self, deep=False):
        return self.estimator_list.copy()

    def init_propensity_model(self, propensity_model: str):
        # user can choose between flaml and dummy for propensity model.
        if propensity_model == "dummy":
            self.propensity_model = DummyClassifier(strategy="prior")
        elif propensity_model == "auto":
            automl_args = {**self._settings["component_models"], "task": "classification"}
            if self._settings["propensity_automl_estimators"]:
                automl_args["estimator_list"] = self._settings["propensity_automl_estimators"]

            self.propensity_model = AutoML(**automl_args)
        elif hasattr(propensity_model, "fit") and hasattr(propensity_model, "predict_proba"):
            self.propensity_model = propensity_model
        else:
            raise ValueError(
                'propensity_model valid values are "dummy", "auto", or a classifier object'
            )

    def init_outcome_model(self, outcome_model):
        # TODO: implement filtering like below, when there are propensity-only features
        # feature_filter below acts on classes not instances
        # to preserve all the extra methods through inheritance
        # if we are only supplying certain features to the propensity function,
        # make them invisible to the outcome component model
        # This is a workaround for the DoWhy/EconML data model which doesn't
        # support that out of the box

        if hasattr(outcome_model, "fit") and hasattr(outcome_model, "predict"):
            return outcome_model
        elif outcome_model == "auto":
            # Will be dynamically chosen at optimization time
            return outcome_model
        elif outcome_model == "nested":
            # The current default behavior
            return self.auto_outcome_model()
        else:
            raise ValueError('outcome_model valid values are None, "auto", or an estimator object')

    def auto_outcome_model(self):
        data = self.data
        propensity_only_cols = [
            p
            for p in data.propensity_modifiers
            if p not in data.common_causes + data.effect_modifiers
        ]

        if len(propensity_only_cols):
            # TODO: implement feature_filter for arbitrary outcome models
            outcome_model_class = feature_filter(
                AutoML, data.effect_modifiers + data.common_causes, first_cols=True
            )
        else:
            outcome_model_class = AutoML

        return outcome_model_class(**self._settings["component_models"])

    def fit(
        self,
        data: Union[pd.DataFrame, CausalityDataset],
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        common_causes: Optional[List[str]] = None,
        effect_modifiers: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        propensity_modifiers: Optional[List[str]] = None,
        estimator_list: Optional[Union[str, List[str]]] = None,
        resume: Optional[bool] = False,
        time_budget: Optional[int] = None,
        preprocess: bool = False,
        encoder_type: Optional[str] = None,
        encoder_outcome: Optional[str] = None,
        use_ray: Optional[bool] = None,
    ):
        """Performs AutoML on list of causal inference estimators
        - If estimator has a search space specified in its parameters, HPO is performed on the whole model.
        - Otherwise, only its component models are optimised

        Args:
            data (pandas.DataFrame): dataset for causal inference
            treatment (str): name of treatment variable
            outcome (str): name of outcome variable
            common_causes (List[str]): list of names of common causes
            effect_modifiers (List[str]): list of names of effect modifiers
            instruments (List[str]): list of names of instrumental variables
            propensity_modifiers (List[str]): list of names of propensity modifiers
            estimator_list (Optional[Union[str, List[str]]]): subset of estimators to consider
            resume (Optional[bool]): set to True to continue previous fit
            time_budget (Optional[int]): change new time budget allocated to fit, useful for warm starts.
            preprocess (bool): preprocess CausalityDataset if needed.
            encoder_type (Optional[str]): Categorical Encoder for preprocessing
            encoder_outcome (Optional[str]): Categorical Encoder target for preprocessing: TargetEncoder, WOE.

        Returns:
            None
        """
        if use_ray is not None:
            self.use_ray = use_ray

        if outcome is None and isinstance(data, CausalityDataset):
            outcome = data.outcomes[0]

        if not isinstance(data, CausalityDataset):
            assert isinstance(data, pd.DataFrame)
            data = CausalityDataset(
                data,
                treatment,
                outcome,
                common_causes=common_causes,
                effect_modifiers=effect_modifiers,
                instruments=instruments,
                propensity_modifiers=propensity_modifiers,
            )

        if preprocess:
            data = copy.deepcopy(data)
            self.dataset_processor = CausalityDatasetProcessor()
            self.dataset_processor.fit(data, encoder_type=encoder_type, outcome=encoder_outcome)
            data = self.dataset_processor.transform(data)
        else:
            self.dataset_processor = None

        self.data = data
        treatment_values = data.treatment_values

        assert len(treatment_values) > 1, "Treatment must take at least 2 values, eg 0 and 1!"

        self._control_value = treatment_values[0]
        self._treatment_values = list(treatment_values[1:])

        # To be used for component model training/selection
        self.train_df, self.test_df = train_test_split(
            self.data.data, train_size=self._settings["train_size"], shuffle=True
        )

        # smuggle propensity modifiers into common causes, filter later in component models
        self.causal_model = CausalModel(
            data=self.train_df,
            treatment=data.treatment,
            outcome=outcome,
            common_causes=data.common_causes + data.propensity_modifiers,
            effect_modifiers=data.effect_modifiers,
            instruments=data.instruments,
        )

        self.init_propensity_model(self._settings["propensity_model"])

        self.identified_estimand: IdentifiedEstimand = self.causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )

        if bool(self.identified_estimand.estimands["iv"]) and bool(data.instruments):
            self.problem = "iv"
        elif bool(self.identified_estimand.estimands["backdoor"]):
            self.problem = "backdoor"
        else:
            raise ValueError(
                "Couldn't identify the kind of problem from "
                + str(self.identified_estimand.estimands)
            )

        # This must be stateful because we need to train the treatment propensity function
        self.scorer = Scorer(
            self.causal_model,
            self.propensity_model,
            self.problem,
            treatment_is_multivalue(self._treatment_values),
        )

        self.metric = self.scorer.resolve_metric(self._settings["metric"])
        self.metrics_to_report = self.scorer.resolve_reported_metrics(
            self._settings["metrics_to_report"], self.metric
        )

        if self.metric in ["energy_distance", "psw_energy_distance"]:
            self._best_estimators = defaultdict(lambda: (float("inf"), None))

        # TODO: allow specifying an exclusion list, too
        used_estimator_list = (
            self.original_estimator_list if estimator_list is None else estimator_list
        )

        assert (
            isinstance(used_estimator_list, str) or len(used_estimator_list) > 0
        ), "estimator_list must either be a str or an iterable of str"

        # config with method-specific params
        self.cfg = SimpleParamService(
            n_jobs=self._settings["component_models"]["n_jobs"],
            include_experimental=self._settings["include_experimental_estimators"],
            multivalue=treatment_is_multivalue(self._treatment_values),
            sample_outcome_estimators=self._settings["outcome_model"] == "auto",
        )

        self.estimator_list = self.cfg.estimator_names_from_patterns(
            self.problem,
            used_estimator_list,
            len(self.data),
        )

        if not self.estimator_list:
            raise ValueError(
                f"No valid estimators in {str(used_estimator_list)}, "
                f"available estimators: {str(self.cfg.estimator_names)}"
            )

        if time_budget:
            self._settings["tuner"]["time_budget_s"] = time_budget

        if self._settings["component_models"]["time_budget"] is None:
            self._settings["component_models"]["time_budget"] = self._settings["tuner"][
                "time_budget_s"
            ] / (2.5 * len(self.estimator_list))

        if (
            self._settings["tuner"]["time_budget_s"] is None
            and self._settings["tuner"]["num_samples"] == -1
        ):
            self._settings["tuner"]["time_budget_s"] = (
                2.5 * len(self.estimator_list) * self._settings["component_models"]["time_budget"]
            )

        cmtb = self._settings["component_models"]["time_budget"]

        if cmtb < 300:
            warnings.warn(
                f"Component model time budget is {cmtb}. "
                f"Recommended value is at least 300 for smallish datasets, 1800 for datasets with> 100K rows"
            )

        if self._settings["test_size"] is not None:
            self.test_df = self.test_df.sample(self._settings["test_size"])

        if "r_scorer" in self.metrics_to_report:
            raise NotImplementedError(
                "R-squared scorer no longer suported, please raise an issue if you want it back"
            )
        # self.r_scorer = (
        #     None
        #     if "r_scorer" not in self.metrics_to_report
        #     else RScoreWrapper(
        #         self.outcome_model,
        #         self.propensity_model,
        #         self.train_df,
        #         self.test_df,
        #         outcome,
        #         treatment,
        #         common_causes,
        #         effect_modifiers,
        #     )
        # )

        search_space = self.cfg.search_space(self.estimator_list, data_size=data.data.shape)
        init_cfg = (
            self.cfg.default_configs(self.estimator_list, data_size=data.data.shape)
            if self._settings["try_init_configs"]
            else []
        )

        if resume and self.results:
            # pull out configs and resume_scores from previous trials:
            for _, result in self.results.results.items():
                self.resume_scores.append(result[self.metric])
                self.resume_cfg.append(result["config"])
            # append init_cfgs that have not yet been evaluated
            for cfg in init_cfg:
                self.resume_cfg.append(cfg) if cfg not in self.resume_cfg else None
        try:
            self.results = tune.run(
                self._tune_with_config,
                search_space,
                metric=self.metric,
                # use_ray=self.use_ray,
                cost_attr="evaluation_cost",
                points_to_evaluate=(init_cfg if len(self.resume_cfg) == 0 else self.resume_cfg),
                evaluated_rewards=([] if len(self.resume_scores) == 0 else self.resume_scores),
                mode=("min" if self.metric in metrics_to_minimize() else "max"),
                # resources_per_trial= {"cpu": 1} if self.use_ray else None,
                low_cost_partial_config={},
                **self._settings["tuner"],
            )

            if self.results.get_best_trial() is None:
                raise Exception(
                    "Optimization failed! Did you set large enough time_budget and components_budget?"
                )
        except Exception:
            # we must have an older FLAML version that doesn't support the cost_attr parameter
            self.results = tune.run(
                self._tune_with_config,
                search_space,
                metric=self.metric,
                points_to_evaluate=(init_cfg if len(self.resume_cfg) == 0 else self.resume_cfg),
                evaluated_rewards=([] if len(self.resume_scores) == 0 else self.resume_scores),
                mode=("min" if self.metric in metrics_to_minimize() else "max"),
                low_cost_partial_config={},
                **self._settings["tuner"],
            )
            # print("Optimization failed!\n", traceback.format_exc())
            # raise e
        self.update_summary_scores()

    def update_summary_scores(self):
        """Stores scores for metric of interest for each estimator

        Returns:
            None
        """
        self.scores = Scorer.best_score_by_estimator(self.results.results, self.metric)
        # now inject the separately saved model objects
        for est_name in self.scores:
            # Todo: Check approximate scores for OrthoIV (possibly other IV estimators)
            # assert (
            #     self._best_estimators[est_name][0] == self.scores[est_name][self.metric]
            # ), "Can't match best model to score"
            self.scores[est_name]["estimator"] = self._best_estimators[est_name][1]

    def _tune_with_config(self, config: dict) -> dict:
        """
        Performs Hyperparameter Optimisation for a causal inference estimator.

        Args:
            config (dict): Dictionary with search space for all tunable parameters.

        Returns:
            (dict): values of metrics after optimisation
        """
        from causaltune.remote import remote_exec

        if self.use_ray:
            # flaml.tune handles the interaction with Ray itself
            # estimates = self._estimate_effect(config)
            estimates = remote_exec(CausalTune._estimate_effect, (self, config), self.use_ray)
        else:
            estimates = remote_exec(CausalTune._estimate_effect, (self, config), self.use_ray)

        #     Parallel(n_jobs=2, backend="threading")(
        #     delayed(self._estimate_effect)(config) for i in range(1)
        # ))[0]

        if "exception" not in estimates:
            est_name = estimates["estimator_name"]
            current_score = estimates[self.metric]

            estimates["optimization_score"] = current_score
            estimates["evaluation_cost"] = 1e8  # will be overwritten for successful runs

            # Initialize best_score if this is the first estimator for this name
            if est_name not in self._best_estimators:
                self._best_estimators[est_name] = (
                    (
                        np.inf
                        if self.metric
                        in [
                            "energy_distance",
                            "psw_energy_distance",
                            "frobenius_norm",
                            "psw_frobenius_norm",
                            "codec",
                            "policy_risk",
                        ]
                        else -np.inf
                    ),
                    None,
                )

            best_score = self._best_estimators[est_name][0]

            # Determine if the current estimator performs better, handling inf values
            if self.metric in [
                "energy_distance",
                "psw_energy_distance",
                "frobenius_norm",
                "psw_frobenius_norm",
                "codec",
                "policy_risk",
            ]:
                is_better = (np.isfinite(current_score) and current_score < best_score) or (
                    np.isinf(best_score) and np.isfinite(current_score)
                )
            else:
                is_better = (np.isfinite(current_score) and current_score > best_score) or (
                    np.isinf(best_score) and np.isfinite(current_score)
                )

            # Store the estimator if we're storing all, if it's better, or if it's the first valid (non-inf) estimator
            if (
                self._settings["store_all"]
                or is_better
                or (self._best_estimators[est_name][1] is None and np.isfinite(current_score))
            ):
                self._best_estimators[est_name] = (
                    current_score,
                    (
                        estimates["estimator"]
                        if self._settings["store_all"]
                        else estimates.pop("estimator")
                    ),
                )
            if "Dummy" not in est_name:
                estimates["evaluation_cost"] = estimates.pop("elapsed_time")

        return estimates

    def _est_effect_stub(self, method_params):
        return self.causal_model.estimate_effect(
            self.identified_estimand,
            method_name=self.estimator_name,
            control_value=self._control_value,
            treatment_value=self._treatment_values,
            target_units="ate",  # condition used for CATE
            confidence_intervals=False,
            method_params=method_params,
        )

    def _estimate_effect(self, config):
        """estimates effect with chosen estimator"""

        # Do we need an boject property for this, instead of a local var?
        self.estimator_name = config["estimator"]["estimator_name"]
        outcome_model = self.init_outcome_model(self._settings["outcome_model"])
        method_params = self.cfg.method_params(config, outcome_model, self.propensity_model)

        try:  #
            # This calls the causal model's estimate_effect method
            start_time = time.time()
            estimate = self._est_effect_stub(method_params)
            scores = {
                "estimator_name": self.estimator_name,
                "train": self._compute_metrics(
                    estimate,
                    self.train_df,
                ),
                "validation": self._compute_metrics(
                    estimate,
                    self.test_df,
                ),
            }
            elapsed_time = time.time() - start_time
            return {
                self.metric: scores["validation"][self.metric],
                "estimator": estimate,
                "estimator_name": self.estimator_name,
                "scores": scores,
                # TODO: return full config!
                "config": config,
                "elapsed_time": elapsed_time,
            }
        except Exception as e:
            print("Evaluation failed!\n", config, traceback.format_exc())
            return {
                self.metric: -np.inf,
                "estimator_name": self.estimator_name,
                "exception": e,
                "traceback": traceback.format_exc(),
            }

    def _compute_metrics(self, estimator, df: pd.DataFrame) -> dict:
        return self.scorer.make_scores(estimator, df, self.metrics_to_report, r_scorer=None)

    def score_dataset(self, df: pd.DataFrame, dataset_name: str):
        """
        After fitting, generate scores for an additional dataset, add them to the scores dict.

        Args:
            df (pandas.DataFrame): input dataframe
            dataset_name (str): dictionary key

        Returns:
            None.
        """
        for scr in self.scores.values():
            if scr["estimator"] is None:
                warnings.warn("Skipping scoring for estimator %s", scr["estimator_name"])
            else:
                scr["scores"][dataset_name] = self._compute_metrics(scr["estimator"], df)

    @property
    def best_estimator(self) -> str:
        """A string indicating the best estimator found

        Returns:
            None
        """
        return self.results.best_result["estimator_name"]

    @property
    def model(self):
        """Return the *trained* best estimator

        Returns:
            CausalEstimator
        """
        return self.results.best_result["estimator"].estimator

    def best_model_for_estimator(self, estimator_name):
        """Return the best model found for a particular estimator.
        estimator: self.tune_results[estimator].best_config

        Args:
            estimator_name (str): the estimator's name.

        Returns:
            (dowhy.causal_estimator.CausalEstimate): the best model for estimator_name.
        """
        # Note that this returns the trained Econml estimator, whose attributes include
        # fitted  models for E[T | X, W], for E[Y | X, W], CATE model, etc.
        return self.scores[estimator_name]["estimator"]

    @property
    def best_config(self):
        """
        Returns:
            (dict): the best configuration
        """
        return self.results.best_config

    @property
    def best_config_per_estimator(self):
        """
        Returns:
            (dict): all estimators' best configuration."""
        return {e: s["config"] for e, s in self.scores.values()}

    @property
    def best_score_per_estimator(self):
        """A dictionary of all estimators' best score."""
        return {}

    @property
    def best_score(self):
        """
        Returns:
            (float):  the best score found."""
        return self.results.best_result[self.metric]

    def effect(self, df, *args, **kwargs):
        """Heterogeneous Treatment Effects for data df

        Args:
            df (pd.DataFrame): data to predict treatment effect for

        Returns:
            (np.ndarray): predicted treatment effect for each datapoint

        """
        return self.model.effect(df, *args, **kwargs)

    def predict(self, cd: CausalityDataset, preprocess: Optional[bool] = False, *args, **kwargs):
        """Heterogeneous Treatment Effects for data CausalityDataset

        Args:
            cd (CausalityDataset): data to predict treatment effect for

        Returns:
            (np.ndarray): predicted treatment effect for each datapoint

        """
        if preprocess:
            cd = copy.deepcopy(cd)
            if self.dataset_processor:
                cd = self.dataset_processor.transform(cd)
            else:
                raise ValueError("CausalityDatasetProcessor has not been trained")
        return self.model.effect(cd.data, *args, **kwargs)

    def effect_inference(self, df, *args, **kwargs):
        """Inference (uncertainty) results produced by best estimator
        Only implemented for EconML estimators so far

        Args:
            df (pd.DataFrame): data to run inference on
            args: passed through to underlying estimator
            kwargs: passed through to underlying estimator

        Returns:
            (from EconML: NormalInferenceResults):
                EconML results object for inference assuming a normal distribution.
                from EconML: NormalInferenceResults:

        """

        if "Econml" in str(type(self.model)):
            # Get a list of "Inference" objects from EconML, one per treatment
            self.model.__class__.apply_multitreatment = apply_multitreatment
            if self.cfg._configs()[self.best_estimator].inference == "bootstrap":
                raise NotImplementedError(
                    f"Can't calculate stds for estimator \
                {self.best_estimator} \
                as boostrap inference is not supported yet"
                )
            return self.model.effect_inference(df, *args, **kwargs)
        else:
            raise NotImplementedError(
                "No pointwise error estimates for non-EconML estimators implemented yet"
            )

    def effect_stderr(self, df, n_bootstrap_samples=5, n_jobs=1, *args, **kwargs):
        """Compute standard errors for best causal estimator
            Currently implemented for EconML estimators.
            Computes analytical standard errors if available and boostraps otherwise.
        Args:
            df (pd.DataFrame): data to run inference on
            n_bootstrap_samples (int, optional): number of runs if standard errors are boostrapped. Defaults to 5.
            n_jobs (int, optional): Number of bootstrap estimates to run in parallel. Defaults to 1.

        Returns:
            (np.ndarray): standard error for each data point from df
        Args:
            df (pd.DataFrame): data to run inference on
            n_bootstrap_samples (int, optional): number of runs if standard errors are boostrapped. Defaults to 5.
            n_jobs (int, optional): Number of bootstrap estimates to run in parallel. Defaults to 1.

        Returns:
            np.ndarray: standard error for each data point from df
        """

        if "Econml" in str(type(self.model)):
            # Get a list of "Inference" objects from EconML, one per treatment
            self.model.__class__.effect_stderr = effect_stderr
            outcome_model = self.init_outcome_model(self._settings["outcome_model"])
            method_params = self.cfg.method_params(
                self.best_config, outcome_model, self.propensity_model
            )

            if self.cfg.full_config(self.best_estimator).inference == "bootstrap":
                # TODO: before bootstrapping, check whether that's already been done
                bootstrap = BootstrapInference(
                    n_bootstrap_samples=n_bootstrap_samples, n_jobs=n_jobs
                )
                method_params["fit_params"]["inference"] = bootstrap
                self.estimator_name = (
                    self.best_estimator
                )  # needed for _est_effect_stub, just in case
                self.bootstrapped_estimate = self._est_effect_stub(method_params)
                est = self.bootstrapped_estimate.estimator
            else:
                # If the estimator supports other inference methods,
                # those have already been included
                est = self.model
            return est.effect_stderr(df, *args, **kwargs)
        else:
            raise NotImplementedError(
                "No pointwise error estimates for non-EconML estimators implemented yet"
            )
