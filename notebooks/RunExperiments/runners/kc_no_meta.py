import os

from experiment_runner import run_batch, get_estimator_list
from experiment_plots import generate_plots

identifier = "Egor_test"
kind = "KC"
metrics = [
    "erupt",
    # "greedy_erupt",  # regular erupt was made probabilistic,
    "policy_risk",  # NEW
    "qini",
    "auc",
    "psw_energy_distance",
    "frobenius_norm",  # NEW
    "codec",  # NEW
    "bite",  # NEW
]
estimators = get_estimator_list(kind, exclude_patterns=["SLearner", "TLearner", "XLearner"])
ptt_estimators = [
    "lgbm",
    "lrl2",
]

use_ray = True
out_dir = run_batch(
    identifier,
    kind,
    metrics,
    estimators=estimators,
    propensity_automl_estimators=ptt_estimators,
    dataset_path=os.path.realpath("../RunDatasets"),
    use_ray=use_ray,
)
# plot results
# upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
# lower_bounds = {"erupt": 0.06, "bite": 0.75}
generate_plots(os.path.join(out_dir, kind))  # , upper_bounds, lower_bounds)
print("yay!")
