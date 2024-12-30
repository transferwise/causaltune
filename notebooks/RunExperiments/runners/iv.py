import os
import ray

from experiment_runner import run_batch
from experiment_plots import generate_plots

identifier = "Egor_test"
kind = "IV"
metrics = ["energy_distance", "frobenius_norm", "codec"]
use_ray = False
remote_function = ray.remote(run_batch)
calls = []

out_dir = run_batch(
    identifier, kind, metrics, dataset_path=os.path.realpath("../RunDatasets"), use_ray=use_ray
)
# plot results
# upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
# lower_bounds = {"erupt": 0.06, "bite": 0.75}
generate_plots(os.path.join(out_dir, kind))  # , upper_bounds, lower_bounds)
print("yay!")
