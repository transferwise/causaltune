import argparse
import copy
import os
import pickle
import sys
import time
import warnings
from typing import List, Optional

import numpy as np
import ray
from sklearn.model_selection import train_test_split


from causaltune import CausalTune
from causaltune.data_utils import CausalityDataset
from causaltune.models.passthrough import passthrough_model
from experiment_plots import make_filename

# Ensure CausalTune is in the Python path
root_path = os.path.realpath("../../../../..")  # noqa: E402
sys.path.append(os.path.join(root_path, "causaltune"))  # noqa: E402

# Import CausalTune and other custom modules after setting up the path
from causaltune.datasets import load_dataset  # noqa: E402
from causaltune.search.params import SimpleParamService  # noqa: E402
from causaltune.score.scoring import (
    metrics_to_minimize,  # noqa: E402
    # noqa: E402
)

# Configure warnings
warnings.filterwarnings("ignore")

RAY_NAMESPACE = "causaltune_experiments"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CausalTune experiments")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psw_frobenius_norm"],
        help="Metrics to use for evaluation",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Small Linear_RCT"],
        help="Datasets to use (format: Size Name, e.g., Small Linear_RCT)",
    )
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--num_samples", type=int, default=-1, help="Maximum number of iterations")

    parser.add_argument("--outcome_model", type=str, default="nested", help="Outcome model type")
    parser.add_argument(
        "--timestamp_in_dirname",
        type=bool,
        default="False",
        help="Include timestampl in out_dir name?",
    )

    parser.add_argument("--test_size", type=float, default=0.33, help="Test set size")
    parser.add_argument(
        "--time_budget", type=int, default=None, help="Time budget for optimization"
    )
    parser.add_argument(
        "--components_time_budget",
        type=int,
        default=None,
        help="Time budget for component optimization",
    )
    parser.add_argument(
        "--identifier", default="", help="Additional identifier for output directory"
    )
    return parser.parse_args()


def get_estimator_list(
    dataset_name,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
):
    assert (
        include_patterns is None or exclude_patterns is None
    ), "Cannot specify both include and exclude patterns"
    if "IV" in dataset_name:
        problem = "iv"
    else:
        problem = "backdoor"

    cfg = SimpleParamService(
        n_jobs=-1,
        include_experimental=False,
        multivalue=False,
    )
    estimator_list = cfg.estimator_names_from_patterns(problem, "all", 1001)

    out = [est for est in estimator_list if "Dummy" not in est]

    if include_patterns is not None:
        out = [est for est in out if any(pat in est for pat in include_patterns)]

    if exclude_patterns is not None:
        out = [est for est in out if not any(pat in est for pat in exclude_patterns)]

    return out


def run_experiment(
    args,
    estimators: List[str],
    dataset_path: str,
    use_ray: bool,
):
    # Process datasets
    data_sets = {}
    for dataset in args.datasets:
        parts = dataset.split()
        if len(parts) < 2:
            raise ValueError(
                f"Invalid dataset format: {dataset}. Expected format: Size Name (e.g., Small Linear_RCT)"
            )
        size = parts[0]
        name = " ".join(parts[1:])
        file_path = f"{dataset_path}/{size}/{name}.pkl"
        data_sets[f"{size} {name}"] = load_dataset(file_path)

    out_dir = f"../EXPERIMENT_RESULTS_{args.identifier}"
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.realpath(os.path.join(out_dir, size))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded datasets: {list(data_sets.keys())}")

    already_running = False
    if use_ray:
        try:
            runner = ray.get_actor("TaskRunner")
            print("\n" * 4)
            print(
                "!!! Found an existing detached TaskRunner. Will assume the tasks have already been submitted."
            )
            print(
                "!!! If you want to re-run the experiments from scratch, "
                'run ray.kill(ray.get_actor("TaskRunner", namespace="{}")) or recreate the cluster.'.format(
                    RAY_NAMESPACE
                )
            )
            print("\n" * 4)
            already_running = True
        except ValueError:
            print("Ray: no detached TaskRunner found, creating...")
            # This thing will be alive even if the host program exits
            # Must be killed explicitly: ray.kill(ray.get_actor("TaskRunner"))
            runner = TaskRunner.options(name="TaskRunner", lifetime="detached").remote()

    out = []
    if not already_running:
        tasks = []
        i_run = 1

        for dataset_name, cd in data_sets.items():

            # Extract case while preserving original string checking logic
            if "KCKP" in dataset_name:
                case = "KCKP"
            elif "KC" in dataset_name:
                case = "KC"
            elif "IV" in dataset_name:
                case = "IV"
            else:
                case = "RCT"

            os.makedirs(f"{out_dir}/{case}", exist_ok=True)
            for metric in args.metrics:
                fn = make_filename(metric, dataset_name, i_run)
                out_fn = os.path.join(out_dir, case, fn)
                if os.path.isfile(out_fn):
                    print(f"File {out_fn} exists, skipping...")
                    continue
                if use_ray:
                    tasks.append(
                        runner.remote_single_run.remote(
                            dataset_name,
                            cd,
                            metric,
                            args.test_size,
                            args.num_samples,
                            args.components_time_budget,
                            out_fn,
                            estimators,
                        )
                    )
                else:
                    results = single_run(
                        dataset_name,
                        cd,
                        metric,
                        args.test_size,
                        args.num_samples,
                        args.components_time_budget,
                        out_fn,
                        estimators,
                    )
                    out.append(results)

    if use_ray:
        while True:
            completed, in_progress = ray.get(runner.get_progress.remote())
            print(f"Ray: {completed}/{completed + in_progress} tasks completed")
            if not in_progress:
                print("Ray: all tasks completed!")
                break
            time.sleep(10)

        print("Ray: fetching results...")
        out = ray.get(runner.get_results.remote())

    for out_fn, results in out:
        with open(out_fn, "wb") as f:
            pickle.dump(results, f)

    if use_ray:
        destroy = input("Ray: seems like the results fetched OK. Destroy TaskRunner? ")
        if destroy.lower().startswith("y"):
            print("Destroying TaskRunner... ", end="")
            ray.kill(runner)
            print("success!")

    return out_dir


def run_batch(
    identifier: str,
    kind: str,
    metrics: List[str],
    estimators: List[str],
    dataset_path: str,
    use_ray: bool = False,
):
    args = parse_arguments()
    args.identifier = identifier
    args.metrics = metrics
    # run_experiment assumes we don't mix large and small datasets in the same call
    args.datasets = [f"Large Linear_{kind}", f"Large NonLinear_{kind}"]
    args.num_samples = 100
    args.timestamp_in_dirname = False
    args.outcome_model = "auto"  # or use "nested" for the old-style nested model
    args.components_time_budget = 120

    if use_ray:
        import ray

        # Assuming we port-mapped already by running ray dashboard
        ray.init(
            "ray://localhost:10001",
            runtime_env={"working_dir": ".", "pip": ["causaltune", "catboost", "ray[tune]"]},
            namespace=RAY_NAMESPACE,
        )

    out_dir = run_experiment(
        args, estimators=estimators, dataset_path=dataset_path, use_ray=use_ray
    )
    return out_dir


@ray.remote
def remote_single_run(*args):
    return single_run(*args)


@ray.remote
class TaskRunner:
    def __init__(self):
        self.futures = {}

    def remote_single_run(self, *args):
        ref = remote_single_run.remote(*args)
        self.futures[ref.hex()] = ref
        return ref.hex()

    def get_results(self):
        return ray.get(list(self.futures.values()))

    def get_single_result(self, ref_hex):
        return ray.get(self.futures[ref_hex])

    def is_ready(self, ref_hex):
        ready, _ = ray.wait([self.futures[ref_hex]], timeout=0, fetch_local=False)
        return bool(ready)

    def all_tasks_ready(self):
        _, in_progress = ray.wait(list(self.futures.values()), timeout=0, fetch_local=False)
        return not bool(in_progress)

    def get_progress(self):
        completed, in_progress = ray.wait(
            list(self.futures.values()), num_returns=len(self.futures), timeout=0, fetch_local=False
        )
        return len(completed), len(in_progress)


def single_run(
    dataset_name: str,
    cd: CausalityDataset,
    metric: str,
    test_size: float,
    num_samples: int,
    components_time_budget: int,
    out_fn: str,
    estimators: List[str],
    outcome_model: str = "auto",
    i_run: int = 1,
):

    cd_i = copy.deepcopy(cd)
    train_df, test_df = train_test_split(cd_i.data, test_size=test_size)
    test_df = test_df.reset_index(drop=True)
    cd_i.data = train_df
    print(f"Optimizing {metric} for {dataset_name} (run {i_run})")
    try:

        # Set propensity model using string checking like original version
        if "KCKP" in dataset_name:
            print(f"Using passthrough propensity model for {dataset_name}")
            propensity_model = passthrough_model(cd_i.propensity_modifiers, include_control=False)
        elif "KC" in dataset_name:
            print(f"Using auto propensity model for {dataset_name}")
            propensity_model = "auto"
        else:
            print(f"Using dummy propensity model for {dataset_name}")
            propensity_model = "dummy"

        ct = CausalTune(
            metric=metric,
            estimator_list=estimators,
            num_samples=num_samples,
            components_time_budget=components_time_budget,  # Use this instead
            verbose=1,
            components_verbose=1,
            store_all_estimators=True,
            propensity_model=propensity_model,
            outcome_model=outcome_model,
            use_ray=False,
        )

        ct.fit(
            data=cd_i,
            treatment="treatment",
            outcome="outcome",
        )

        # Embedding this so it ships well to Ray remotes

        def compute_scores(ct, metric, test_df):
            datasets = {"train": ct.train_df, "validation": ct.test_df, "test": test_df}
            estimator_scores = {est: [] for est in ct.scores.keys() if "NewDummy" not in est}

            all_scores = []
            for trial in ct.results.trials:
                try:
                    estimator_name = trial.last_result["estimator_name"]
                    if "estimator" in trial.last_result and trial.last_result["estimator"]:
                        estimator = trial.last_result["estimator"]
                        scores = {}
                        for ds_name, df in datasets.items():
                            scores[ds_name] = {}
                            est_scores = ct.scorer.make_scores(
                                estimator,
                                df,
                                metrics_to_report=ct.metrics_to_report,
                            )
                            est_scores["estimator_name"] = estimator_name

                            scores[ds_name]["CATE_estimate"] = np.squeeze(
                                estimator.estimator.effect(df)
                            )
                            scores[ds_name]["CATE_groundtruth"] = np.squeeze(df["true_effect"])
                            est_scores["MSE"] = np.mean(
                                (
                                    scores[ds_name]["CATE_estimate"]
                                    - scores[ds_name]["CATE_groundtruth"]
                                )
                                ** 2
                            )
                            scores[ds_name]["scores"] = est_scores
                        scores["optimization_score"] = trial.last_result.get("optimization_score")
                        estimator_scores[estimator_name].append(copy.deepcopy(scores))
                    # Will use this in the nex
                    all_scores.append(scores)
                except Exception as e:
                    print(f"Error processing trial: {e}")

            for k in estimator_scores.keys():
                estimator_scores[k] = sorted(
                    estimator_scores[k],
                    key=lambda x: x["validation"]["scores"][metric],
                    reverse=metric not in metrics_to_minimize(),
                )

            # Debugging: Log final result structure
            print(f"Returning scores for metric {metric}: Best estimator: {ct.best_estimator}")

            return {
                "best_estimator": ct.best_estimator,
                "best_config": ct.best_config,
                "best_score": ct.best_score,
                "optimised_metric": metric,
                "scores_per_estimator": estimator_scores,
                "all_scores": all_scores,
            }

        # Compute scores and save results
        results = compute_scores(ct, metric, test_df)

        return out_fn, results
    except Exception as e:
        print(f"Error processing {dataset_name}_{metric}_{i_run}: {e}")
