def remote_exec(function, args, use_ray=False):
    if use_ray:
        import ray

        remote_function = ray.remote(function)
        return ray.get(remote_function.remote(*args))
    else:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=1, backend="threading")(delayed(function)(*args) for i in range(1))[
            0
        ]
