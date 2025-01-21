import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


def random_nn(ids):
    """
    Generate a list of random nearest neighbors.

    Parameters:
    ids (array-like): List of indices to sample from.

    Returns:
    numpy.ndarray: Array of sampled indices with no position i having x[i] == i.
    """
    m = len(ids)
    x = np.random.choice(m - 1, m, replace=True)
    x = x + (x >= np.arange(m))
    return np.array(ids)[x]


def estimate_conditional_q(Y, X, Z):
    """
    Estimate Q(Y, Z | X), the numerator of the measure of conditional dependence of Y on Z given X.

    Parameters:
    Y (array-like): Vector of responses (length n).
    X (array-like): Matrix of predictors (n by p).
    Z (array-like): Matrix of predictors (n by q).

    Returns:
    float: Estimation of Q(Y, Z | X).
    """
    # Ensure X and Z are numpy arrays
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    Z = np.array(Z) if not isinstance(Z, np.ndarray) else Z

    # Reshape Z if needed
    Z = Z.reshape(-1, 1)

    n = len(Y)
    W = np.hstack((X, Z))

    # Compute nearest neighbors for X
    nn_X = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    nn_dists_X, nn_indices_X = nn_X.kneighbors(X)
    nn_index_X = nn_indices_X[:, 1]

    # Handle repeated data for X
    repeat_data = np.where(nn_dists_X[:, 1] == 0)[0]
    df_X = pd.DataFrame({"id": repeat_data, "group": nn_indices_X[repeat_data, 0]})
    df_X["rnn"] = df_X.groupby("group")["id"].transform(random_nn)
    nn_index_X[repeat_data] = df_X["rnn"].values

    # Handle ties for X
    ties = np.where(nn_dists_X[:, 1] == nn_dists_X[:, 2])[0]
    ties = np.setdiff1d(ties, repeat_data)

    if len(ties) > 0:

        def helper_ties(a):
            distances = distance.cdist(
                X[a].reshape(1, -1), np.delete(X, a, axis=0)
            ).flatten()
            ids = np.where(distances == distances.min())[0]
            x = np.random.choice(ids)
            return x + (x >= a)

        nn_index_X[ties] = [helper_ties(a) for a in ties]

    # Compute nearest neighbors for W
    nn_W = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(W)
    nn_dists_W, nn_indices_W = nn_W.kneighbors(W)
    nn_index_W = nn_indices_W[:, 1]

    # Handle repeated data for W
    repeat_data = np.where(nn_dists_W[:, 1] == 0)[0]
    df_W = pd.DataFrame({"id": repeat_data, "group": nn_indices_W[repeat_data, 0]})
    df_W["rnn"] = df_W.groupby("group")["id"].transform(random_nn)
    nn_index_W[repeat_data] = df_W["rnn"].values

    # Handle ties for W
    ties = np.where(nn_dists_W[:, 1] == nn_dists_W[:, 2])[0]
    ties = np.setdiff1d(ties, repeat_data)

    if len(ties) > 0:
        nn_index_W[ties] = [helper_ties(a) for a in ties]

    # Estimate Q
    R_Y = np.argsort(np.argsort(Y))  # Rank Y with ties method 'max'
    Q_n = (
        np.sum(np.minimum(R_Y, R_Y[nn_index_W]))
        - np.sum(np.minimum(R_Y, R_Y[nn_index_X]))
    ) / (n**2)

    return Q_n


def estimate_conditional_s(Y, X):
    """
    Estimate S(Y, X), the denominator of the measure of dependence of Y on Z given X.

    Parameters:
    Y (array-like): Vector of responses (length n).
    X (array-like): Matrix of predictors (n by p).

    Returns:
    float: Estimation of S(Y, X).
    """
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    n = len(Y)

    # Compute nearest neighbors
    nn_X = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    nn_dists_X, nn_indices_X = nn_X.kneighbors(X)
    nn_index_X = nn_indices_X[:, 1]

    # Handle repeated data
    repeat_data = np.where(nn_dists_X[:, 1] == 0)[0]
    df_X = pd.DataFrame({"id": repeat_data, "group": nn_indices_X[repeat_data, 0]})
    df_X["rnn"] = df_X.groupby("group")["id"].transform(random_nn)
    nn_index_X[repeat_data] = df_X["rnn"].values

    # Handle ties
    ties = np.where(nn_dists_X[:, 1] == nn_dists_X[:, 2])[0]
    ties = np.setdiff1d(ties, repeat_data)

    if len(ties) > 0:

        def helper_ties(a):
            distances = distance.cdist(
                X[a].reshape(1, -1), np.delete(X, a, axis=0)
            ).flatten()
            ids = np.where(distances == distances.min())[0]
            x = np.random.choice(ids)
            return x + (x >= a)

        nn_index_X[ties] = [helper_ties(a) for a in ties]

    # Estimate S
    R_Y = np.argsort(np.argsort(Y))  # Rank Y with ties method 'max'
    S_n = np.sum(R_Y - np.minimum(R_Y, R_Y[nn_index_X])) / (n**2)

    return S_n


def estimate_conditional_t(Y, Z, X):
    """
    Estimate T(Y, Z | X), the measure of dependence of Y on Z given X.

    Parameters:
    Y (array-like): Vector of responses (length n).
    Z (array-like): Matrix of predictors (n by q).
    X (array-like): Matrix of predictors (n by p).

    Returns:
    float: Estimation of T(Y, Z | X).
    """
    S = estimate_conditional_s(Y, X)
    return 1 if S == 0 else estimate_conditional_q(Y, X, Z) / S


def codec(Y, Z, X=None, na_rm=True):
    """
    Estimate the conditional dependence coefficient (CODEC).

    The conditional dependence coefficient (CODEC) is a measure of the amount of conditional
    dependence between a random variable Y and a random vector Z given a random vector X,
    based on an i.i.d. sample of (Y, Z, X). The coefficient is asymptotically guaranteed
    to be between 0 and 1.

    Parameters:
        Y (array-like): Vector of responses (length n).
        Z (array-like): Matrix of predictors (n by q).
        X (array-like, optional): Matrix of predictors (n by p). Default is None.
        na_rm (bool): If True, remove NAs.

    Returns:
        float: The conditional dependence coefficient (CODEC) of Y and Z given X.
        If X is None, this is just a measure of the dependence between Y and Z.

    References:
        Azadkia, M. and Chatterjee, S. (2019). A simple measure of conditional dependence.
        https://arxiv.org/pdf/1910.12327.pdf
    """
    if X is None:
        Y = np.array(Y) if not isinstance(Y, np.ndarray) else Y
        Z = np.array(Z) if not isinstance(Z, np.ndarray) else Z

        if len(Y) != Z.shape[0]:
            raise ValueError("Number of rows of Y and Z should be equal.")

        if na_rm:
            mask = np.isfinite(Y) & np.all(np.isfinite(Z), axis=1)
            Z = Z[mask]
            Y = Y[mask]

        n = len(Y)
        if n < 2:
            raise ValueError("Number of rows with no NAs should be greater than 1.")

        return estimate_conditional_q(Y, Z, np.zeros((n, 0)))

    # Ensure inputs are in proper format for conditional case
    Y = np.array(Y) if not isinstance(Y, np.ndarray) else Y
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    Z = np.array(Z) if not isinstance(Z, np.ndarray) else Z

    if len(Y) != X.shape[0] or len(Y) != Z.shape[0] or X.shape[0] != Z.shape[0]:
        raise ValueError("Number of rows of Y, X, and Z should be equal.")

    n = len(Y)
    if n < 2:
        raise ValueError("Number of rows with no NAs should be greater than 1.")

    return estimate_conditional_t(Y, Z, X)


def identify_confounders(
    df: pd.DataFrame, treatment_col: str, outcome_col: str
) -> list:
    """
    Identify confounders in a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe
        treatment_col (str): Name of the treatment column
        outcome_col (str): Name of the outcome column

    Returns:
        list: List of confounders' column names
    """
    return [
        col
        for col in df.columns
        if col not in [treatment_col, outcome_col, "random", "index"]
    ]


def codec_score(estimate, df: pd.DataFrame) -> float:
    """
    Calculate the CODEC score for the effect of treatment on y_factual.

    Args:
        estimate: Causal estimate to evaluate
        df (pd.DataFrame): input dataframe

    Returns:
        float: CODEC score
    """
    est = estimate.estimator
    treatment_name = (
        est._treatment_name
        if isinstance(est._treatment_name, str)
        else est._treatment_name[0]
    )
    outcome_name = est._outcome_name
    confounders = identify_confounders(df, treatment_name, outcome_name)

    cate_est = est.effect(df)
    standard_deviations = np.std(cate_est)

    df = df.copy()
    df["dy"] = est.effect_tt(df)
    df["yhat"] = df[est._outcome_name] - df["dy"]

    # Use corrected y, not y factual to get the estimators contribution
    Y = df["yhat"]
    Z = df[treatment_name]
    X = df[confounders]

    if standard_deviations < 0.01:
        return np.inf

    return codec(Y, Z, X)
