import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
import dagger
import matplotlib.pyplot as plt
import seaborn as sns
from bounds import hb_p_value
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
import networkx as nx
import itertools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
import pickle
from matplotlib.colors import Normalize, LinearSegmentedColormap
from sklearn.metrics import recall_score
from scipy.optimize import minimize

def calculate_all_p_values(calib_tables, alphas):
    n = calib_tables.shape[1]
    # Get p-values for each loss
    r_hats = calib_tables.mean(axis=1) # empirical risk at each lambda combination
    p_values = np.zeros_like(r_hats)
    for r in range(p_values.shape[0]):
        p_values[r] = hb_p_value(r_hats[r], n, alphas)

    return p_values


def calculate_corrected_p_values(calib_tables, alphas):
    # Combine them
    p_values = calculate_all_p_values(calib_tables, alphas)
    return p_values


def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any((costs[:i]) >= c, axis=1)) and np.all(np.any((costs[i + 1:]) >= c, axis=1))
    return is_efficient

def compute_scores_bt_model(p_vals, n_p=0, eta=None):
    """
    Compute Bradley-Terry model scores for hyperparameters using p-values.

    Parameters:
        p_vals (numpy.ndarray): Array of p-values (p_OPT^lambda_i).
        n_p (float): Pseudocount weight for prior observations. Default is 0.
        eta (numpy.ndarray): Prior probabilities (eta_ij) matrix of shape (n, n),
                             where n is the number of hyperparameters.
                             Default is None, meaning no prior information.

    Returns:
        numpy.ndarray: Array of scores (s(lambda)) for each hyperparameter.
    """
    # Number of hyperparameters
    n = len(p_vals)

    # If no prior is given, initialize eta to uniform prior
    if eta is None:
        eta = np.ones((n, n)) * 0.5
        np.fill_diagonal(eta, 0)  # No self-comparison

    # Compute pairwise weights w_ij
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                epsilon = 1e-9  # Small constant to avoid division by zero
                denominator = p_vals[i] + p_vals[j] + epsilon
                w[i, j] = n * (p_vals[i] / denominator) + n_p * eta[i, j]

    # Define the log-likelihood function to maximize
    def log_likelihood(scores):
        likelihood = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    epsilon = 1e-9  # Small constant to avoid division by zero
                    numerator = scores[i] + epsilon
                    denominator = scores[i] + scores[j] + epsilon

                    # Ensure both numerator and denominator are positive
                    if numerator <= 0 or denominator <= 0:
                        #print(
                        #    f"Skipping pair ({i}, {j}) due to invalid log argument. numerator: {numerator}, denominator: {denominator}")
                        continue

                    likelihood += w[i, j] * np.log(numerator / denominator)
        return -likelihood  # Negative because we minimize in scipy

    # Initial guess for scores (uniform)
    initial_scores = np.maximum(p_vals, 1e-6)

    # Add constraints to ensure positive scores
    constraints = ({'type': 'ineq', 'fun': lambda x: x})  # x > 0

    # Optimize the log-likelihood function
    result = minimize(log_likelihood, initial_scores, constraints=constraints, method='SLSQP', options={'maxiter': 100000})

    if result.success:
        return result.x  # Optimized scores
    else:
        return np.zeros(n)


def create_eta_array(p_vals, group_size=4):
    """
    Create the eta array based on p-values and their alpha grouping.

    Parameters:
        p_vals (numpy.ndarray): Array of p-values structured such that every `group_size` consecutive values share the same alpha.
        group_size (int): Number of consecutive p-values sharing the same alpha.

    Returns:
        numpy.ndarray: Eta array of shape (len(p_vals), len(p_vals)).
    """
    n = len(p_vals)
    eta = np.zeros((n, n))

    # Assign alpha group indices for each p-value
    alpha_groups = np.repeat(np.arange(n // group_size), group_size)

    for i in range(n):
        for j in range(n):
            if i != j:
                if alpha_groups[i] < alpha_groups[j]:
                    eta[i, j] = 1  # i is more reliable than j
                elif alpha_groups[i] > alpha_groups[j]:
                    eta[i, j] = 0  # j is more reliable than i
                # Otherwise, eta[i, j] remains 0 (same group)

    return eta


def create_DAG(data, p_vals, n_clusters=20, method='regression', L=1, alpha=0.5):
    n, m = data.shape  # n: number of nodes, m: number of features
    # Step 1: Transform p-values to make lower ones more separable
    def transform_p_values(p_vals, alpha):
        #transformed_p_vals = np.where(p_vals <= alpha, -np.log(p_vals), p_vals)
        p_vals = np.asarray(p_vals, dtype=np.float64)
        alpha = np.float64(alpha)  # Ensure alpha is a float
        transformed_p_vals = -np.log(p_vals/alpha)
        #transformed_p_vals = np.power(np.exp(1), -p_vals/alpha)
        #transformed_p_vals = np.where(p_vals <= alpha, np.power(np.exp(1), -p_vals), p_vals)
        return transformed_p_vals

    eta = create_eta_array(p_vals, 4)
    print(p_vals)
    transformed_p_vals = compute_scores_bt_model(p_vals, 1, eta)

    # Step 2: Cluster the nodes based on transformed p_vals using Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(transformed_p_vals.reshape(-1, 1))
    cluster_labels = clustering.labels_

    # Step 3: Sort clusters by the average p-values within each cluster
    cluster_p_vals = [np.mean(p_vals[cluster_labels == i]) for i in range(n_clusters)]
    sorted_clusters = np.argsort(cluster_p_vals)  # Sorting clusters by average p-values
    unique_clusters = np.unique(cluster_labels)

    # Step 4: Create adjacency matrix
    adj_matrix = np.zeros((n, n))  # Initialize adjacency matrix

    # Step 5: Form connections based on method chosen (regression or classification)
    for cluster_idx in range(1, len(unique_clusters)):  # Start from the second cluster (index 1)
        # Get nodes in the current cluster (upper cluster)
        upper_cluster = np.where(cluster_labels == sorted_clusters[cluster_idx])[0]

        # Determine how many previous levels to look at (limited by L)
        start_level = max(0, cluster_idx - L)
        lower_clusters = np.where(np.isin(cluster_labels, sorted_clusters[start_level:cluster_idx]))[0]

        if len(lower_clusters) == 0:
            continue  # Skip if there are no previous layers to connect to

        # Stack the data from all lower clusters to create the feature matrix
        X = data[lower_clusters].T  # Shape (m, num_lower_nodes), each column is a node's data
        # Track which lower-level nodes get connected
        lower_nodes_connected = np.zeros(len(lower_clusters), dtype=bool)

        # For each node in the upper cluster, connect to nodes in the selected previous clusters
        for upper_node in upper_cluster:
            current_node_data = data[upper_node]  # Data from the current upper node

            if method == 'regression':
                # Regression approach: Lasso with positive coefficients
                model = Lasso(alpha=0.1, positive=True)  # Lasso regression with positive coefficients
                model.fit(X, current_node_data)

                # Feature selection: keep features with coefficients larger than 0.5 (or change threshold)
                selected_features = np.where(model.coef_ > 0)[0]
                if len(selected_features) > 0:
                    adj_matrix[lower_clusters[selected_features], upper_node] = 1
                    lower_nodes_connected[selected_features] = True

            elif method == 'classification':
                # Classification approach: Logistic regression with positive coefficients
                y = (current_node_data > alpha).astype(int)  # Binary target based on alpha threshold

                # Check if y has both 0s and 1s; if not, skip logistic regression
                if len(np.unique(y)) < 2:
                    if np.all(y == 1):
                        adj_matrix[lower_clusters, upper_node] = 1
                        lower_nodes_connected[:] = True  # All nodes are connected
                    continue

                # Apply logistic regression with L1 regularization and positive coefficients
                model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
                model.fit(X, y)

                # Feature selection: Select features with coefficients larger than 1
                selected_features = np.where(np.abs(model.coef_)[0] > 1)[0]
                if len(selected_features) > 0:
                    adj_matrix[lower_clusters[selected_features], upper_node] = 1
                    lower_nodes_connected[selected_features] = True


    return adj_matrix


def Pareto_testing(R1, R2, alpha, delta, num_calib, length):
    calib1 = R1[:, :num_calib]
    calib2 = R1[:, num_calib:]
    calib1_mean = calib1.mean(axis=1).reshape(length, 1)
    R2_mean = R2.mean(axis=1).reshape(length, 1)
    to_pareto = np.hstack((calib1_mean, R2_mean))
    is_efficient = is_pareto(to_pareto)
    all_ids = np.arange(calib1_mean.shape[0])
    efficient_ids = all_ids[is_efficient]
    p_vals = calculate_corrected_p_values(calib1, alpha)
    p_vals = p_vals[efficient_ids]
    p_vals_cal2 = calculate_corrected_p_values(calib2, alpha)
    p_vals_cal2 = p_vals_cal2[efficient_ids]
    adjacency_matrix = create_DAG(calib1[efficient_ids, :], p_vals, alpha=delta, method='regression', n_clusters= 5)
    list_rejected_temp = dagger.DAGGER(adjacency_matrix, p_vals_cal2, delta)
    list_rejected_temp = list_rejected_temp[0]
    list_rejected = [i for i, val in enumerate(list_rejected_temp) if val]
    list_rejected = efficient_ids[list_rejected]
    return list_rejected, adjacency_matrix, efficient_ids