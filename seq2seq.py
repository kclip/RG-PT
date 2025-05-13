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


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)





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
    #efficient_ids = all_ids[is_efficient]
    efficient_ids = all_ids #Update later
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


def create_sequential_matrix(sorted_ids):
    n = len(sorted_ids)
    # Initialize an empty adjacency matrix
    adj_1d = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj_1d[sorted_ids[i], sorted_ids[i + 1]] = 1
    return adj_1d
def sequential_testing(R1, R2, alpha, delta, num_calib, length):
    calib1 = R1[:, :num_calib]
    calib2 = R1[:, num_calib:]
    calib1_mean = calib1.mean(axis=1).reshape(length, 1)
    R2_mean = R2.mean(axis=1).reshape(length, 1)
    to_pareto = np.hstack((calib1_mean, R2_mean))
    is_efficient = is_pareto(to_pareto)
    all_ids = np.arange(calib1_mean.shape[0])
    #efficient_ids = all_ids[is_efficient]
    efficient_ids = all_ids #Update later
    p_vals = calculate_corrected_p_values(calib1, alpha)
    p_vals = p_vals[efficient_ids]
    p_vals_cal2 = calculate_corrected_p_values(calib2, alpha)
    p_vals_cal2 = p_vals_cal2[efficient_ids]
    n, m = calib1[efficient_ids, :].shape
    efficent_sorted = np.array([2, 3, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    sequential_matrix = create_sequential_matrix(efficent_sorted)
    adjacency_matrix = np.zeros((n, n))
    list_rejected_temp = dagger.DAGGER(sequential_matrix, p_vals_cal2, delta)
    list_rejected_temp = list_rejected_temp[0]
    list_rejected = [i for i, val in enumerate(list_rejected_temp) if val]
    list_rejected = efficient_ids[list_rejected]
    efficent_sorted = np.argsort(p_vals_cal2)
    efficent_sorted = np.array([2, 3, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    return list_rejected, sequential_matrix, efficient_ids



def ltt_testing(R1, R2, alpha, delta, num_calib, length):
    calib1 = R1[:, :num_calib]
    calib2 = R1[:, num_calib:]
    calib1_mean = calib1.mean(axis=1).reshape(length, 1)
    R2_mean = R2.mean(axis=1).reshape(length, 1)
    to_pareto = np.hstack((calib1_mean, R2_mean))
    is_efficient = is_pareto(to_pareto)
    all_ids = np.arange(calib1_mean.shape[0])
    #efficient_ids = all_ids[is_efficient]
    efficient_ids = all_ids #Update later
    p_vals = calculate_corrected_p_values(calib1, alpha)
    p_vals = p_vals[efficient_ids]
    p_vals_cal2 = calculate_corrected_p_values(calib2, alpha)
    p_vals_cal2 = p_vals_cal2[efficient_ids]
    n, m = calib1[efficient_ids, :].shape
    adjacency_matrix = np.zeros((n, n))
    list_rejected_temp = dagger.DAGGER(adjacency_matrix, p_vals_cal2, delta)
    list_rejected_temp = list_rejected_temp[0]
    list_rejected = [i for i, val in enumerate(list_rejected_temp) if val]
    list_rejected = efficient_ids[list_rejected]
    efficent_sorted = np.argsort(p_vals_cal2)
    sequential_matrix = create_sequential_matrix(efficent_sorted)
    return list_rejected, adjacency_matrix, efficient_ids


def Pareto_testing_BH(calib1, calib2, alphas, delta):
    n_cal1 = calib1.shape[0]
    n_cal2 = calib2.shape[0]
    calib1_mean = calib1.mean(axis=0).squeeze().flatten(start_dim=1)  # empirical risk at each lambda combination
    calib1_mean = calib1_mean.T
    calib1_mean = calib1_mean.numpy()
    is_efficient = is_pareto(calib1_mean)
    all_ids = np.arange(calib1_mean.shape[0])
    efficient_ids = all_ids[is_efficient]
    p_vals = calculate_corrected_p_values(calib1, alphas)
    p_vals = p_vals[efficient_ids]
    p_vals_cal2 = calculate_corrected_p_values(calib2, alphas)
    p_vals_cal2 = p_vals_cal2[efficient_ids]
    calib1 = calib1.view(calib1.shape[0], 3, -1)
    calib1 = calib1.numpy()
    calib1 = calib1[:, 1, :]  # Any dimension can be picked
    calib1 = calib1.T
    n, m = calib1[efficient_ids, :].shape
    adjacency_matrix = np.zeros((n, n))
    list_rejected_temp = dagger.DAGGER(adjacency_matrix, p_vals_cal2, delta)
    list_rejected_temp = list_rejected_temp[0]
    list_rejected = [i for i, val in enumerate(list_rejected_temp) if val]
    lr_toplot = list_rejected
    max_index = max(range(len(list_rejected)), key=lambda i: p_vals[list_rejected[i]])
    list_rejected = efficient_ids[list_rejected]
    return list_rejected


def plot(df_list, alphas, methods):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    recalls = []
    mious = []
    mcvgs = []
    labels = []
    for i in range(len(df_list)):
        method = methods[i]
        if method == "Split Fixed Sequence":
            method = "Split Fixed\nSequence"
        df = df_list[i]
        recalls = recalls + [df['recall'], ]
        mious = mious + [df['mIOU'], ]
        mcvgs = mcvgs + [df['mean coverage'], ]
        labels = labels + [method, ]
        violations = (df['mean coverage'] < (1 - alphas[0])) | (df['mIOU'] < (1 - alphas[1])) | (
                    df['recall'] < (1 - alphas[2]))
        print(f'{method}: fraction of violations is {violations.mean()}')

    sns.violinplot(data=recalls, ax=axs[0], orient='h', inner=None)
    sns.violinplot(data=mious, ax=axs[1], orient='h', inner=None)
    sns.violinplot(data=mcvgs, ax=axs[2], orient='h', inner=None)

    # Limits, lines, and labels
    axs[2].set_xlabel('Mean Coverage')
    axs[2].axvline(x=1 - alphas[0], c='#999999', linestyle='--', alpha=0.7)
    axs[2].locator_params(axis='x', nbins=4)
    axs[2].locator_params(axis='y', nbins=4)
    axs[1].set_xlabel('Mean IOU')
    axs[1].axvline(x=1 - alphas[1], c='#999999', linestyle='--', alpha=0.7)
    axs[1].locator_params(axis='x', nbins=4)
    axs[1].locator_params(axis='y', nbins=4)
    axs[0].set_xlabel('ROUGE-L')
    axs[0].axvline(x=1 - alphas[2], c='#999999', linestyle='--', alpha=0.7)
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].locator_params(axis='y', nbins=4)
    axs[0].set_yticklabels(labels)
    sns.despine(ax=axs[0], top=True, right=True)
    sns.despine(ax=axs[1], top=True, right=True)
    sns.despine(ax=axs[2], top=True, right=True)
    fig.tight_layout()
    os.makedirs("./outputs/histograms/", exist_ok=True)
    plt.savefig(
        "./" + f"outputs/histograms/detector_{alphas[0]}_{alphas[1]}_delta_histograms".replace(".", "_") + ".pdf")


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


def flatten_lambda_meshgrid(lambda1s, lambda2s, lambda3s):
    l1_meshgrid, l2_meshgrid, l3_meshgrid = torch.meshgrid(
        (torch.tensor(lambda1s), torch.tensor(lambda2s), torch.tensor(lambda3s)))
    l1_meshgrid = l1_meshgrid.flatten()
    l2_meshgrid = l2_meshgrid.flatten()
    l3_meshgrid = l3_meshgrid.flatten()
    return l1_meshgrid, l2_meshgrid, l3_meshgrid


def split_fixed_sequence(calib_tables, alphas, delta):
    # Split the data
    n_calib = calib_tables.shape[0]
    n_coarse = n_calib // 2
    perm = torch.randperm(n_calib)
    calib_tables = calib_tables[perm]
    coarse_tables, fine_tables = (calib_tables[:n_coarse], calib_tables[n_coarse:])
    p_values_coarse = calculate_all_p_values(coarse_tables, alphas)
    # Find a lambda for each value of beta that controls the risk best.
    num_betas = 200
    betas = np.logspace(-9, 0, num_betas)
    lambda_sequence = np.zeros_like(betas)
    for b in range(num_betas):
        beta = betas[b]
        differences = np.abs(p_values_coarse - beta)
        infty_norms = np.linalg.norm(differences, ord=np.inf, axis=0)
        lambda_sequence[b] = infty_norms.argmin()

    _, idx = np.unique(lambda_sequence, return_index=True)
    lambda_sequence_ordered = lambda_sequence[np.sort(idx)]

    # Now test these lambdas
    fine_tables = fine_tables.flatten(start_dim=2)[:, :, lambda_sequence_ordered]
    p_values_fine = calculate_corrected_p_values(fine_tables, alphas)
    rejections = lambda_sequence_ordered[np.nonzero(p_values_fine < delta)[0]].astype(int)

    return rejections


def bar_plot(R, L):
    indices = list(range(L))
    colors = ['green' if i in R else 'black' for i in indices]

    # Create a figure for the plot
    plt.figure(figsize=(10, 2))  # Adjust the size as needed

    # Plot using scatter (or you can use bar)
    plt.scatter(indices, [1] * L, color=colors, s=100)

    # Label the plot
    plt.title("Array Indices with Highlighted R Indices")
    plt.xlabel("Index")
    plt.yticks([])  # Remove y-axis ticks since they are unnecessary
    plt.show()


# Choose subset sizes
train_size = 500  # Number of training samples
test_size = 3000   # Number of test samples



# (X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()
#
# # Reshape if needed and normalize
# X_train_full = X_train_full.reshape(X_train_full.shape[0], -1) / 255.0
# X_test_full = X_test_full.reshape(X_test_full.shape[0], -1) / 255.0
#
#
#
# # Create a smaller training set
# X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=train_size, stratify=y_train_full, random_state=42)
#
# # Create a smaller test set
# X_test, _, y_test, _ = train_test_split(X_test_full, y_test_full, train_size=test_size, stratify=y_test_full, random_state=42)
#
# # Define the range of values for C and gamma
# C_values = np.logspace(-3, 3, 5)
# gamma_values = np.logspace(-4, 1, 5)
#
# # Initialize arrays to store accuracy and calibration (ECE) errors
# num_test_samples = X_test.shape[0]
# accuracy_errors = np.zeros((len(C_values)*len(gamma_values), num_test_samples))
# calibration_errors = np.zeros((len(C_values)*len(gamma_values), num_test_samples))
# length = len(C_values)*len(gamma_values)
#
# # Iterate over all combinations of C and gamma
# config_index = 0
# for C in C_values:
#     for gamma in gamma_values:
#         # Initialize and train the SVM with the given C and gamma
#         svm = SVC(C=C, gamma=gamma, probability=True, random_state=42)
#         svm.fit(X_train, y_train)
#
#         # Calibrate the SVM using CalibratedClassifierCV
#         calibrated_svm = CalibratedClassifierCV(svm, method="sigmoid")
#         calibrated_svm.fit(X_train, y_train)
#
#         # Predict probabilities on the test set
#         probs = calibrated_svm.predict_proba(X_test)
#         predictions = calibrated_svm.predict(X_test)
#
#         # Calculate accuracy error for each sample
#         for i, (pred, true) in enumerate(zip(predictions, y_test)):
#             accuracy_errors[config_index, i] = 0 if pred == true else 1
#
#         # Calculate Expected Calibration Error (ECE) for each sample using Brier score as proxy
#         for i, (prob, true) in enumerate(zip(probs, y_test)):
#             calibration_errors[config_index, i] = brier_score_loss(
#                 [1 if k == true else 0 for k in range(len(prob))], prob
#             )
#
#
#         for i, (prob, true) in enumerate(zip(probs, y_test)):
#             # Calculate confidence as the maximum predicted probability
#             confidence = np.max(prob)
#
#             # Determine if the prediction is correct
#             accuracy = 1 if np.argmax(prob) == true else 0
#
#             # Calculate the calibration error for this sample
#             calibration_error = abs(accuracy - confidence)
#
#             # Store the maximum calibration error encountered so far
#             calibration_errors[config_index, i] = max(calibration_errors[config_index, i - 1] if i > 0 else 0,
#                                                       calibration_error)
#
#         for i, (prob, true) in enumerate(zip(probs, y_test)): # ROUGE-L
#             # Get the predicted class as the one with the highest probability
#             predicted_class = prob.argmax()
#
#             # Update calibration_scores to store recall
#             # ROUGE-L is calculated as recall_score(y_true, y_pred, average='binary' or 'macro')
#             # For this case, assume we compute binary recall for the "true" class
#
#             # Binary one-hot vector for the true class
#             true_binary = [1 if k == true else 0 for k in range(len(prob))]
#
#             # Binary one-hot vector for the predicted class
#             predicted_binary = [1 if k == predicted_class else 0 for k in range(len(prob))]
#
#             # Calculate recall for this prediction (use binary=True for simplicity)
#             calibration_errors[config_index, i] = recall_score(
#                 true_binary, predicted_binary, average='binary', zero_division=0
#             )
#
#         # Move to the next configuration
#         config_index += 1
#         print(f"Configuration {config_index}/100 done: C={C}, gamma={gamma}")
# #
# #
# #
# #
# np.savez("processed_results_recall.npz",
#          accuracy_errors=accuracy_errors,
#          calibration_errors=calibration_errors,
#          C_values=C_values,
#          gamma_values=gamma_values)
#
# # Display the results


data = np.load("seq2seq_hyperparam_bleu_results_fine.npz", allow_pickle=True)
results = data["results"]

# Extract alphas, epsilons, and accuracy errors
C_values = np.unique([row[0] for row in results])  # Unique alpha values
gamma_values = np.unique([row[1] for row in results])  # Unique epsilon values

# Create an array where each row contains the BLEU scores for one hyperparameter combination
accuracy_errors = -np.array([row[2] for row in results], dtype=object)

# Example outputs
print("Alphas:", C_values)
print("Epsilons:", gamma_values)
print("Accuracy Errors (BLEU scores):", accuracy_errors)
length = len(C_values)*len(gamma_values)
print(accuracy_errors.shape)


data2 = np.load("seq2seq_hyperparam_rouge_results_fine.npz", allow_pickle=True)
results2 = data2["results"]
calibration_errors = -np.array([row[2] for row in results2], dtype=object)





num_calib = accuracy_errors.shape[1]//2
# Define the alpha range
alphas = np.linspace(-20, -14, 50)

# Initialize lists to store maximum calibration errors for each method
rgpt_errors = []
pt_errors = []
ltt_errors = []
z = calibration_errors.mean(axis = 1)

# Loop over each alpha
for alpha in alphas:
    print("ALPHAS:", alpha)
    # RG-PT method
    lr_toplot, adjacency_matrix, efficient_ids = Pareto_testing(
        accuracy_errors[:, :200], calibration_errors[:, :200], alpha, 0.1, num_calib, length
    )
    print("RG-PT:", lr_toplot)
    if lr_toplot.size > 0:
        rgpt_errors.append(-np.min(z[lr_toplot]))
    else:
        rgpt_errors.append(None)

    # PT method
    lr_toplot, adjacency_matrix, efficient_ids = sequential_testing(
        accuracy_errors[:, :200], calibration_errors[:, :200], alpha, 0.1, num_calib, length
    )
    print("PT:", lr_toplot)
    if lr_toplot.size > 0:
        pt_errors.append(-np.min(z[lr_toplot]))
    else:
        pt_errors.append(None)

    # LTT method
    lr_toplot, adjacency_matrix, efficient_ids = ltt_testing(
        accuracy_errors[:, :200], calibration_errors[:, :200], alpha, 0.1, num_calib, length
    )
    print("LTT:", lr_toplot)
    if lr_toplot.size > 0:
        ltt_errors.append(-np.min(z[lr_toplot]))
    else:
        ltt_errors.append(None)

# Plot the results
plt.rcParams['text.usetex'] = True

# Set font family and font size
plt.rcParams['font.family'] = 'serif'    # Options: 'serif', 'sans-serif', 'monospace'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(10, 6))
plt.plot(-alphas, rgpt_errors, label='RG-PT', marker='o', linestyle='-', markerfacecolor='blue', markeredgecolor='blue')
plt.plot(-alphas, pt_errors, label='PT', marker='s', linestyle='-', markerfacecolor='orange', markeredgecolor='orange')
plt.plot(-alphas, ltt_errors, label='LTT', marker='^', linestyle='-', markerfacecolor='green', markeredgecolor='green')
# Customize the plot
plt.xlabel(r'$\alpha$')
plt.ylabel('Achieved ROUGE-L score')
#plt.title('Maximum Calibration Error vs. Alpha', fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)

# Save and display the plot
plt.savefig("alpha_vs_calibration_error.pdf", format="pdf")
plt.show()