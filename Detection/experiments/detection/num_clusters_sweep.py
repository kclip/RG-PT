import os, sys, inspect

sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, gc, time, json, cv2, random, sys, traceback
from experiments.detection.utils import *
from core.bounds import hb_p_value
from core.concentration import *

import multiprocessing as mp

import pickle as pkl
from tqdm import tqdm
import pdb
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
import dagger
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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


def create_DAG(data, p_vals, n_clusters=20, method='regression', L=1, alpha=0.5):
    n, m = data.shape  # n: number of nodes, m: number of features

    # Step 1: Transform p-values to make lower ones more separable
    def transform_p_values(p_vals, alpha):
        # transformed_p_vals = np.where(p_vals <= alpha, -np.log(p_vals), p_vals)
        transformed_p_vals = -np.log(p_vals / alpha)
        # transformed_p_vals = np.power(np.exp(1), -p_vals/alpha)
        # transformed_p_vals = np.where(p_vals <= alpha, np.power(np.exp(1), -p_vals), p_vals)
        return transformed_p_vals

    transformed_p_vals = transform_p_values(p_vals, alpha)

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
                model = Lasso(alpha=0.01, positive=True)  # Lasso regression with positive coefficients
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


def Pareto_testing(calib1, calib2, alphas, delta, num_clusters):
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
    adjacency_matrix = create_DAG(calib1[efficient_ids, :], p_vals, n_clusters= num_clusters, alpha=np.min(alphas), method='regression')
    list_rejected_temp = dagger.DAGGER(adjacency_matrix, p_vals_cal2, delta)
    list_rejected_temp = list_rejected_temp[0]
    list_rejected = [i for i, val in enumerate(list_rejected_temp) if val]
    lr_toplot = list_rejected
    max_index = max(range(len(list_rejected)), key=lambda i: p_vals[list_rejected[i]])
    list_rejected = efficient_ids[list_rejected]
    return list_rejected


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
    axs[0].set_xlabel('Recall')
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
    n = calib_tables.shape[0]
    # Get p-values for each loss
    r_hats = calib_tables.mean(axis=0).squeeze().flatten(start_dim=1)  # empirical risk at each lambda combination
    p_values = np.zeros_like(r_hats)
    for r in range(p_values.shape[0]):
        for i in range(p_values.shape[1]):
            p_values[r, i] = hb_p_value(r_hats[r, i], n, alphas[r])

    return p_values


def calculate_corrected_p_values(calib_tables, alphas):
    # Combine them
    p_values = calculate_all_p_values(calib_tables, alphas)
    p_values_corrected = p_values.max(axis=0)
    return p_values_corrected


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


def trial(i, method, alphas, delta, lambda1s, lambda2s, lambda3s, l1_meshgrid, l2_meshgrid, l3_meshgrid, num_calib,
          loss_tables, risks, lhats, num_par, num_clusters):
    fix_randomness(seed=(i * 10000))
    n = loss_tables["tensor"].shape[0]
    perm = torch.randperm(n)

    local_tables = loss_tables["tensor"][perm]
    calib_tables, val_tables = (local_tables[:num_calib], local_tables[num_calib:])

    if method == "Bonferroni":
        p_values_corrected = calculate_corrected_p_values(calib_tables, alphas)
        R = bonferroni(p_values_corrected, delta)

    elif method == "Split Fixed Sequence":
        R = split_fixed_sequence(calib_tables, alphas, delta)
    elif method == "DAG":
        calib_tables1, calib_tables2 = (calib_tables[:num_par], calib_tables[num_par:])
        R = Pareto_testing(calib_tables1, calib_tables2, alphas, delta, num_clusters)
    elif method == "BH":
        calib_tables1, calib_tables2 = (calib_tables[:num_par], calib_tables[num_par:])
        R = Pareto_testing_BH(calib_tables1, calib_tables2, alphas, delta)

    if R.shape[0] == 0:
        lhats[i] = np.array([1.0, 1.0, 1.0])
        risks[i] = np.array([0.0, 0.0, 0.0])
        loss_tables["curr_proc"] -= 1

    # Index the lambdas
    l1s = l1_meshgrid[R]
    l2s = l2_meshgrid[R]
    l3s = l3_meshgrid[R]

    global l1_DAG, l2_DAG, l3_DAG, l1_BH, l2_BH, l3_BH

    if method == "DAG":
        l1_DAG.append(l1s.tolist())
        l2_DAG.append(l2s.tolist())
        l3_DAG.append(l3s.tolist())
    if method == "BH":
        l1_BH.append(l1s.tolist())
        l2_BH.append(l2s.tolist())
        l3_BH.append(l3s.tolist())

    l3 = l3s[l3s > l1s].min()
    l2 = l2s[(l3s > l1s) & (l3s == l3)].median()
    l1 = l1s[(l3s > l1s) & (l2s == l2) & (l3s == l3)].min()

    lhats[i] = np.array([l1, l2, l3])

    # Validate

    idx1 = torch.nonzero(np.abs(lambda1s - lhats[i][0]) < 1e-10)[0][0].item()
    idx2 = torch.nonzero(np.abs(lambda2s - lhats[i][1]) < 1e-10)[0][0].item()
    idx3 = torch.nonzero(np.abs(lambda3s - lhats[i][2]) < 1e-10)[0][0].item()

    risks[i] = val_tables[:, :, idx1, idx2, idx3].mean(dim=0)
    loss_tables["curr_proc"] -= 1
    del calib_tables
    del val_tables
    del local_tables
    gc.collect()


l1_DAG = []
l2_DAG = []
l3_DAG = []

l1_BH = []
l2_BH = []
l3_BH = []

if __name__ == "__main__":
    sns.set(palette='pastel', font='serif')
    sns.set_style('white')
    num_trials = 200
    num_calib = 3000
    num_par = 1500
    num_processes = 30
    mp.set_start_method('fork')
    alphas = [0.25, 0.5, 0.5]  # neg_m_coverage, neg_miou, neg_recall
    delta = 0.1
    lambda1s = torch.linspace(0.5, 1, 10)  # Top score threshold
    lambda2s = torch.linspace(0, 1, 10)  # Segmentation threshold
    lambda3s = torch.tensor([0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 1])

    # Multiprocessing setup
    manager = mp.Manager()
    loss_tables = manager.dict({"tensor": None, "curr_proc": 0})

    df_list = []
    methods = ["DAG"]
    L1 = []
    L2 = []
    L3 = []
    for num_clusters in range(1, 50):
        for method in methods:
            fname = f'{method}_{alphas}_{delta}_{num_calib}_{num_trials}_dataframe.pkl'
            try:
                df = pd.read_pickle(fname)
            except FileNotFoundError:
                with torch.no_grad():
                    # Load cache
                    with open('example_loss_tables.pt', 'rb') as f:
                        # loss_tables["tensor"] = torch.tensor(np.random.random(size=(num_calib*2,3,lambda1s.shape[0],lambda2s.shape[0],lambda3s.shape[0])))/10
                        loss_tables["tensor"] = torch.load(f)

                    risks = manager.dict({k: np.zeros((3,)) for k in range(num_trials)})
                    lhats = manager.dict({k: np.zeros((3,)) for k in range(num_trials)})
                    l1_meshgrid, l2_meshgrid, l3_meshgrid = flatten_lambda_meshgrid(lambda1s, lambda2s, lambda3s)

                    # Test trial
                    # trial(0, method, alphas, delta, lambda1s, lambda2s, lambda3s, l1_meshgrid, l2_meshgrid, l3_meshgrid, num_calib, loss_tables, risks, lhats)
                    # Queue the jobs
                    jobs = []
                    for i in range(num_trials):
                        p = mp.Process(target=trial, args=(
                        i, method, alphas, delta, lambda1s, lambda2s, lambda3s, l1_meshgrid, l2_meshgrid, l3_meshgrid,
                        num_calib, loss_tables, risks, lhats, num_par, num_clusters))
                        jobs.append(p)

                    pbar = tqdm(total=num_trials)

                    # Run the jobs
                    for proc in jobs:
                        while loss_tables["curr_proc"] >= num_processes:
                            time.sleep(2)
                        proc.start()
                        loss_tables["curr_proc"] += 1
                        pbar.update(1)

                    pbar.close()

                    for proc in jobs:
                        proc.join()

                    # Form the large dataframe
                    local_df_list = []
                    for i in tqdm(range(num_trials)):
                        dict_local = {"$\\hat{\\lambda}$": [lhats[i], ],
                                      "mean coverage": 1 - risks[i][0].item(),
                                      "mIOU": 1 - risks[i][1].item(),
                                      "recall": 1 - risks[i][2].item(),
                                      "alpha1": alphas[0],
                                      "alpha2": alphas[1],
                                      "alpha3": alphas[2],
                                      "delta": delta,
                                      "index": [0],
                                      }
                        df_local = pd.DataFrame(dict_local)
                        local_df_list = local_df_list + [df_local]
                    df = pd.concat(local_df_list, axis=0, ignore_index=True)


            df_list = df_list + [df, ]
            average_lambda = np.concatenate([arr[None, :] for arr in df["$\\hat{\\lambda}$"].tolist()], axis=0).mean(axis=0)
            L1.append(average_lambda[0])
            L2.append(average_lambda[1])
            L3.append(average_lambda[2])
            print(f"{method}: the average lambda_hat from the runs was: {list(average_lambda)}!")

    x_values = range(1, 50)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, L1, marker='o', label="L1")
    plt.plot(x_values, L2, marker='s', label="L2")
    plt.plot(x_values, L3, marker='^', label="L3")

    # Adding labels and title
    plt.xlabel("Range")
    plt.ylabel("Values")
    plt.title("Plot of L1, L2, and L3 against Range(1,5)")
    plt.legend()

    # Display the plot
    plt.show()
    print("Done!")
