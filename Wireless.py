import numpy as np
import os
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import itertools
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import ast
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde
import math
import dagger
import networkx as nx
import statsmodels.stats.multitest as smm
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

My = 2
Mt = 16
n_testing = 300000000
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'

import torch

from bounds import hb_p_value
from data_utils.dataset import max_seq_length, n_test, n_cals, n_cals1


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


def fixed_sequence_testing(h_sorted, p_vals):
    list_rejected = []
    for b in range(len(h_sorted)):
        xx = h_sorted[b]
        if p_vals[xx] < args.delta:
            list_rejected.append(xx)
        else:
            break

    return list_rejected


def discretize_2d_array(array, n_bins=4):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    n_samples, n_features = array.shape
    discretized_array = np.zeros((n_samples, n_features), dtype=int)

    for i in range(n_features):
        discretized_array[:, i] = discretizer.fit_transform(array[:, i].reshape(-1, 1)).flatten()

    return discretized_array

def topological_sort(adj_matrix):
    """
    Sort nodes to topological order.
    Output: an array with sorted indices of nodes.
    """
    g = nx.from_numpy_array(adj_matrix, create_using=nx.MultiDiGraph())
    sorted_inds = nx.topological_sort(g)
    sorted_inds = list(sorted_inds)
    adj = adj_matrix[sorted_inds][:, sorted_inds]
    return sorted_inds


def plot_directed_graph(adj_matrix, list_rejected, node_labels):
    # Perform topological sort
    sorted_inds = topological_sort(adj_matrix)

    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Generate positions with horizontal spacing for nodes in the same row
    pos = {}
    y_spacing = 1  # Control vertical spacing between layers
    for i, node in enumerate(sorted_inds[::-1]):  # Reverse sorted indices for bottom to top
        layer = i  # Layer based on topological rank

        # Use node label (the original node index) to determine horizontal position
        x_position = node  # You can scale or modify this if you want
        y_position = layer * y_spacing  # Vertical position based on topological order

        pos[node] = (x_position, y_position)

    # Create a color map: green for rejected nodes, red for others
    color_map = []
    for node in G.nodes():
        if node in list_rejected:
            color_map.append('green')
        else:
            color_map.append('red')

    label_mapping = {node: node_labels[node] for node in G.nodes()}
    # Draw nodes with labels and colors
    nx.draw(G, pos, labels = label_mapping, node_color=color_map, with_labels=True, node_size=700, font_size=10, arrows=True, arrowstyle='-|>',
            arrowsize=12)

    # Draw edges (with curvature to avoid overlap if necessary)
    arc_rad = 0.25  # Adjust curvature for bidirectional edges
    for (u, v) in G.edges():
        if G.has_edge(v, u):  # If there's an edge in the opposite direction
            rad = arc_rad
        else:
            rad = 0  # Straight line for unidirectional edges
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rad}', arrows=True,
                               arrowstyle='-|>', arrowsize=12)

    # Show the plot
    plt.show()


def create_DAG(data, p_vals, n_clusters=20, method='regression', L=1, alpha=0.5):
    n, m = data.shape  # n: number of nodes, m: number of features

    # Step 1: Transform p-values to make lower ones more separable
    def transform_p_values(p_vals, alpha):
        #transformed_p_vals = np.where(p_vals <= alpha, -np.log(p_vals), p_vals)
        transformed_p_vals = -np.log(p_vals/alpha)
        #transformed_p_vals = np.power(np.exp(1), -p_vals/alpha)
        #transformed_p_vals = np.where(p_vals <= alpha, np.power(np.exp(1), -p_vals), p_vals)
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
                model = Lasso(alpha=0.1, positive=True)  # Lasso regression with positive coefficients
                model.fit(X, current_node_data)

                # Feature selection: keep features with coefficients larger than 0.5 (or change threshold)
                selected_features = np.where(model.coef_ > 0.2)[0]
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


def Pareto_testing(calib1, calib2, alpha, delta):
    n_cal1 = calib1.shape[1]
    n_cal2 = calib2.shape[1]
    calib1_mean = calib1.mean(-1)
    calib2_mean = calib2.mean(-1)
    is_efficient = is_pareto(calib1_mean.reshape(-1, 1))
    all_ids = np.arange(calib1.shape[0])
    efficient_ids = all_ids[is_efficient]
    p_vals = hb_p_value(calib1_mean[efficient_ids].reshape(-1), n_cal1, alpha)
    p_vals_cal2 = hb_p_value(calib2_mean[efficient_ids].reshape(-1), n_cal2, alpha)
    adjacency_matrix = create_DAG(calib1[efficient_ids, :], p_vals, alpha=alpha, method='regression')
    list_rejected_temp = dagger.DAGGER(adjacency_matrix, p_vals_cal2, delta)
    list_rejected_temp = list_rejected_temp[0]
    list_rejected = [i for i, val in enumerate(list_rejected_temp) if val]
    lr_toplot = list_rejected
    max_index = max(range(len(list_rejected)), key=lambda i: p_vals[list_rejected[i]])
    list_rejected = efficient_ids[list_rejected]
    return  efficient_ids[max_index]



def main(args):
    plt.close('all')
    D0 = []
    with open('delays0.csv', 'r') as file:
        reader = csv.reader(file)
        # Read each row from the CSV file and append it to the list
        for row in reader:
            # Convert each value in the row from string to integer
            int_row = [float(value) for value in row]
            D0.append(int_row)
    D0 = np.array(D0)
    D1 = []
    with open('delays1.csv', 'r') as file:
        reader = csv.reader(file)
        # Read each row from the CSV file and append it to the list
        for row in reader:
            # Convert each value in the row from string to integer
            int_row = [float(value) for value in row]
            D1.append(int_row)
    D1 = np.array(D1)
    D2 = []
    with open('delays2.csv', 'r') as file:
        reader = csv.reader(file)
        # Read each row from the CSV file and append it to the list
        for row in reader:
            # Convert each value in the row from string to integer
            int_row = [float(value) for value in row]
            D2.append(int_row)
    D2 = np.array(D2)
    D3 = []
    with open('delays3.csv', 'r') as file:
        reader = csv.reader(file)
        # Read each row from the CSV file and append it to the list
        for row in reader:
            # Convert each value in the row from string to integer
            int_row = [float(value) for value in row]
            D3.append(int_row)
    D3 = np.array(D3)
    D0 = D0.T
    D1 = D1.T
    D2 = D2.T
    D3 = D3.T
    alpha = 2.07705283
    beta = 7.74421717
    gamma = 0.0507541467
    mu = -0.00151544198
    scale_factor = 1
    steps_alpha = 1
    steps_beta = 1
    steps_gamma = 10
    steps_mu = 10
    alphas = np.linspace(alpha / scale_factor, alpha * scale_factor, steps_alpha)
    betas = np.linspace(beta / scale_factor, beta * scale_factor, steps_beta)
    gammas = np.linspace(gamma / scale_factor, gamma * scale_factor, steps_gamma)
    mus = np.linspace(mu / scale_factor, mu * scale_factor, steps_mu)
    gammas = np.linspace(0.02, 0.2, steps_gamma)
    mus = np.linspace(-0.1, 0.1, steps_mu)
    lambdas = itertools.product(alphas, betas, gammas, mus)
    lambdas = [list(item) for item in lambdas]
    lambdas = np.array(lambdas)
    data_len = D3.shape[1]
    n_cal = args.n_cal
    n_cal1 = args.n_cal1
    n_cal2 = n_cal - n_cal1
    alphas = [float(f) for f in args.alphas.split(',')]
    D_DAG = []
    D_normal = []
    lambdas_DAG = []
    lambdas_normal = []


    for t in tqdm(range(args.n_trials)):

        all_idx = np.arange(data_len)
        np.random.shuffle(all_idx)
        cal_idx = all_idx[:n_cal]
        test_idx = all_idx[n_cal:]

        D0_cal = D0[:, cal_idx]
        D1_cal = D1[:, cal_idx]
        D2_cal = D2[:, cal_idx]
        D3_cal = D3[:, cal_idx]

        D0_cal1 = D0_cal[:, :n_cal1].mean(-1)
        D1_cal1 = D1_cal[:, :n_cal1].mean(-1)
        D2_cal1 = D2_cal[:, :n_cal1].mean(-1)
        D3_cal1 = D3_cal[:, :n_cal1].mean(-1)

        D0_cal2 = D0_cal[:, n_cal1:].mean(-1)
        D1_cal2 = D1_cal[:, n_cal1:].mean(-1)
        D2_cal2 = D2_cal[:, n_cal1:].mean(-1)
        D3_cal2 = D3_cal[:, n_cal1:].mean(-1)

        D0_test = D0[:, test_idx].mean(-1)
        D1_test = D1[:, test_idx].mean(-1)
        D2_test = D2[:, test_idx].mean(-1)
        D3_test = D3[:, test_idx].mean(-1)

        ##########################################
        ############# Pareto Frontier ############
        ##########################################

        D0p = D0_cal1.reshape(-1, 1)
        D1p = D1_cal1.reshape(-1, 1)
        D2p = D2_cal1.reshape(-1, 1)
        D3p = D3_cal1.reshape(-1, 1)
        utilities = np.hstack((D0p, D1p, D2p, D3p))

        is_efficient = is_pareto(utilities)
        all_ids = np.arange(D0p.shape[0])
        efficient_ids = all_ids[is_efficient]
        # Sort according to p-values


        for a, alpha in enumerate(alphas):

            p_vals = hb_p_value(D3_cal1[efficient_ids].reshape(-1), n_cal1, alpha)
            p_vals_cal2 = hb_p_value(D3_cal2[efficient_ids].reshape(-1), n_cal2, alpha)


            p_vals2 = p_vals
            efficent_sorted = np.argsort(p_vals2)

            def create_sequential_matrix(sorted_ids):
                n = len(sorted_ids)
                # Initialize an empty adjacency matrix
                adj_1d = np.zeros((n, n), dtype=int)
                for i in range(n - 1):
                    adj_1d[sorted_ids[i], sorted_ids[i + 1]] = 1
                return adj_1d

            adjacency_matrix = create_DAG(D3_cal[efficient_ids, :], p_vals, alpha = alpha, method='regression')
            list_rejected_temp = dagger.DAGGER(adjacency_matrix, p_vals_cal2, args.delta)
            list_rejected_temp = list_rejected_temp[0]
            list_rejected = [i for i, val in enumerate(list_rejected_temp) if val]
            lr_toplot = list_rejected
            list_rejected = efficient_ids[list_rejected]

            sequential_matrix = create_sequential_matrix(efficent_sorted)
            n = len(efficent_sorted)
            list_rejected_temp2 = dagger.DAGGER(np.zeros((n, n), dtype=int), p_vals_cal2, args.delta)
            list_rejected_temp2 = list_rejected_temp2[0]
            list_rejected2 = [i for i, val in enumerate(list_rejected_temp2) if val]
            lrn_toplot = list_rejected2
            list_rejected_normal = efficient_ids[list_rejected2]



            ##########################################
            ######## Select ##########
            ##########################################
            score = [-D1_cal2[id_rej - 1] for id_rej in list_rejected]
            score_normal = [-D1_cal2[id_rej - 1] for id_rej in list_rejected_normal]
            if len(score) > 0 & len(score_normal) > 0:
                id = score.index(max(score))
                id_max_score = list_rejected[id] - 1
                id = score_normal.index(max(score_normal))
                id_max_score_normal = list_rejected_normal[id] - 1
                D_DAG += [[D0_test[id_max_score], D1_test[id_max_score], D2_test[id_max_score], D3_test[id_max_score]]]
                D_normal += [[D0_test[id_max_score_normal], D1_test[id_max_score_normal], D2_test[id_max_score_normal], D3_test[id_max_score_normal]]]
                lambdas_DAG.append(lambdas[id_max_score])
                lambdas_normal.append(lambdas[id_max_score_normal])
        if t==0:
            fixed_lambda_0 = lambdas[0, 0]
            fixed_lambda_3 = lambdas[0, 1]

            # Filtering lambdas for the found fixed values
            filtered_lambdas = lambdas[(lambdas[:, 0] == fixed_lambda_0) & (lambdas[:, 1] == fixed_lambda_3)]

            # Extract x, y, and z
            x = filtered_lambdas[:, 3]
            y = filtered_lambdas[:, 2]
            z1 = -D0_test[(lambdas[:, 0] == fixed_lambda_0) & (lambdas[:, 1] == fixed_lambda_3)]
            z2 = -D2_test[(lambdas[:, 0] == fixed_lambda_0) & (lambdas[:, 1] == fixed_lambda_3)]
            z = z2

            # Create a meshgrid for plotting
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            X, Y = np.meshgrid(x_unique, y_unique)

            # We need to reshape z accordingly to match the meshgrid shape
            Z = np.zeros_like(X)
            for i in range(len(x)):
                x_index = np.where(x_unique == x[i])[0][0]
                y_index = np.where(y_unique == y[i])[0][0]
                Z[y_index, x_index] = z[i]

            # Plotting the 3D surface
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))

            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha = 0.5)

            # Customize the z axis
            ax.set_zlim(np.min(Z), np.max(Z))
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter('{x:.02f}')

            # Add a color bar which maps values to colors
            fig.colorbar(surf, shrink=0.5, aspect=5)

            # Create and plot the graph
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

            # Extract 3D positions of the efficient nodes
            lambdas_chosen = lambdas[efficient_ids, :]
            pos = {i: [lambdas_chosen[i, 3], lambdas_chosen[i, 2], -D2_test[efficient_ids[i]]] for i in G.nodes}

            # Plot the edges of the graph in 3D space
            for edge in G.edges:
                # Get the start and end positions for the edge
                start_pos = pos[edge[0]]
                end_pos = pos[edge[1]]

                # Compute the direction vector (dx, dy, dz)
                dx, dy, dz = np.array(end_pos) - np.array(start_pos)

                # Draw the directed edge using Arrow3D
                ax.arrow3D(start_pos[0], start_pos[1], start_pos[2], dx, dy, dz,
                           mutation_scale=20, lw=1, arrowstyle="-|>", color="black")

            # Plot the nodes of the graph in 3D space
            for node, (x_node, y_node, z_node) in pos.items():
                if node in lr_toplot:
                    ax.scatter(x_node, y_node, z_node, c='green', s=100)
                else:
                    ax.scatter(x_node, y_node, z_node, c='red', s=100)

            # Show the plot
            plt.savefig("DAG.pdf", format="pdf")
            plt.show()



            ###### Plot 2


            fixed_lambda_0 = lambdas[0, 0]
            fixed_lambda_3 = lambdas[0, 1]

            # Filtering lambdas for the found fixed values
            filtered_lambdas = lambdas[(lambdas[:, 0] == fixed_lambda_0) & (lambdas[:, 1] == fixed_lambda_3)]

            # Extract x, y, and z
            x = filtered_lambdas[:, 3]
            y = filtered_lambdas[:, 2]
            z1 = -D0_test[(lambdas[:, 0] == fixed_lambda_0) & (lambdas[:, 1] == fixed_lambda_3)]
            z2 = -D2_test[(lambdas[:, 0] == fixed_lambda_0) & (lambdas[:, 1] == fixed_lambda_3)]
            z = z2

            # Create a meshgrid for plotting
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            X, Y = np.meshgrid(x_unique, y_unique)

            # We need to reshape z accordingly to match the meshgrid shape
            Z = np.zeros_like(X)
            for i in range(len(x)):
                x_index = np.where(x_unique == x[i])[0][0]
                y_index = np.where(y_unique == y[i])[0][0]
                Z[y_index, x_index] = z[i]

            # Plotting the 3D surface
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))

            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)

            # Customize the z axis
            ax.set_zlim(np.min(Z), np.max(Z))
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter('{x:.02f}')

            # Add a color bar which maps values to colors
            fig.colorbar(surf, shrink=0.5, aspect=5)

            # Create and plot the graph
            G = nx.from_numpy_array(sequential_matrix, create_using=nx.DiGraph)

            # Extract 3D positions of the efficient nodes
            lambdas_chosen = lambdas[efficient_ids, :]
            pos = {i: [lambdas_chosen[i, 3], lambdas_chosen[i, 2], -D2_test[efficient_ids[i]]] for i in G.nodes}

            # Plot the edges of the graph in 3D space
            for edge in G.edges:
                # Get the start and end positions for the edge
                start_pos = pos[edge[0]]
                end_pos = pos[edge[1]]

                # Compute the direction vector (dx, dy, dz)
                dx, dy, dz = np.array(end_pos) - np.array(start_pos)

                # Draw the directed edge using Arrow3D
                ax.arrow3D(start_pos[0], start_pos[1], start_pos[2], dx, dy, dz,
                           mutation_scale=20, lw=1, arrowstyle="-|>", color="black")

            # Plot the nodes of the graph in 3D space
            for node, (x_node, y_node, z_node) in pos.items():
                if node in lrn_toplot:
                    ax.scatter(x_node, y_node, z_node, c='green', s=100)
                else:
                    ax.scatter(x_node, y_node, z_node, c='red', s=100)


            # Show the plot
            plt.savefig("BH.pdf", format="pdf")
            plt.show()












if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--res_folder", type=str, default='ag_pruning_results')
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--n_cal", type=int, default=1999)
    parser.add_argument("--n_cal1", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--alphas", type=str, default='0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2')

    args = parser.parse_args()

    args.n_test = 50
    args.n_cal = 45
    args.n_cal1 = 20
    args.n_trials = 50
    args.alphas = '14'

    main(args)
