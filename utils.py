# -*- coding: UTF-8 -*-
"""
@Project : SDUCL
@File    : utils.py
@Author  : Nan Chen
@Date    : 2024-01-15 10:48 
"""

import ot
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
    homogeneity_completeness_v_measure, calinski_harabasz_score, davies_bouldin_score

from umap_test import get_umap


def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def calculate_clustering_matrix(df, y_pred, ground_truth, sample, methods_):
    ari = adjusted_rand_score(y_pred, ground_truth)
    nmi = normalized_mutual_info_score(y_pred, ground_truth)
    purity = purity_score(y_pred, ground_truth)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_pred, ground_truth)
    print(ari)

    df = df._append(pd.Series([sample, ari, nmi, purity, homogeneity, completeness, v_measure, methods_],
                             index=['Sample', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure',
                                    'methods']), ignore_index=True)
    return df


def calculate_one_parameter_matrix(df, parameter, y_pred, ground_truth, sample, methods_):
    ari = adjusted_rand_score(y_pred, ground_truth)
    nmi = normalized_mutual_info_score(y_pred, ground_truth)
    purity = purity_score(y_pred, ground_truth)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_pred, ground_truth)
    print(ari)

    df = df._append(pd.Series([sample, parameter, ari, nmi, purity, homogeneity, completeness, v_measure, methods_],
                             index=['Sample', 'Parameter', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure',
                                    'methods']), ignore_index=True)
    return df


def nolabel_clustering_matrix(df, X, labels, sample, methods_):
    sc_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    df = df.append(pd.Series([sc_score, ch_score, db_score, sample, methods_],
                             index=['Silhouette-Coefficient', 'Calinski-Harabasz', 'Davies-Bouldin', 'Sample',
                                    'methods']),
                   ignore_index=True)

    return df


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2024):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    print(res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01,
               refinement=False):
    pca = PCA(n_components=20, random_state=2024)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters).fit(embedding)
        kmeans_result = [i + 1 for i in kmeans.labels_]
        adata.obs['domain'] = list(map(lambda x: str(x), kmeans_result))
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


# from adjustText import adjust_text
#
#
# def gen_mpl_labels(
#         adata, groupby, exclude=(), ax=None, adjust_kwargs=None, text_kwargs=None, color_by_group=False
# ):
#     if adjust_kwargs is None:
#         adjust_kwargs = {"text_from_points": False}
#     if text_kwargs is None:
#         text_kwargs = {}
#
#     medians = {}
#
#     for g, g_idx in adata.obs.groupby(groupby).groups.items():
#         if g in exclude:
#             continue
#         medians[g] = np.median(adata[g_idx].obsm["X_umap"], axis=0)
#
#     # Fill the text colors dictionary
#     text_colors = {group: None for group in adata.obs[groupby].cat.categories}
#
#     if color_by_group and groupby + "_colors" in adata.uns:
#         for i, group in enumerate(adata.obs[groupby].cat.categories):
#             if group in exclude:
#                 continue
#             text_colors[group] = adata.uns[groupby + "_colors"][i]
#
#     if ax is None:
#         texts = [
#             plt.text(x=x, y=y, s=k, color=text_colors[k], **text_kwargs) for k, (x, y) in medians.items()
#         ]
#     else:
#         texts = [ax.text(x=x, y=y, s=k, color=text_colors[k], **text_kwargs) for k, (x, y) in medians.items()]
#
#
#     adjust_text(texts, arrowprops=dict(arrowstyle='->',
#                                        color='red',
#                                        lw=1))
