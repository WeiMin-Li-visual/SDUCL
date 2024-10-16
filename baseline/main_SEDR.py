# -*- coding: UTF-8 -*-
"""
@Project : SDUCL
@File    : main_SEDR.py
@Author  : Nan Chen
@Date    : 2024-03-06 14:31 
"""

import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch

from utils_baseline import calculate_clustering_matrix,mclust_R

import os
from pathlib import Path
import warnings
import SEDR

warnings.filterwarnings('ignore')

os.environ['R_HOME'] = '/root/miniconda3/envs/SEDR_env_CN/lib/R'
os.environ['R_USER'] = '/root/miniconda3/envs/SEDR_env_CN/lib/python3.10/site-packages/rpy2'

# Setting parameters
random_seed = 2023
SEDR.fix_seed(random_seed)
# gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# path
data_root = Path('/data/STAGATE_pyG/Data/')

dataset = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
           "151675", "151676"]
df = pd.DataFrame(columns=['Sample', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure', 'methods'])
n_clusters_all = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

for sample_name, n_clusters in zip(dataset, n_clusters_all):
    print(sample_name)

    # loading data
    adata = sc.read_visium(data_root / sample_name)
    adata.var_names_make_unique()

    df_meta = pd.read_csv(data_root / sample_name / 'metadata.tsv', sep='\t')
    adata.obs['layer_guess'] = df_meta['layer_guess']

    # Preprocessing
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.

    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    # Constructing neighborhood graph
    graph_dict = SEDR.graph_construction(adata, 12)
    print(graph_dict)

    # Training SEDR
    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat

    # Clustering
    mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')
    sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
    methods_ = "SEDR"
    df = calculate_clustering_matrix(df, sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'], sample_name,
                                     methods_)

df.to_csv('SEDR_result.csv')