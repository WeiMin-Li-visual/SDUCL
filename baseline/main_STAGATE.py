# -*- coding=utf-8 -*-
# name: nan chen
# date: 2023/5/21 16:46

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
# import sklearn
import STAGATE
from Train_STAGATE import train_STAGATE
from utils import Transfer_pytorch_Data, Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, Cal_Spatial_Net_3D, Batch_Data
from PIL import ImageFile
from utils import calculate_clustering_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ['R_HOME'] = '/usr/local/miniconda3/envs/STAGATE/lib/R'
os.environ['R_USER'] = '/usr/local/miniconda3/envs/STAGATE/lib/python3.8/site-packages/rpy2'

# dataset = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675",
#           "151676"]
# knn = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

df = pd.DataFrame(columns=['Sample', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure', 'methods'])
for section_id, k in zip(dataset, knn):
    print(section_id, k)
    input_dir = os.path.join('/hy-tmp/', section_id)
    # adata = sc.read_visium(path=input_dir, count_file=section_id + '_filtered_feature_bc_matrix.h5')
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ground_truth
    df_meta = pd.read_csv(input_dir + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color=["ground_truth"])

    Cal_Spatial_Net(adata, rad_cutoff=150)
    Stats_Spatial_Net(adata)

    adata = train_STAGATE(adata, section_id)
    # print(adata)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["ground_truth"], save=str(section_id) + ".png")
    adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=k)

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['mclust'], obs_df['ground_truth'])
    # ARIlist.append(ARI)
    # print('Adjusted rand index = %.2f' % ARI)
