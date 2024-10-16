import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import numpy as np
from GraphST import GraphST
import warnings
import matplotlib.pyplot as plt
from utils import clustering, calculate_clustering_matrix

# 设置R语言路径&CUDA
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = '/root/miniconda3/envs/py310cu118_CN/lib/R'
os.environ['R_USER'] = '/root/miniconda3/envs/py310cu118_CN/lib/python3.10/site-packages/rpy2'

# set the section id and cluster nums
#dataset = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
#           "151675", "151676"]
#n_clusters = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

df = pd.DataFrame(columns=['Sample', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure', 'methods'])
ARIs = []
# 遍历每一个section
for section_id, k in zip(dataset, n_clusters):
    print(section_id, k)
    # read data
    input_dir = os.path.join('/data/STAGATE_pyG/Data/', section_id)
    # adata = read_10X_Visium(path=input_dir, section_id=section_id)
    adata = sc.read_visium(input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    print("spots_num: ", len(adata.X.toarray()[:, ]))

    # model
    model = GraphST(adata, device=device)
    adata = model.train()

    # clustering
    from utils import clustering

    radius = 50
    tool = 'mclust'
    if tool == 'mclust':
        clustering(adata, n_clusters=k, radius=radius, method=tool,
                   refinement=True)  # For DLPFC dataset, we use optional refinement step.
    elif tool == 'kmeans':
        clustering(adata, n_clusters=k, method=tool)
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters=k, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=True)

    # add ground_truth
    df_meta = pd.read_csv(input_dir + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    # # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    
    # ARI判断聚类结果
    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    ARI = round(ARI, 3)
    print(ARI)
