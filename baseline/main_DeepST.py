import os
from DeepST import run
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn import metrics
import numpy as np
import pandas as pd
import warnings
import time
from utils_func import calculate_clustering_matrix

# warning
warnings.filterwarnings("ignore")

# 初始化参数
data_path = "/data/STAGATE_pyG/Data/"
save_path = "Results"

#dataset = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
#           "151675", "151676"]
#n_clusters = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

ARIs = []
df = pd.DataFrame(columns=['Sample', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure', 'methods'])
for data_name, n_domains in zip(dataset, n_clusters):
    print("-------section_id:" + data_name + " n_domains:" + str(n_domains) + "-------")
    deepen = run(save_path=save_path,
                 task="Identify_Domain",
                 # "Identify_Domain" and "Integration"
                 pre_epochs=800,
                 # choose the number of training
                 epochs=1000,
                 # choose the number of training
                 use_gpu=True)

    # 读取数据
    adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)

    # 分割组织学图像
    adata = deepen._get_image_crop(adata, data_name=data_name)

    # 数据增强
    adata = deepen._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)

    # 构建图
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree")

    # PCA对数据进一步处理
    data = deepen._data_process(adata, pca_n_comps=200)

    # 训练模型
    deepst_embed = deepen._fit(
        data=data,
        graph_dict=graph_dict, )

    # DeepST得到的embedding
    adata.obsm["DeepST_embed"] = deepst_embed

    # priori表示数据是否有先验知识（簇数）
    adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori=True)
    path = os.path.join(data_path, data_name)
    df_meta = pd.read_csv(path + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    
    # ARI判断聚类结果
    ARI = metrics.adjusted_rand_score(adata.obs['DeepST_refine_domain'], adata.obs['ground_truth'])
    print(ARI)
    ARI = round(ARI, 3)
    print(ARI)

