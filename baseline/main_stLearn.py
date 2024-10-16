import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
    homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

BASE_PATH = Path('/data/STAGATE_pyG/Data/')
# sample_list = ['151672', '151673', '151674', '151675', '151676']
sample_list = ['151509']

for sample in sample_list:
    print("================ Start======================")
    OUTPUT_PATH = Path(f"/data/Test_sim/output/stLearn_results")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    TILE_PATH = Path(f'{OUTPUT_PATH}/tiles/')
    TILE_PATH.mkdir(parents=True, exist_ok=True)
    data = st.Read10X(os.path.join(BASE_PATH, sample))
    ground_truth_df = pd.read_csv(BASE_PATH / sample / 'metadata.tsv', sep='\t')
    ground_truth_df['ground_truth'] = ground_truth_df['layer_guess']
    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))
    n_cluster = len((set(ground_truth_df["ground_truth"]))) - 1
    data.obs['ground_truth'] = ground_truth_df["ground_truth"]
    ground_truth_df["ground_truth_le"] = ground_truth_le
    # pre-processing for gene count table
    st.pp.filter_genes(data, min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)
    st.em.run_pca(data, n_comps=15)
    st.pp.tiling(data, TILE_PATH)
    st.pp.extract_feature(data)
    # stSME
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm['raw_SME_normalized']
    st.pp.scale(data_)
    st.em.run_pca(data_, n_comps=30)
    st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans", algorithm='lloyd')
    st.pl.cluster_plot(data_, use_label="X_pca_kmeans")
    methods_ = "stSME_disk"
    results_df = calculate_clustering_matrix(data_.obs["X_pca_kmeans"], ground_truth_le, sample, methods_)
    # plt.savefig(OUTPUT_PATH / 'cluster.png')
    data_.obs["X_pca_kmeans"].to_csv(OUTPUT_PATH / 'stlearn_151509.csv', sep='\t', index=False)
    # df_PCA = pd.DataFrame(data=data_.obsm['X_pca'], index=data_.obs.index)
    # df_PCA.to_csv(OUTPUT_PATH / 'PCs.tsv', sep='\t')

    s = "/" + str(sample) + 'stLearn_151509.csv'
    results_df.to_csv(str(OUTPUT_PATH) + s)
