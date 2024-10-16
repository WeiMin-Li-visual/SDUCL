library(BayesSpace)
library(ggplot2)

data_path <- "D:/Code/GraphST/Data/DLPFC"
save_path <- "C:/Users/nanchen/Desktop/Baseline/BayesSpace"

sample <- c("151507", "151508",'151509','151510', '151669', '151670', '151671', '151672', '151673','151674','151675','151676')

for (sample.name in sample ){
dir.input <- file.path(data_path, sample.name)
dir.output <- file.path(save_path, sample.name)


print(file.exists("D:/Code/GraphST/Data/DLPFC/151507/filtered_feature_bc_matrix.h5"))

if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}

if(sample.name %in% c('151669', '151670', '151671', '151672')) {
  n_clusters <- 5} else {
  n_clusters <- 7}

### load data
dlpfc <- readVisium(dir.input) 
dlpfc <- logNormCounts(dlpfc)

set.seed(88)
dec <- scran::modelGeneVar(dlpfc)
top <- scran::getTopHVGs(dec, n = 2000)

set.seed(66)
dlpfc <- scater::runPCA(dlpfc, subset_row=top)

## Add BayesSpace metadata
dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)

q <- n_clusters  # Number of clusters
d <- 15  # Number of PCs

## Run BayesSpace clustering
set.seed(104)
dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium',
                        nrep=50000, gamma=3, save.chain=TRUE)

labels <- dlpfc$spatial.cluster

## View results
clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
  scale_fill_viridis_d(option = "A", labels = 1:7) +
  labs(title="BayesSpace")

ggsave(file.path(dir.output, 'clusterPlot.png'), width=5, height=5)

##### save data
write.table(colData(dlpfc), file=file.path(dir.output, 'bayesSpace.csv'), sep='\t', quote=FALSE)

}
