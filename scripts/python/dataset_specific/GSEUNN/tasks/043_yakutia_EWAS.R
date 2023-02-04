rm(list=ls())

if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("DSS")
BiocManager::install(c("minfi","ChAMPdata","Illumina450ProbeVariants.db","sva","IlluminaHumanMethylation450kmanifest","limma","RPMM","DNAcopy","preprocessCore","impute","marray","wateRmelon","goseq","plyr","GenomicRanges","RefFreeEWAS","qvalue","isva","doParallel","bumphunter","quadprog","shiny","shinythemes","plotly","RColorBrewer","DMRcate","dendextend","IlluminaHumanMethylationEPICmanifest","FEM","matrixStats","missMethyl","combinat"))
BiocManager::install("ramwas")
BiocManager::install("ChAMP")
install.packages(c( "foreach", "doParallel"))
install.packages('reticulate')

library("ChAMP")
Sys.setenv(RETICULATE_PYTHON = "C:/Users/user/anaconda3/envs/py39/python.exe")
library("reticulate")
py_config()
Sys.which('python')
use_condaenv('py39')
py_run_string('print(1+1)')

dmp_pval <- 1
dmr_pval <- 0.05
dmr_min_probes <- 10

pd <- import("pandas")
path_load <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/data_for_R"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/data_for_R"
setwd(path_work)

# all_region ===========================================================================================================
pheno <- pd$read_pickle(paste(path_load, "/pheno_R_all_region.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
betas <- pd$read_pickle(paste(path_load, "/betas_R_all_region.pkl", sep=''))

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$Central_to_Yakutia, file = "DMP_all_region.csv")
DMP.GUI(
  DMP=dmp$Yakutia_to_Central,
  beta=betas,
  pheno=pheno$Region
)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 12,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR_all_region.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_all_region_genes.csv", row.names=FALSE)

DMR.GUI(
  DMR = dmr,
  beta = betas,
  pheno = pheno$Region,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "fisher",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = 0.05,
  cores = 4
)
write.csv(data.frame(gsea$DMR), file = "DMR_GSEA_all_region.csv", row.names=FALSE)

# central_sex ==========================================================================================================
rm(list=ls())

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_central_sex.pkl", sep=''))
pheno$Sex <- as.factor(pheno$Sex)
betas <- pd$read_pickle(paste(path_load, "/betas_R_central_sex.pkl", sep=''))

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Sex,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$F_to_M, file = "DMP_central_sex.csv")
DMP.GUI(
  DMP=dmp$F_to_M,
  beta=betas,
  pheno=pheno$Sex
)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Sex,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 12,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR_central_sex.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_central_sex_genes.csv", row.names=FALSE)

DMR.GUI(
  DMR = dmr,
  beta = betas,
  pheno = pheno$Sex,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Sex,
  method = "fisher",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = 0.05,
  cores = 4
)
write.csv(data.frame(gsea$DMR), file = "DMR_GSEA_central_sex.csv", row.names=FALSE)


# yakutia_sex ==========================================================================================================
rm(list=ls())

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_yakutia_sex.pkl", sep=''))
pheno$Sex <- as.factor(pheno$Sex)
betas <- pd$read_pickle(paste(path_load, "/betas_R_yakutia_sex.pkl", sep=''))

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Sex,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$F_to_M, file = "DMP_yakutia_sex.csv")
DMP.GUI(
  DMP=dmp$F_to_M,
  beta=betas,
  pheno=pheno$Sex
)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Sex,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 12,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR_yakutia_sex.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_yakutia_sex_genes.csv", row.names=FALSE)

DMR.GUI(
  DMR = dmr,
  beta = betas,
  pheno = pheno$Sex,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Sex,
  method = "fisher",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = 0.05,
  cores = 4
)
write.csv(data.frame(gsea$DMR), file = "DMR_GSEA_yakutia_sex.csv", row.names=FALSE)


# females_region =======================================================================================================
rm(list=ls())

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_females_region.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
betas <- pd$read_pickle(paste(path_load, "/betas_R_females_region.pkl", sep=''))

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$Central_to_Yakutia, file = "DMP_females_region.csv")
DMP.GUI(
  DMP=dmp$Yakutia_to_Central,
  beta=betas,
  pheno=pheno$Region
)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 12,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR_females_region.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_females_region_genes.csv", row.names=FALSE)

DMR.GUI(
  DMR = dmr,
  beta = betas,
  pheno = pheno$Region,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "fisher",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = 0.05,
  cores = 4
)
write.csv(data.frame(gsea$DMR), file = "DMR_GSEA_females_region.csv", row.names=FALSE)


# males_region =========================================================================================================
rm(list=ls())

pheno <- pd$read_pickle(paste(path_load, "/pheno_R_males_region.pkl", sep=''))
pheno$Region <- as.factor(pheno$Region)
betas <- pd$read_pickle(paste(path_load, "/betas_R_males_region.pkl", sep=''))

dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Region,
  compare.group = NULL,
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPIC"
)
write.csv(dmp$Central_to_Yakutia, file = "DMP_males_region.csv")
DMP.GUI(
  DMP=dmp$Yakutia_to_Central,
  beta=betas,
  pheno=pheno$Region
)

dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Region,
  compare.group = NULL,
  arraytype = "EPIC",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 12,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR_males_region.csv")
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPIC", annotation = "ilm10b4.hg19"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_males_region_genes.csv", row.names=FALSE)

DMR.GUI(
  DMR = dmr,
  beta = betas,
  pheno = pheno$Region,
  runDMP = TRUE,
  compare.group = NULL,
  arraytype = "EPIC"
)

gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Region,
  method = "fisher",
  arraytype = "EPIC",
  Rplot = TRUE,
  adjPval = 0.05,
  cores = 4
)
write.csv(data.frame(gsea$DMR), file = "DMR_GSEA_males_region.csv", row.names=FALSE)