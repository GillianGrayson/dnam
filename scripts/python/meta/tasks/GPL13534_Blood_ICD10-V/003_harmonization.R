rm(list=ls())

install.packages('reticulate')
py_install("pandas")
install_github("https://github.com/regRCPqn/regRCPqn")

library("reticulate")
library(devtools)
library(minfi)
library("regRCPqn")

path <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/GPL13534_Blood_ICD10-V/R"
setwd(path)

pd <- import("pandas")

pheno <- pd$read_pickle("pheno.pkl")
betas <- pd$read_pickle("betasT.pkl")

betas <- logit2(betas)

betas <- cbind(ID_REF = rownames(betas), betas)
rownames(betas) <- 1:nrow(betas)


M_data_norm <- regRCPqn(M_data=betas, ref_path=path, data_name="regRCPqn", save_ref=TRUE)



data(BloodParkinson1)
betas <- as.data.frame.matrix(betas)

install.packages(c( "foreach", "doParallel"))
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install(c("minfi","ChAMPdata","Illumina450ProbeVariants.db","sva","IlluminaHumanMethylation450kmanifest","limma","RPMM","DNAcopy","preprocessCore","impute","marray","wateRmelon","goseq","plyr","GenomicRanges","RefFreeEWAS","qvalue","isva","doParallel","bumphunter","quadprog","shiny","shinythemes","plotly","RColorBrewer","DMRcate","dendextend","IlluminaHumanMethylationEPICmanifest","FEM","matrixStats","missMethyl","combinat"))
BiocManager::install("ramwas")

library("ChAMP")
library("xlsx")
library("doParallel")
library(sva)
library(minfi)
detectCores()

path_idat <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific/002_prepare_pd_for_ChAMP/GSE164056/raw/idat"
path_work <- "E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific/003_ChAMP_pipeline"
setwd(path_work)

myLoad = champ.load(directory = path_idat,
                    arraytype = "EPIC", # Choose microarray type is "450K" or "EPIC".(default = "450K")
                    method = "minfi", # Method to load data, "ChAMP" method is newly provided by ChAMP group, while "minfi" is old minfi way.(default = "ChAMP")
                    methValue = "B", # Indicates whether you prefer m-values M or beta-values B. (default = "B")
                    autoimpute = TRUE, # If after filtering (or not do filtering) there are NA values in it, should impute.knn(k=3) should be done for the rest NA?
                    filterDetP = TRUE, # If filter = TRUE, then probes above the detPcut will be filtered out.(default = TRUE)
                    ProbeCutoff = 0.1, # The NA ratio threshhold for probes. Probes with above proportion of NA will be removed.
                    SampleCutoff = 0.1, # The failed p value (or NA) threshhold for samples. Samples with above proportion of failed p value (NA) will be removed.
                    detPcut = 0.01, # The detection p-value threshold. Probes about this cutoff will be filtered out. (default = 0.01)
                    filterBeads = TRUE, # If filterBeads=TRUE, probes with a beadcount less than 3 will be removed depending on the beadCutoff value.(default = TRUE)
                    beadCutoff = 0.05, # The beadCutoff represents the fraction of samples that must have a beadcount less than 3 before the probe is removed.(default = 0.05)
                    filterNoCG = TRUE, # If filterNoCG=TRUE, non-cg probes are removed.(default = TRUE)
                    filterSNPs = TRUE, # If filterSNPs=TRUE, probes in which the probed CpG falls near a SNP as defined in Nordlund et al are removed.(default = TRUE)
                    filterMultiHit = TRUE, # If filterMultiHit=TRUE, probes in which the probe aligns to multiple locations with bwa as defined in Nordlund et al are removed.(default = TRUE)
                    filterXY = TRUE, # If filterXY=TRUE, probes from X and Y chromosomes are removed.(default = TRUE)
                    force = TRUE
)
save(myLoad, file="myLoad_ChAMP.RData")
save(myLoad, file="myLoad_minfi.RData")
load("myLoad_ChAMP.RData")
load("myLoad_minfi.RData")
myLoad$pd$Slide <- as.factor(myLoad$pd$Slide)
myLoad$pd$Array <- as.factor(myLoad$pd$Array)
myLoad$pd$Sex <- as.factor(myLoad$pd$Sex)
myLoad$pd$Sample_Group <- as.factor(myLoad$pd$Sample_Group)


CpG.GUI(CpG=rownames(myLoad$beta), arraytype="EPIC")
champ.QC(beta = myLoad$beta,
         pheno = myLoad$pd$Sample_Group,
         mdsPlot = TRUE,
         densityPlot = TRUE,
         dendrogram = TRUE,
         PDFplot = TRUE,
         Rplot = TRUE,
         Feature.sel = "None",
         resultsDir = "./QC_raw/")
QC.GUI(beta = myLoad$beta,
       pheno = myLoad$pd$Sample_Group,
       arraytype = "EPIC")

# BMIQ =================================================================================================================
myNorm <- champ.norm(beta = myLoad$beta,
                     rgSet = myLoad$rgSet,
                     mset = myLoad$mset,
                     resultsDir = "./QC_BMIQ/norm/",
                     method = "BMIQ",
                     plotBMIQ = FALSE,
                     arraytype = "EPIC",
                     cores = 4)
save(myNorm, file="myNorm_BMIQ.RData")
load("myNorm_BMIQ.RData")

champ.QC(beta = myNorm,
         pheno = myLoad$pd$Sample_Group,
         mdsPlot = TRUE,
         densityPlot = TRUE,
         dendrogram = TRUE,
         PDFplot = TRUE,
         Rplot = TRUE,
         Feature.sel = "None",
         resultsDir = "./QC_BMIQ/")

# FunctionalNormalize ==================================================================================================
myNorm <- champ.norm(beta = myLoad$beta,
                     rgSet = myLoad$rgSet,
                     mset = myLoad$mset,
                     resultsDir = "./QC_FunctionalNormalization/norm/",
                     method = "FunctionalNormalization",
                     plotBMIQ = FALSE,
                     arraytype = "EPIC",
                     cores = 4)
save(myNorm, file="myNorm_FunctionalNormalization.RData")
load("myNorm_FunctionalNormalization.RData")

myNorm_df <- data.frame(row.names(myNorm), myNorm)
colnames(myNorm_df)[1] <- "CpG"
write.table(myNorm_df, file = "myNorm_FunctionalNormalization.txt", row.names = F, sep = "\t", quote = F)

champ.QC(beta = myNorm,
         pheno = myLoad$pd$Sample_Group,
         mdsPlot = TRUE,
         densityPlot = TRUE,
         dendrogram = TRUE,
         PDFplot = TRUE,
         Rplot = TRUE,
         Feature.sel = "None",
         resultsDir = "./QC_FunctionalNormalization/")

# ======================================================================================================================
QC.GUI(beta = myNorm,
       pheno = myLoad$pd$Sample_Group,
       arraytype = "EPIC")

# SVD ==================================================================================================================
champ.SVD(beta = myNorm,
          rgSet = myLoad$rgSet,
          pd = myLoad$pd,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_before_Combat/")

myCombat <- champ.runCombat(beta = myNorm,
                            pd = myLoad$pd,
                            variablename = "Sample_Group",
                            batchname = c("Slide", "Array"),
                            logitTrans = TRUE)

myCombat <- champ.runCombat(beta = myNorm,
                            pd = myLoad$pd,
                            variablename = "Age",
                            batchname = c("Sex", "Slide", "Array"),
                            logitTrans = TRUE)
save(myCombat, file="myCombat_FunctionalNormalization_vae(Age)_batch(Sex_Slide_Array).RData")
myCombat_df <- data.frame(row.names(myCombat), myCombat)
colnames(myCombat_df)[1] <- "CpG"
write.table(myCombat_df, file = "myCombat_FunctionalNormalization_vae(Age)_batch(Sex_Slide_Array).txt", row.names = F, sep = "\t", quote = F)

mod.combat <- model.matrix( ~ 1 + myLoad$pd$Sex + myLoad$pd$Age)
myCombat <- sva::ComBat(dat = as.matrix(myNorm), batch = myLoad$pd$Sample_Group, mod = mod.combat, par.prior = T)

champ.SVD(beta = myCombat,
          rgSet = myLoad$rgSet,
          pd = myLoad$pd,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_after_Combat/")

# DMP ==================================================================================================================
myDMP <- champ.DMP(beta = myNorm,
                   pheno = myLoad$pd$Sample_Group,
                   compare.group = NULL,
                   adjPVal = 0.001,
                   adjust.method = "BH",
                   arraytype = "EPIC")
save(myNorm, file="myDMP.RData")
load("myDMP.RData")
colnames(myDMP[[1]])[0] <- "CpG"
write.csv(myDMP[[1]], file = "myDMP.csv")
head(myDMP[[1]])
DMP.GUI(DMP=myDMP[[1]],
        beta=myNorm,
        pheno=myLoad$pd$Sample_Group,
        cutgroupnumber=4)

# DMR ==================================================================================================================
myDMR <- champ.DMR(beta = myCombat,
                   pheno = myLoad$pd$Sample_Group,
                   compare.group = NULL,
                   arraytype = "EPIC",
                   method = "Bumphunter",
                   minProbes = 10,
                   adjPvalDmr = 0.001,
                   cores = 4,
                   ## following parameters are specifically for Bumphunter method.
                   maxGap = 300,
                   cutoff = NULL,
                   pickCutoff = TRUE,
                   smooth = TRUE,
                   smoothFunction = loessByCluster,
                   useWeights = FALSE,
                   permutations = NULL,
                   B = 250,
                   nullMethod = "bootstrap")
write.csv(myDMR, file = "myDMR_Bumphunter.csv")
save(myDMR, file="myDMR_Bumphunter.RData")

myDMR <- champ.DMR(beta = myNorm,
                   pheno = myLoad$pd$Sample_Group,
                   compare.group = NULL,
                   arraytype = "EPIC",
                   method = "ProbeLasso",
                   minProbes = 10,
                   adjPvalDmr = 0.001,
                   cores = 4,
                   ## following parameters are specifically for probe ProbeLasso method.
                   meanLassoRadius = 375,
                   minDmrSep = 1000,
                   minDmrSize = 50,
                   adjPvalProbe = 0.001,
                   Rplot = FALSE,
                   PDFplot = FALSE,
                   resultsDir = "./ProbeLasso/")

myDMR <- champ.DMR(beta = myNorm,
                   pheno = myLoad$pd$Sample_Group,
                   compare.group = NULL,
                   arraytype = "EPIC",
                   method = "DMRcate",
                   minProbes = 10,
                   adjPvalDmr = 0.001,
                   cores = 4,
                   ## following parameters are specifically for DMRcate method.
                   rmSNPCH = T,
                   fdr = 0.001,
                   dist = 2,
                   mafcut = 0.001,
                   lambda = 1000,
                   C = 2)

DMR.GUI(DMR = myDMR,
        beta = myNorm,
        pheno = myLoad$pd$Sample_Group,
        runDMP = TRUE,
        compare.group = NULL,
        arraytype = "EPIC")

myGSEA <- champ.GSEA(beta = myNorm,
                     DMP = myDMP[[1]],
                     DMR = myDMR,
                     CpGlist = NULL,
                     Genelist = NULL,
                     pheno = myLoad$pd$Sample_Group,
                     method = "fisher",
                     arraytype = "EPIC",
                     Rplot = TRUE,
                     adjPval = 0.001,
                     cores = 4)




# Check ChAMP with preprocessed files ==================================================================================
tmpNorm <- read.table("E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific/007_prepare_combined_data_for_R/GSE87571/betas.csv",
                      header = TRUE,
                      sep = ",",
                      dec = ".",
                      row.names = "CpG")

pd <- read.table("E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific/007_prepare_combined_data_for_R/GSE87571/pheno.csv",
                 header = TRUE,
                 sep = ",",
                 dec = ".",
                 row.names = "subject_id")

champ.SVD(beta = tmpNorm,
          pd = pd,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_tmp/")

tmpCombat <- champ.runCombat(beta = tmpNorm,
                             pd = pd,
                             variablename = "Age",
                             batchname = c("Dataset"),
                             logitTrans = TRUE)

champ.SVD(beta = tmpCombat,
          pd = pd,
          RGEffect = FALSE,
          PDFplot = TRUE,
          Rplot = TRUE,
          resultsDir = "./SVD_tmp/")

tmpCombat_df <- data.frame(row.names(tmpCombat), tmpCombat)
colnames(tmpCombat_df)[1] <- "CpG"
write.table(tmpCombat_df, file = "tmpCombat.txt", row.names = F, sep = "\t", quote = F)
save(tmpCombat, file="tmpCombat.RData")
load("tmpCombat.RData")
tmpCombat <- as.matrix(tmpCombat)
tmpCombat <- read.table("tmpCombat.txt",
                      header = TRUE,
                      sep = "\t",
                      dec = ".",
                      row.names = "CpG")

tmpDMR <- champ.DMR(beta = tmpCombat,
                   pheno = pd$Dataset,
                   compare.group = NULL,
                   arraytype = "450K",
                   method = "Bumphunter",
                   minProbes = 5,
                   adjPvalDmr = 0.05,
                   cores = 4,
                   ## following parameters are specifically for Bumphunter method.
                   maxGap = 300,
                   cutoff = NULL,
                   pickCutoff = TRUE,
                   smooth = TRUE,
                   smoothFunction = loessByCluster,
                   useWeights = FALSE,
                   permutations = NULL,
                   B = 250,
                   nullMethod = "bootstrap")
write.csv(tmpDMR, file = "tmpDMR_Bumphunter.csv")
save(tmpDMR, file="tmpDMR_Bumphunter.RData")
load("tmpDMR_Bumphunter.RData")

myDMR <- champ.DMR(beta = tmpCombat,
                   pheno = pd$Dataset,
                   compare.group = NULL,
                   arraytype = "450K",
                   method = "ProbeLasso",
                   minProbes = 5,
                   adjPvalDmr = 0.05,
                   cores = 4,
                   ## following parameters are specifically for probe ProbeLasso method.
                   meanLassoRadius = 375,
                   minDmrSep = 1000,
                   minDmrSize = 50,
                   adjPvalProbe = 0.001,
                   Rplot = FALSE,
                   PDFplot = FALSE,
                   resultsDir = "./ProbeLasso/")
write.csv(myDMR, file = "myDMR_ProbeLasso.csv")
save(myDMR, file="myDMR_ProbeLasso.RData")

myDMR <- champ.DMR(beta = tmpCombat,
                   pheno = pd$Dataset,
                   compare.group = NULL,
                   arraytype = "450K",
                   method = "DMRcate",
                   minProbes = 5,
                   adjPvalDmr = 0.05,
                   cores = 4,
                   ## following parameters are specifically for DMRcate method.
                   rmSNPCH = T,
                   fdr = 0.05,
                   dist = 2,
                   mafcut = 0.05,
                   lambda = 1000,
                   C = 2)

DMR.GUI(DMR = tmpDMR,
        beta = tmpCombat,
        pheno = pd$Dataset,
        runDMP = TRUE,
        compare.group = NULL,
        arraytype = "450K")

myGSEA <- champ.GSEA(beta = tmpCombat,
                     DMP = NULL,
                     DMR = tmpDMR,
                     CpGlist = NULL,
                     Genelist = NULL,
                     pheno = pd$Dataset,
                     method = "fisher",
                     arraytype = "450K",
                     Rplot = TRUE,
                     adjPval = 0.05,
                     cores = 4)