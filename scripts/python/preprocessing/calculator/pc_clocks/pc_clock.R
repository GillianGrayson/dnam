rm(list=ls())

install.packages('reticulate')
install.packages("writexl")
install.packages("data.table")

Sys.setenv(RETICULATE_PYTHON = "C:/Users/user/anaconda3/envs/py39/python.exe")
library("reticulate")
library("data.table")
library("writexl")
py_config()
Sys.which('python')
use_condaenv('py39')

pd <- import("pandas")

path_dataset <- "D:/YandexDisk/Work/pydnameth/datasets/GPL13534/GSE55763/calculator/pc_clock"
path_clocks <- "D:/YandexDisk/Work/pydnameth/datasets/lists/cpgs/PC_clocks/"
setwd(path_dataset)

pheno <- pd$read_pickle(paste(path_dataset, "/pheno.pkl", sep=''))
betas <- pd$read_pickle(paste(path_dataset, "/betas.pkl", sep=''))

betas_t <- transpose(betas)
rownames(betas_t) <- colnames(betas)
colnames(betas_t) <- rownames(betas)

source(paste(path_clocks, "run_calcPCClocks.R", sep = ""))
source(paste(path_clocks, "run_calcPCClocks_Accel.R", sep = ""))

PCClock_DNAmAge <- calcPCClocks(
  path_to_PCClocks_directory = path_clocks,
  datMeth = betas_t,
  datPheno = pheno
)

PCClock_DNAmAge <- calcPCClocks_Accel(PCClock_DNAmAge)
PCClock_DNAmAge_exp <- cbind("ID"=rownames(PCClock_DNAmAge), PCClock_DNAmAge)
write_xlsx(PCClock_DNAmAge_exp, paste(path_dataset, "/result.xlsx", sep=''))
