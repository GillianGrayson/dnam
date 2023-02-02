rm(list=ls())
install.packages('reticulate')
library("reticulate")
Sys.setenv(RETICULATE_PYTHON = "C:/Users/alena/anaconda3/envs/py39/python.exe")
use_condaenv('py39')
pd <- import("pandas")

path_dataset <- "D:/pc_clock"
setwd(path_dataset)
betas <- pd$read_pickle(paste(path_dataset, "/beta.pkl", sep=''))

devtools::install_github("danbelsky/DunedinPACE")
BiocManager::install("preprocessCore")
library("preprocessCore")

library("DunedinPACE")
age <- PACEProjector(betas)
result <- age$DunedinPACE


datf <- list(ID = names(result))
class(datf) <- c("data.frame", "tbl_df")
attr(datf, "row.names") <- .set_row_names(length(result))
datf$mPACE <- result

library("writexl")
write_xlsx(datf, "D:/pc_clock/dunedin.xlsx")