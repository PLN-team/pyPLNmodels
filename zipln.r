library(devtools)
library(PLNmodels)
library(torch)


microcosm = readRDS("microcosm_reduced.rds")
best <- colMeans(microcosm$Abundance > 0) > 0.05
microcosm$Abundance <- microcosm$Abundance[, best]
microcosm$site_time <- droplevels(microcosm$site_time)

n <- dim(microcosm$Abundance)[1]
p <- dim(microcosm$Abundance)[2]

zipln <- ZIPLN("Abundance~ 1 + site_time", data = microcosm)
print(zipln)

covariance = sigma(zipln)
B_ = coef(zipln)
Y_ = microcosm$Abundance
X = torch_ones(n,1)
O_ = torch_zeros(n,p)
M_ = zipln$var_par$M
S_ = zipln$var_par$S
R_ = zipln$var_par$R
Pi_ = zipln$model_par$Pi

write.csv(B_, "zi_no_covariates_beta_106836.csv")
write.csv(M_, "zi_no_covariates_M_106836.csv")
write.csv(S_, "zi_no_covariates_S_106836.csv")
write.csv(R_, "zi_no_covariates_R_106836.csv")
write.csv(Pi_, "zi_no_covariates_Pi_106836.csv")
write.csv(covariance, "zi_no_covariates_cov_106836.csv")
