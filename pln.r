library(devtools)
library(PLNmodels)
library(torch)
# install("../PLNmodels")

microcosm = readRDS("microcosm_reduced.rds")
best <- colMeans(microcosm$Abundance > 0) > 0.30
microcosm$Abundance <- microcosm$Abundance[, best]
microcosm$site_time <- droplevels(microcosm$site_time)

n <- dim(microcosm$Abundance)[1]
p <- dim(microcosm$Abundance)[2]

myPLN <- PLN("Abundance~ 1", data = microcosm, control = PLN_param(backend = "torch", config_optim = list(lr = 0.01, numepoch = 5000, maxeval = 100000, ftolrel = -1, xtolrel = -1)))
print(myPLN)

covariance = sigma(myPLN)
Y_ = microcosm$Abundance
X = torch_ones(n,1)
O_ = torch_zeros(n,p)
M_ = myPLN$var_par$M
S_ = myPLN$var_par$S
Omega = torch_inverse(covariance)
B_ = coef(myPLN)

Y = torch_tensor(Y_)
O = torch_tensor(O_)
M = torch_tensor(M_)
S = torch_tensor(S_)
B = torch_tensor(B_)

logfactorial_torch <- function(endog_){
  endog = torch_clone(endog_)
  endog[endog_ == 0] <- 1 ## 0! = 1!
  return(endog*torch_log(endog) - endog + torch_log(8*torch_pow(endog,3) +
                                                    4*torch_pow(endog,2) +
                                                    endog + 1/30)/6 +
         torch_log(pi)/2)
}
torch_vloglik <- function(Y__,X,O,M,S,Omega,B) {
  S2  <- torch_square(S)
  Z  <- O + M + torch_mm(X, B)
  A <- torch_exp(Z + .5 * S2)
  p = Y$shape[2]
  n = Y$shape[1]
  Ji_tmp = .5 * torch_logdet(Omega) + torch_sum(Y__ * Z - A + .5 * torch_log(S2), dim = 2) - .5 * torch_sum(torch_mm(M, Omega) * M + S2 * torch_diag(Omega), dim = 2)
  other <- n/2*torch_logdet(Omega)
  other = other + torch_sum(Y * Z - A + .5 * torch_log(S2))
  other = other - .5 * torch_sum(torch_mm(M, Omega) * M + S2 * torch_diag(Omega))
  Ji <- - torch_sum(logfactorial_torch(Y__), dim = 2) + Ji_tmp
  Ji <- .5 * p + as.numeric(Ji$cpu())
  return(Ji)
}
elbo = torch_vloglik(Y,X,O,M,S,Omega,B)
print('ELBO PLN at convergence')
print(sum(elbo))

write.csv(covariance, "no_covariates_cov_20779.csv")
write.csv(B_, "no_covariates_beta_20779.csv")
write.csv(M_, "no_covariates_M_20779.csv")
write.csv(S_, "no_covariates_S_20779.csv")
