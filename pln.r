library(PLNmodels)
library(factoextra)
library(ggplot2)


microcosm = readRDS("microcosm_reduced.rds")
abund = microcosm$Abundance
best <- colMeans(microcosm$Abundance > 0) > 0.05
microcosm$Abundance <- microcosm$Abundance[, best]
microcosm$site_time <- droplevels(microcosm$site_time)

myPLN <- PLN("Abundance~ 1 + site_time +  offset(log(Offset))", data = microcosm)
# help(PLN)

pdf("pln.pdf")
## bricolage car quelques points explosent, sans conséquences sur la convergence finale
obj_PLN <- -myPLN$optim_par$objective
obj_PLN
plot(obj_PLN, ## optimisation nlopt de la vraisemblance profilée
     ylim = c(obj_PLN[1], # fonction objective sans les constantes
              tail(obj_PLN,1)), log = "xy")
print(myPLN)
M <- myPLN$latent
microcosm$site_time


res <- prcomp(M, center = TRUE, scale = TRUE)
fviz_pca_ind(res, col.ind = microcosm$site_time)


plot(myPLN$optim_par$objective, type = "l", log = "xy") ## Les itérations du VEM: la vraisemblance estimée
dev.off()

myzi <- ZIPLN("Abundance~ 1 +  offset(log(Offset))", data = microcosm)
# help(PLN)

pdf("zi.pdf")
## bricolage car quelques points explosent, sans conséquences sur la convergence finale
obj_zi <- myzi$optim_par$objective
obj_zi
plot(obj_zi, ## optimisation nlopt de la vraisemblance profilée
     ylim = c(obj_zi[1], # fonction objective sans les constantes
              tail(obj_zi,1)), log = "xy")
print(myzi)
plot(myzi$optim_par$objective, type = "l", log = "xy") ## Les itérations du VEM: la vraisemblance estimée

B = myzi$model_par[[1]]
covariance = sigma(myzi)
pi = myzi$model_par[[3]]

myzi$var_par[[1]]
myzi$Abundance


options(max.print=3000)




dev.off()
