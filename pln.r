library(PLNmodels)


microcosm = readRDS("microcosm_reduced.rds")
abund = microcosm$Abundance
best <- colMeans(microcosm$Abundance > 0) > 0.05
microcosm$Abundance <- microcosm$Abundance[, best]
microcosm$site_time <- droplevels(microcosm$site_time)



myPLN = PLN("Abundance~ 1 +  offset(log(Offset))", data = microcosm)
# help(PLN)

pdf("pln.pdf")
## bricolage car quelques points explosent, sans conséquences sur la convergence finale
obj_PLN <- -myPLN$optim_par$objective
plot(obj_PLN, ## optimisation nlopt de la vraisemblance profilée
     ylim = c(obj_PLN[1], # fonction objective sans les constantes
              tail(obj_PLN,1)), log = "xy")
print(myPLN)
plot(myPLN$optim_par$objective, type = "l", log = "xy") ## Les itérations du VEM: la vraisemblance estimée
dev.off()
