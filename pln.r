library(PLNmodels)


microcosm = readRDS("microcosm_reduced.rds")
abund = microcosm$Abundance
best <- colMeans(microcosm$Abundance > 0) > 0.05
microcosm$Abundance <- microcosm$Abundance[, best]
microcosm$site_time <- droplevels(microcosm$site_time)



pln = PLN("Abundance~ 1 +  offset(log(Offset))", data = microcosm)
# help(PLN)
print(pln)
