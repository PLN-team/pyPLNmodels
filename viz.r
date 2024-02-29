library(ggplot2)
library(cowplot)
library(gridExtra)
library(hash)
library(latex2exp)

pdf("test.pdf", width = 20)
df = subset(read.csv("csv_data/poisson_viz_column-wise.csv"), select = -X)
df[,"model_name"] = as.factor(df[,"model_name"])
df[,"moyenne"] = as.factor(df[,"moyenne"])
df[,"NBITER"] = as.numeric(df[,"NBITER"])

# criterions <- names(df[,c(3:(dim(df)[2]))])
criterions <- names(df[,c(3:(dim(df)[2]))])
h = hash()
h[["ELBO"]] = "ELBO"
h[["Reconstruction_error"]] =TeX('$\\hat {Y} - Y$')
h[["RMSE_SIGMA"]] = "ELBO"
h[["RMSE_B"]] = "ELBO"
h[["RMSE_PI"]] = "ELBO"
h[["RMSE_B0"]] = "ELBO"
h[["TIME"]] = "ELBO"
h[["NBITER"]] = "ELBO"


print(colnames(df))
print(criterions)
str(df)
# print(length(criterions))
# plist = c()
# for (criterion in criterions){
#     p = ggplot(df, aes(x = moyenne, y = criterion, fill = model_name, color = model_name)) + geom_boxplot() + scale_y_log10()+ theme(legend.position="none")  #+
#     plist = c(plist, p)
#     print("length")
#     print(length(plist))
# }

# l = list(p,g)
# n = length(l)
plot_data_column = function(column){
    print('column')
    print(column)
    return (ggplot(df, aes(x = moyenne, y = df[,column], fill = model_name, color =
               model_name)) + geom_boxplot() + theme(legend.position="none")+ labs(y = h[[column]])  + scale_y_log10())  #+
}
# ggplot(data, aes(x = moyenne, y = column))#, fill = model_name, color =
# p = plot_data_column(criterions[[2]])
# p
# ggplot(df, aes(x = moyenne, y = ELBO)) + geom_boxplot()#, fill = model_name, color =

myplots <- lapply(criterions, plot_data_column)
grid.arrange(grobs = myplots, ncol = length(myplots)/2)
# print(length(myplots))
dev.off()
