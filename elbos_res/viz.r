library(ggplot2)
# install.packages("formatR")
library(formatR)
tidy_source()

pdf("../figures/elbos.pdf", width = 20)


df = subset(read.csv("dict_elbos.csv"), select = -X)
y = c()
all_names <- names(df)
for (i in c(1:4)){
    y = c(y,df[,all_names[[i]]])
}
print(y)
x = rep(c(1:length(df[,all_names[1]])), 4)
# fist <- rep(all_names[[1]], nrow(df)))
df_reshaped <- data.frame(x = x,y = -y,
                       group = c(rep(all_names[[1]], nrow(df)),
                                 rep(all_names[[2]], nrow(df)),
                                 rep(all_names[[3]], nrow(df)),
                                 rep(all_names[[4]], nrow(df))
))
print('df')
print(df_reshaped)
plot = ggplot(df_reshaped, aes(x,y, col = group)) +
    geom_line() + labs(y = "Negative ELBO", x = "Iteration number") + scale_y_log10() + theme_bw()
plot

# print(df)



dev.off()
