library(ggplot2)
library(latex2exp)

pdf("../figures/elbos.pdf", width = 20)


df = subset(read.csv("dict_elbos_2000_all_cov_stdanalytic.csv"), select = -X)
first = 1
df = df[-c(1:first),]
df = df[,c("Enhanced","Standard","Enhanced.Analytic","Standard.Analytic")]


y = c()

# real_names = list(TeX('Enhanced ($J_2$)'), TeX('Enhanced Analytic($\\tilde J_2$)'),TeX('Standard ($J_1$)'), TeX('Standard Analytic($\\tilde J_1$)'))

# for (i in c(1:4)){
#     names(df)[[i]] = real_names[[i]]
# }
# real_names = TeX(c('Enhanced ($J_2$)', 'Enhanced Analytic($\\tilde{J}_2$)','Standard ($J_1$)','Standard Analytic($\\tilde{J}_1$)'))
real_names = colnames(df)

# for (i in c(1:4)){
#     names(df)[[i]] = real_names[[i]]
# }
nb_plots = 4

all_names <- names(df)
for (i in c(1:nb_plots)){
    y = c(y,df[,all_names[[i]]])
}
y = - y
y = y -min(y)
y = y + 10

# y = -1/y
# y = -1/y
# lower_y = min(y)
# y = y - lower_y + 1e-6
to_rep <- c(1:length(df[,all_names[1]])) + first
models = c()
for (i in c(1:nb_plots)){
    models = c(models, rep(all_names[[i]], nrow(df)))
}

x = rep(to_rep, nb_plots)
df_reshaped <- data.frame(x = x,y = y, Model = models)

# window_moins = 10
# window_plus = 10000

plot = ggplot(df_reshaped, aes(x, y, col = Model)) + geom_line() + scale_y_log10() + annotation_logticks(side = 'l')+ labs(y = "Negative ELBO", x = "Iteration number")# + scale_y_continuous(trans = "log2")  #+ annotation_logticks()
plot = plot + scale_color_discrete(labels = unname(real_names))
plot = plot + theme_bw() + scale_colour_viridis_d()# + ggtitle("Same graph on simulated data") + theme(plot.title = element_text(hjust = 0.5))
plot

# # plot = plot  + theme_bw() + scale_y_log10(limits = c(lower_y - window_moins,lower_y + window_plus))
# # plot = plot + scale_y_log10()# + annotation_logticks()
# plot = plot + scale_y_continuous(trans = "log2")

# plot = plot + ylim(lower_y -window_moins, lower_y + window_plus) #+ xlim(500,2000)


dev.off()
