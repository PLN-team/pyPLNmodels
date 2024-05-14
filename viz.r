library(ggplot2)
library(grid)
library(gridExtra)
library(cowplot)


file_pln = "csv_res_benchmark/python_pln.csv"
file_plnpca = "csv_res_benchmark/python_plnpca.csv"
filenames = c(file_pln,file_plnpca)
modelnames = c("Pln", "PlnPCA")
nb_N = 2
get_df <- function(filename, modelname){
    df = read.csv(filename, header = TRUE, check.names = F)
    df <- data.frame(df[,-1], check.names = F)
    df[,"Model"] = modelname
    return(df)
}
# ?read.csv
# plot_file(file_pln)

df = get_df(filenames[1], modelnames[1])
for (i in c(2:(length(filenames)))){
    filename = filenames[i]
    modelname = modelnames[i]
    new_df = get_df(filename, modelname)
    df = rbind(df, new_df)
}
df$Model = as.factor(df$Model)

remove_legend <- function(myplot){
    return(myplot + theme(legend.position="none"))
}


get_plot_i <- function(i){
    current_plot <- ggplot(df) + geom_line(aes(x =dim, y = df[,colnames(df)[i]], group = Model, col = Model))
    current_plot <- current_plot + theme_bw()+ theme(plot.title = element_text(hjust = 0.5)) + guides(fill = guide_legend(nrow = 1, byrow = TRUE))
    current_plot <- current_plot + ggtitle(paste("n = ", colnames(df)[i])) + labs(y = "Running time (seconds)")
}

pdf("plots_benchmark.pdf")
print('y')
print(df)
plots <- list()
for (i in c(1:nb_N)){
    tmp_plot <- get_plot_i(i)
    common_legend = get_legend(tmp_plot)
    tmp_plot <- remove_legend(tmp_plot)
    plots <- append(list(tmp_plot), plots)
}
grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 1, bottom = common_legend, common.legend = TRUE)#, top=paste("alpha = ", alpha))

dev.off()
