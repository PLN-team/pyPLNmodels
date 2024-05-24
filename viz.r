library(ggplot2)
library(grid)
library(gridExtra)
library(cowplot)
library(dplyr)
library(naniar)


file_pln_python = "python_pln_cpu.csv"
file_plnpca_python = "python_plnpca_cpu.csv"
file_pln_python_GPU = "python_pln_GPU.csv"
file_plnpca_python_GPU = "python_plnpca_GPU.csv"
file_pln_r = "df_pln_r.csv"
file_plnpca_r = "df_plnpca_r.csv"
file_gllvm = "df_gllvm.csv"
filenames = c(file_pln_python, file_plnpca_python, file_pln_python_GPU, file_plnpca_python_GPU)
modelnames = c("Pln", "PlnPCA", "Pln-GPU", "PlnPCA-GPU")
# filenames = c(file_pln_r,file_plnpca_r, file_gllvm)
# modelnames = c("R-Pln", "R-PlnPCA", "GLLVM")
nb_N = 3
get_df <- function(filename, modelname){
    df = read.csv(paste("csv_res_benchmark/",filename, sep = ""), header = TRUE, check.names = F)
    print('one df')
    print(df)
    df <- data.frame(df[,-1], check.names = F)
    df[,"Model"] = modelname
    # df <- df %>% replace_with_na(replace = list(x = 5000.))
    df[df == 10001]  = NA
    # replace(df, 10001, NA)
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
    current_plot <- current_plot + scale_y_log10()
}

pdf("paper/figures/plots_benchmark.pdf")
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
