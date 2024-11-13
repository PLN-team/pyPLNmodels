library(ggplot2)
library(grid)
library(gridExtra)
library(cowplot)
library(dplyr)
library(naniar)
library(latex2exp)
library(viridis)

french = TRUE

if (french == FALSE){
    nb_var_text = TeX("Number of variables $p$")
    yabs = "Running time (seconds)"
    ours_text = " (ours)"
    name_file = "benchmark.pdf"
}else{
    nb_var_text = TeX("Nombre de variables $p$")
    yabs = "Temps (secondes)"
    ours_text = " (nous)"
    name_file = "benchmark_fr.pdf"
}



file_pln_python = "python_pln_cpu.csv"
file_plnpca_python = "python_plnpca_cpu.csv"
file_pln_python_GPU = "python_pln_GPU.csv"
file_plnpca_python_GPU = "python_plnpca_GPU.csv"
file_pln_r = "df_pln_r.csv"
file_plnpca_r = "df_plnpca_r.csv"
file_gllvm = "df_gllvm.csv"
filenames = c(file_pln_python, file_plnpca_python, file_pln_python_GPU, file_plnpca_python_GPU, file_gllvm, file_pln_r, file_plnpca_r)
modelnames = c(paste("py-PLN-CPU", ours_text, sep = ""), paste("py-PLN-PCA-CPU", ours_text, sep = ""), paste("py-PLN-GPU", ours_text, sep = ""), paste("py-PLN-PCA-GPU", ours_text, sep = ""), "GLLVM", "R-PLN", "R-PLN-PCA")
###            GLLVM          pyplncpu         pyplngpu       pyplnpcaCPU      pyPLNPCAGPU  RPLN    RPLNPCA



col_viridis = viridis(10)

colors =     c(col_viridis[[5]],    col_viridis[[8]],          col_viridis[[5]],      col_viridis[[8]],        col_viridis[[1]],    col_viridis[[5]], col_viridis[[8]])
linestyles = c("solid",        "solid",        "dashed",     "dashed",      "dotdash",   "dotted",   "dotted")
# filenames = c(file_pln_r,file_plnpca_r, file_gllvm)
# modelnames = c("R-Pln", "R-PlnPCA", "GLLVM")

nb_N = 3
get_df <- function(filename, modelname){
    df = read.csv(paste("csv_res_benchmark/",filename, sep = ""), header = TRUE, check.names = F)
    df <- data.frame(df[,-1], check.names = F)
    df[,"Model"] = modelname
    df[df == 10001]  = NA
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
    current_plot <- ggplot(df) + geom_line(aes(x =dim, y = df[,colnames(df)[i]], group = Model, col = Model, linetype = Model), shape = 23, size = 0.6)
    current_plot <- current_plot + ggtitle(paste("n = ", colnames(df)[i])) + labs(y = yabs, x = nb_var_text)
    current_plot <- current_plot + scale_y_log10()
    current_plot <- current_plot + scale_color_manual(values=colors, limits = modelnames) + scale_linetype_manual(values=linestyles, limits = modelnames)
    current_plot <- current_plot + theme_bw()+ theme(legend.key.width = unit(1.2,"cm"), plot.title = element_text(hjust = 0.5),legend.text = element_text(size=7)) + guides(col = guide_legend(nrow = 2, byrow = TRUE, title = ""), linetype =  guide_legend(nrow = 1, byrow = FALSE, title = ""), linetype = guide_legend(override.aes = list(size = 4)) )
    # current_plot <- current_plot + guides()
}

pdf(paste("../paper/figures/", name_file, sep = ""))
plots <- list()
for (i in c(1:nb_N)){
    tmp_plot <- get_plot_i(i)
    common_legend = get_legend(tmp_plot)
    tmp_plot <- remove_legend(tmp_plot)
    # if (i == 1){
        # tmp_plot <- tmp_plot + annotate("text", x = 1, y = -1, label = "Ours", hjust = -0.08, size = 3)+coord_cartesian(xlim=c(10,14), clip = "off")
        # tmp_plot <- tmp_plot + annotate("text", x = 1, y = -2, label = "Others", hjust = -0.08, size = 3)
        # tmp_plot <- tmp_plot + annotation_custom(
        #               grob = textGrob(label = "test", hjust = 1, gp = gpar(cex = 1.5)),
        #               ymin = -14,      # Vertical position of the textGrob
        #               ymax = 14,
        #               xmin = -14.3,         # Note: The grobs are positioned outside the plot area
        #               xmax = 14.3)
        # tmp_plot = tmp_plot + geom_text(aes(label = "test", x = 3, y = 3), hjust = -3, vjust = 3.5)
    # }
    plots <- append(list(tmp_plot), plots)
}
mygrid = grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 1, bottom = common_legend, common.legend = TRUE)#, top=paste("alpha = ", alpha))
# mygrid <- mygrid + annotation_custom(
#                       grob = textGrob(label = "test", hjust = 1, gp = gpar(cex = 1.5)),
#                       ymin = -1,      # Vertical position of the textGrob
#                       ymax = 1,
#                       xmin = -1.3,         # Note: The grobs are positioned outside the plot area
#                       xmax = 1.3)
# mygrid

dev.off()
