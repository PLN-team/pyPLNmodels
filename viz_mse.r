library(ggplot2)
library(cowplot)
library(gridExtra)
library(hash)
library(latex2exp)
library(grid)
library(glue)

options(error=traceback)
# traceback()

h = hash()
h[["ELBO"]] = "ELBO"
h[["Reconstruction_error"]] =TeX('RMSE $Y$')
h[["RMSE_SIGMA"]] = TeX('RMSE $\\Omega$')
h[["RMSE_B"]] =TeX('RMSE $B$')
h[["RMSE_PI"]]=  TeX('RMSE $\\pi$')
h[["RMSE_B0"]] =TeX('RMSE $B^0$')
h[["TIME"]] = "Time"
h[["NBITER"]] = "Number of iterations"


colors = c("lightblue","blue","lightpink","red")

g_legend<-function(a.gplot){
      tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}
scaleFUN <- function(x) sprintf("%.4s", x)
plot_csv = function(namedoss,viz,inflation){
    df = subset(read.csv(paste('csv_data/',namedoss, sep = "")), select = -X)
    df[,"model_name"] = as.factor(df[,"model_name"])
    df[,"moyenne"] = as.factor(df[,"moyenne"])
    df[,"NBITER"] = as.numeric(df[,"NBITER"])
    df = df[c("model_name","moyenne","RMSE_SIGMA","RMSE_PI","RMSE_B")]
    third_column = names(df)[3]
    last_column = names(df)[length(names(df))]
    criterions <- names(df[,c(3:(dim(df)[2]))])
    if (viz == "poisson"){
        xlab = TeX('$XB$')
    }
    else{
        xlab = TeX('$\\pi')
    }
    plot_data_column = function(column){
        current_plot <- (ggplot(df, aes(x = moyenne, y = df[,column], fill =
                                        as.factor(model_name)) ) +
                         geom_boxplot(lwd = 0.03,outlier.shape = NA) +
                         geom_point(position = position_jitterdodge(), aes(fill =
                                                                     model_name,group
                                                                 = model_name,
                                                                 color =
                                                                     model_name),
                                    size = 0.2, alpha = 0.6)
                         + scale_y_log10() + scale_fill_manual(values = colors, name = "")
        + scale_color_manual(values = colors,name ="") + scale_x_discrete(labels=scaleFUN)
        +guides(fill=guide_legend(nrow=1,byrow=TRUE))) + theme_bw()
        if (column == third_column){
            current_plot = current_plot + ggtitle(inflation)+ theme(plot.title = element_text(hjust = 0.5))
        }
        if (column == last_column){
            current_plot = current_plot + labs(y = h[[column]],x = xlab)
        }
        else{

            if(inflation == "Non-dependent (3a)"){
                current_plot = current_plot + labs(y=h[[column]], x = NULL)
            }
            else{
            current_plot = current_plot + labs(y= NULL,x = NULL)
            }
        }
        return(current_plot)
         #+
    }
    remove_legend = function(myplot){
        return(myplot + theme(legend.position="none"))
    }
    vect = c(1:(length(criterions)))
    myplots <- lapply(criterions, plot_data_column)
    legend_all = get_legend(myplots[[1]])
    myplots <- lapply(myplots,remove_legend)
    return(myplots)
}
viz = "poisson"
pdf(paste("figures/",viz,".pdf",sep=""), width = 20)


name_doss = "_viz_global.csv"
first_plots = plot_csv(paste(viz,name_doss,sep=""),viz,"Non-dependent (3a)")

name_doss = "_viz_column-wise.csv"
second_plots = plot_csv(paste(viz,name_doss,sep=""),viz,"Column-dependent (3b)")

name_doss = "_viz_row-wise.csv"
third_plots = plot_csv(paste(viz,name_doss,sep =""),viz,"Row-dependent (3c)")

# if (viz == "proba"){
#     title = textGrob(TeX(glue("$n=350,p=100,d=1,XB=2,${inflation}")),gp=gpar(fontsize=20,font=3))
# }
# else{
#     title = textGrob(TeX(glue("$n=350,p=100,d=1,\\pi=0.3,${inflation}")),gp=gpar(fontsize=20,font=3))
# }
three_plots = c(first_plots,second_plots,third_plots)
grid.arrange(grobs = three_plots, as.table = FALSE, align = c("v"))#,bottom = legend_all,  ncol = 1,top = title, common.legend = TRUE)
dev.off()

# name_doss = "proba_viz_column-wise.csv"
# plot_csv(name_doss,"proba","column-dependent")

# name_doss = "proba_viz_row-wise.csv"
# plot_csv(name_doss,"proba","row-dependent")

# name_doss = "proba_viz_global.csv"
# plot_csv(name_doss,"proba","non-dependent")



# name_doss = "poisson_viz_column-wise_n800_p800.csv"
# plot_csv(name_doss,"poisson","column-dependent")

# name_doss = "poisson_viz_row-wise_n800_p800.csv"
# plot_csv(name_doss,"poisson","row-dependent")

# name_doss = "poisson_viz_global_n800_p_800.csv"
# plot_csv(name_doss,"poisson","non-dependent")

# name_doss = "proba_viz_column-wise_n800_p800.csv"
# plot_csv(name_doss,"proba","column-dependent")

# name_doss = "proba_viz_row-wise_n800_p800.csv"
# plot_csv(name_doss,"proba","row-dependent")

# name_doss = "proba_viz_global_n800_p_800.csv"
# plot_csv(name_doss,"proba","non-dependent")
