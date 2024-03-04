library(ggplot2)
library(cowplot)
library(gridExtra)
library(hash)
library(latex2exp)
library(grid)
library(glue)


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
    pdf(paste("figures/",namedoss,".pdf",sep=""), width = 20)
    df = subset(read.csv(paste('csv_data/',namedoss, sep = "")), select = -X)
    df[,"model_name"] = as.factor(df[,"model_name"])
    df[,"moyenne"] = as.factor(df[,"moyenne"])
    df[,"NBITER"] = as.numeric(df[,"NBITER"])
    print('names:')
    print(names(df))
    df = df[,c(1,2,5,6,8,9,3,4,7,10)]

    criterions <- names(df[,c(3:(dim(df)[2]))])
    if (viz == "poisson"){
        xlab = TeX('$XB$')
    }
    else{
        xlab = TeX('$\\pi')
    }
    plot_data_column = function(column, data){
        return (ggplot(data, aes(x = moyenne, y = data[,column], fill = model_name)) + geom_boxplot(lwd = 0.03) + labs(y = h[[column]], x =
        xlab)  + scale_y_log10() + scale_fill_manual(values = colors, name = "")
        + scale_color_manual(values = colors,name ="") + scale_x_discrete(labels=scaleFUN)
        +guides(fill=guide_legend(nrow=1,byrow=TRUE))
        ) #+
    }
    remove_legend = function(myplot){
        return(myplot + theme(legend.position="none"))
    }
    myplots <- lapply(criterions, plot_data_column, data = df)
    legend_all = get_legend(myplots[[1]])
    myplots <- lapply(myplots,remove_legend)
    if (viz == "proba"){
        title = textGrob(TeX(glue("$n=350,p=100,d=1,XB=2,${inflation}")),gp=gpar(fontsize=20,font=3))
    }
    else{
        title = textGrob(TeX(glue("$n=350,p=100,d=1,\\pi=0.3,${inflation}")),gp=gpar(fontsize=20,font=3))
    }
    grid.arrange(grobs = myplots,bottom = legend_all,  ncol = length(myplots)/2,top = title, common.legend = TRUE)
    dev.off()
}

name_doss = "poisson_viz_column-wise.csv"
plot_csv(name_doss,"poisson","column-dependent")

# name_doss = "poisson_viz_row-wise.csv"
# plot_csv(name_doss,"poisson","row-dependent")

# name_doss = "poisson_viz_global.csv"
# plot_csv(name_doss,"poisson","non-dependent")

# name_doss = "proba_viz_column-wise.csv"
# plot_csv(name_doss,"proba","column-dependent")

# name_doss = "proba_viz_row-wise.csv"
# plot_csv(name_doss,"proba","row-dependent")

# name_doss = "proba_viz_global.csv"
# plot_csv(name_doss,"proba","non-dependent")
