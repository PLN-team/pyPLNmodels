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

get_df = function(namedoss){
    df = subset(read.csv(paste('csv_data/',namedoss, sep = "")), select = -X)
    df[,"model_name"] = as.factor(df[,"model_name"])
    df[,"moyenne"] = as.factor(df[,"moyenne"])
    df[,"NBITER"] = as.numeric(df[,"NBITER"])
    df = df[c("model_name","moyenne","RMSE_SIGMA","RMSE_PI","RMSE_B")]
    return(df)
}

plot_csv = function(namedoss,viz,inflation, list_ylim_moins, list_ylim_plus){
    df = get_df(namedoss)
    third_column = names(df)[3]
    last_column = names(df)[length(names(df))]
    criterions <- names(df[,c(3:(dim(df)[2]))])
    if (viz == "poisson"){
        xlab = TeX('$XB$')
    }
    else{
        xlab = TeX('$\\pi')
    }
    plot_data_column = function(i){
        column = criterions[i]
        y_moins = list_ylim_moins[[i]]
        y_plus = list_ylim_plus[[i]]
        current_plot <- (ggplot(df, aes(x = moyenne, y = df[,column], fill =
                                        as.factor(model_name)) ) +
                         geom_point(position = position_jitterdodge(), aes(fill =
                                                                     model_name,group
                                                                 = model_name,
                                                                 color =
                                                                     model_name),
                                    size = 0.2, alpha = 0.6)+
                         geom_boxplot(lwd = 0.03, outlier.shape = NA)
                         + scale_y_log10() + scale_fill_manual(values = colors, name = "")
        + scale_color_manual(values = colors,name ="") + scale_x_discrete(labels=scaleFUN)
        +guides(fill=guide_legend(nrow=1,byrow=TRUE))) + theme_bw() + ylim(y_moins,y_plus)
        if (column == third_column){
            current_plot = current_plot + ggtitle(inflation)+ theme(plot.title = element_text(hjust = 0.5))
        }
        if (column == last_column){
            current_plot = current_plot + labs(y = h[[column]],x = xlab)
        }
        else{

            if(inflation == "Non-dependent (3a)"){
                current_plot = current_plot + labs(y=h[[column]], x = NULL)
                # current_plot = current_plot + labs(y=h[[column]], x = NULL)
            }
            else{
                print('inflation')
                print(inflation)
                current_plot = current_plot + labs(y= NULL,x = NULL)
            }
        }
        if(column == last_column && inflation != "Non-dependent (3a)"){
            current_plot = current_plot + labs(y=NULL, x = xlab)
        }
        return(current_plot)
         #+
    }
    remove_legend = function(myplot){
        return(myplot + theme(legend.position="none"))
    }
    vect = c(1:(length(criterions)))
    myplots <- lapply(vect, plot_data_column)
    legend_all = get_legend(myplots[[1]])
    myplots <- lapply(myplots,remove_legend)
    return(myplots)
}
viz = "poisson"

pdf(paste("figures/",viz,".pdf",sep=""), width = 20)

name_doss_1 = paste(viz,"_viz_global.csv", sep = "")
name_doss_2 = paste(viz,"_viz_column-wise.csv", sep = "")
name_doss_3 = paste(viz,"_viz_row-wise.csv", sep = "")
name_dosses = c(name_doss_1,name_doss_2,name_doss_3)

get_y_lims = function(name_doss){
    df = get_df(name_doss)
    criterions <- names(df[,c(3:(dim(df)[2]))])
    current_y_lim_plus = c(0,0,0)
    current_y_lim_moins = c(0,0,0)
    for (i in c(1:(length(criterions)))){
        column = criterions[[i]]
        y <- df[,column]
        current_y_lim_moins[[i]] = min(y)
        current_y_lim_plus[[i]] = max(y)
        # current_y_lim_moins[[i]] = min(y_lim_moins, current_y_lim_moins[[i]])
        # y_lim_plus = max(y)
        # current_y_lim_plus[[i]] = min(y_lim_plus, current_y_lim_plus)
        # current_y_lim_plus[[i]] = max(y_lim_plus, current_y_lim_plus)
    }
    return(list(current_y_lim_moins, current_y_lim_plus))

}
plot_all <- function (name_dosses,viz){
    ylim_pluss <- c(0,0,0)
    ylim_moinss <- c(1000,1000,1000)
    for (name_doss in name_dosses){
        current_y_lims <- get_y_lims(name_doss)
        current_ylim_moins <- current_y_lims[[1]]
        current_ylim_plus <- current_y_lims[[2]]
        ylim_moinss = pmin(ylim_moinss,current_ylim_moins)
        ylim_pluss = pmax(ylim_pluss,current_ylim_plus)
    }
    models = c("Non-dependent (3a)","Column-dependent (3b)","Row-dependent (3c)")
    plots = list()
    for (i in rev(c(1:(length(models))))){
        name_doss = name_dosses[i]
        print(name_doss)
        model = models[i]
        print(model)
        plots = append(plot_csv(name_doss, viz,model,ylim_moinss, ylim_pluss), plots)
    }
    return(plots)
}

three_plots <- plot_all(name_dosses, viz)

# first_plots = plot_csv(name_doss_1,viz,"Non-dependent (3a)", ylim_moins, ylim_plus)
# second_plots = plot_csv(name_doss_2,viz,"Column-dependent (3b)", ylim_moins, ylim_plus)
# third_plots = plot_csv(name_doss_3,viz,"Row-dependent (3c)", ylim_moins, ylim_plus)

# if (viz == "proba"){
#     title = textGrob(TeX(glue("$n=350,p=100,d=1,XB=2,${inflation}")),gp=gpar(fontsize=20,font=3))
# }
# else{
#     title = textGrob(TeX(glue("$n=350,p=100,d=1,\\pi=0.3,${inflation}")),gp=gpar(fontsize=20,font=3))
# }
# three_plots = c(first_plots,second_plots,third_plots)
grid.arrange(grobs = three_plots, as.table = FALSE, align = c("v"))#,bottom = legend_all,  ncol = 1,top = title, common.legend = TRUE)
dev.off()
