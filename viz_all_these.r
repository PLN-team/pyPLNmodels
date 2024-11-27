library(ggplot2)
library(cowplot)
library(gridExtra)
library(dplyr)
library(hash)
library(latex2exp)
library(grid)
library(glue)
library(viridis)

options(error=traceback)

viz = "proba"
perf = "stat"

pdf(paste("/home/bastien/These/manuscript/soutenance/figures/",viz,"_",perf,"_column_dependent.pdf",sep=""), height = 20, width = 30)

get_name_computation <- function(viz,formula){
    if (viz == "dims"){
        return(paste(viz,formula,"not_n_or_p_500.csv", sep = "_"))
    }
    else{
        return(paste(viz,formula,"not_n_or_p_150.csv", sep = "_"))
    }
}

base_colors <- viridis(4)
base_colors <- c(base_colors[[3]], base_colors[[1]])
print('base colors')
print(base_colors)

# Adjust the purple color to be lighter
lighter_purple <- adjustcolor(base_colors[2], alpha.f = 0.8)
base_colors[2] <- lighter_purple

lighter_colors <- sapply(base_colors, function(col) adjustcolor(col, alpha.f = 0.5))
print('lighter colors')
print(lighter_colors[2])
all_colors <- c(base_colors, lighter_colors[1][1], lighter_colors[2][1])
all_colors <- c(all_colors[3], all_colors[1], all_colors[2], all_colors[4])
all_colors <- c(all_colors[3], all_colors[1], all_colors[2], all_colors[4])
all_colors <- c(all_colors[2], all_colors[1], all_colors[3], all_colors[4])

name_doss_1 = paste(viz,"_viz_global_right_simu_multin.csv", sep = "")
name_doss_2 = paste(viz,"_viz_column-wise_right_simu_multin.csv", sep = "")
name_doss_3 = paste(viz,"_viz_row-wise_right_simu_multin.csv", sep = "")
name_dosses = c(name_doss_2)

h = hash()
h[["ELBO"]] = "ELBO"
h[["Reconstruction_error"]] =TeX('RMSE $Y$')
h[["RMSE_OMEGA"]] = TeX('RMSE $\\Omega$')
h[["RMSE_SIGMA"]] = TeX('RMSE $\\Sigma$')
h[["RMSE_B"]] =TeX('RMSE $B$')
h[["RMSE_PI"]]=  TeX('RMSE $\\pi$')
h[["RMSE_B0"]] =TeX('RMSE $B^0$')
h[["TIME"]] = "Time"
h[["NBITER"]] = "Number of iterations"

g_legend<-function(a.gplot){
      tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}
scaleFUN <- function(x) sprintf("%.4s", x)

get_df = function(namedoss, perf, viz){
    columns = c("model_name")
    if (perf == "stat"){
        if (viz == "samples" || viz == "dims"){
            columns = c(columns,"xscale")
        }
        else{
            columns = c(columns,"moyenne")
        }
        columns = c(columns,"RMSE_SIGMA","RMSE_PI","RMSE_B")
    }
    if (perf == "computation"){
        if (viz == "poisson" || viz == "proba"){
            columns = c(columns, "moyenne")
        }
        else{
            columns = c(columns,"xscale")
        }
        columns = c(columns,"TIME","Reconstruction_error")
    }
    if (perf == "annexe"){
        columns = c(columns,"moyenne","RMSE_OMEGA","RMSE_PI","RMSE_B","RMSE_B0")
    }
    df = subset(read.csv(paste('csv_data/',namedoss, sep = "")), select = -X)
    for (column in colnames(df)){
        df[df == 666] = NA
        }
    df[df$model_name == "Pln",]$model_name <- "PLN"
    if (viz == "samples" || viz == "dims"){
        df[df$model_name == "fair_Pln",]$model_name <- "Oracle PLN"
    }
    else{
        df[df$model_name == "Fair Pln",]$model_name <- "Oracle PLN"
    }
    df <- df[df$model_name %in% c("Enhanced Analytic", "Enhanced", "Standard", "Standard Analytic"),]
    df[df$model_name == "Enhanced Analytic",]$model_name <- "Dépendant (Analytique)"
    df[df$model_name == "Enhanced",]$model_name <- "Dépendant"
    df[df$model_name == "Standard Analytic",]$model_name <- "Indépendant (Analytique)"
    df[df$model_name == "Standard",]$model_name <- "Indépendant"


    df[,"model_name"] = as.factor(df[,"model_name"])
    print('levels')
    print(levels(df$model_name))
    if ("moyenne" %in% columns){
        df[,"moyenne"] = as.factor(df[,"moyenne"])
    }
    else{
        if (viz == "dims" || viz == "samples"){
            df <- df[df[,"xscale"]%%100 == 0,]
        }
        df[,"xscale"] = as.factor(df[,"xscale"])
    }

    df[,"NBITER"] = as.numeric(df[,"NBITER"])
    df = df[columns]
    df = df[df$model_name != "ZIP",]
    return(df)
}

plot_csv = function(namedoss,viz,inflation, list_ylim_moins, list_ylim_plus, perf){
    df = get_df(namedoss,perf,viz)
    model_levels <- unique(df$model_name)
    names(all_colors) <- model_levels


    criterions <- c("RMSE_SIGMA", "RMSE_B", "RMSE_PI")
    xlab = TeX('$\\pi')
    plot_data_column = function(i){
        column = criterions[i]
        y_moins = list_ylim_moins[[i]]
        y_plus = list_ylim_plus[[i]]
        first_col = "moyenne"
        current_plot <- (ggplot(df, aes(x = df[,first_col], y = df[,column], fill =
                                        as.factor(model_name)) )
                         # + geom_point(position = position_jitterdodge(), aes(color =
                         #                                             model_name,group
                         #                                         = model_name,
                         #                                         ),
                         #            size = 0.05, alpha = 0.2)
                         + geom_boxplot(lwd = 1.2, outlier.shape = NA)
                        + scale_fill_manual(values = all_colors, name = "")
                        + scale_colour_manual(values = all_colors, name = "")
                         # + scale_fill_manual(values = all_colors, name = "")
                         # + scale_colour_manual(values = all_colors, name = "")
                        + scale_x_discrete(labels=scaleFUN)
        +guides(fill=guide_legend(nrow=1,byrow=TRUE)) ) + theme_bw()+
         theme(legend.key.size = unit(1,"cm"),legend.text = element_text(size=50) )

        if (perf != "computation"){
            current_plot = current_plot +  scale_y_log10()
        }
        if (column != "RMSE_B0" & column != "RMSE_PI"){
            if (perf != "computation"){
                current_plot = current_plot +  scale_y_log10(limits = c(y_moins,y_plus))
            }
        }
        current_plot = current_plot + ggtitle(h[[column]]) + theme(plot.title = element_text(hjust = 0.5, size = 40, face = "bold"))
        if (column == "RMSE_B"){
            current_plot = current_plot + labs(y = h[[column]],x = xlab)
        }
        current_plot = current_plot + theme(axis.text.x = element_text(size = 30), axis.text.y = element_text(size = 30), axis.title.x = element_text(size = 40), axis.title.y = element_text(size = 40))
        # else{
        #     current_plot = current_plot + labs(y=h[[column]], x = NULL)
        # }
        current_plot = current_plot + labs(y=NULL, x = xlab)
        return(current_plot)
    }
    remove_legend = function(myplot){
        return(myplot + theme(legend.position="none"))
    }
    vect = c(1:(length(criterions)))
    myplots <- lapply(vect, plot_data_column)
    common_legend = get_legend(myplots[[1]])
    myplots <- lapply(myplots,remove_legend)
    return(list(myplots,common_legend))
}

get_y_lims = function(name_doss,perf,viz){
    df = get_df(name_doss,perf,viz)
    criterions <- c("RMSE_SIGMA", "RMSE_B", "RMSE_PI")
    current_y_lim_plus = c(0,0,0)
    current_y_lim_moins = c(0,0,0)
    for (i in c(1:(length(criterions)))){
        column = criterions[[i]]
        y <- df[,column]
        y <- y[!is.na(y)]

        current_y_lim_moins[[i]] = min(y)
        current_y_lim_plus[[i]] = max(y)
    }
    return(list(current_y_lim_moins, current_y_lim_plus))
}

plot_all <- function (name_dosses,viz, perf){
    ylim_pluss <- c(0,0,0)
    ylim_moinss <- c(1000000000,1000000000,1000000000)
    for (name_doss in name_dosses){
        current_y_lims <- get_y_lims(name_doss,perf,viz)
        current_ylim_moins <- current_y_lims[[1]]
        current_ylim_plus <- current_y_lims[[2]]
        ylim_moinss = pmin(ylim_moinss,current_ylim_moins)
        ylim_pluss = pmax(ylim_pluss,current_ylim_plus)
    }
    model = "Column-dependent (3b)"
    plots = list()
    for (name_doss in name_dosses){
        plots_and_legend <-  plot_csv(name_doss, viz,model,ylim_moinss, ylim_pluss,perf)
        current_plots = plots_and_legend[[1]]
        legend_all = plots_and_legend[[2]]
        plots = append(current_plots, plots)
    }
    return(list(plots,legend_all))
}

three_plots_and_legend <- plot_all(name_dosses, viz, perf)
three_plots <- three_plots_and_legend[[1]]
legend_all <- three_plots_and_legend[[2]]

grid.arrange(grobs = three_plots, as.table = FALSE, align = c("v"),bottom = legend_all,  ncol = 3,top = title, common.legend = TRUE)
d
