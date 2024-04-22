library(ggplot2)
library(cowplot)
library(gridExtra)
library(hash)
library(latex2exp)
library(grid)
library(glue)
# library(naniar)

options(error=traceback)
# traceback()

viz = "dims"
perf = "computation"

pdf(paste("figures/",viz,"_",perf,".pdf",sep=""), height = 10, width = 10)

get_name_computation <- function(viz,formula){
    if (viz == "dims"){
        # return(paste(viz,formula,"not_n_or_p_1000.csv", sep = "_"))
        return(paste(viz,formula,"not_n_or_p_500.csv", sep = "_"))
    }
    else{
        # return(paste(viz,formula,"not_n_or_p_250.csv", sep = "_"))
        return(paste(viz,formula,"not_n_or_p_150.csv", sep = "_"))
    }
}

if (viz =="samples" || viz == "dims"){
    name_doss_1 <- get_name_computation(viz,"global")
    name_doss_2 <- get_name_computation(viz,"column-wise")
    name_doss_3 <- get_name_computation(viz,"row-wise")
} else{
    name_doss_1 = paste(viz,"_viz_global.csv", sep = "")
    name_doss_2 = paste(viz,"_viz_column-wise.csv", sep = "")
    name_doss_3 = paste(viz,"_viz_row-wise.csv", sep = "")
}
name_dosses = c(name_doss_1,name_doss_2,name_doss_3)



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


# if (!(perf == "stat" & viz == "samples") & perf != "computation"){
# # if (perf != "stat" || viz != "samples"){
#    colors =  c("skyblue","blue","yellow","orange")
# } else{
    # colors = c("skyblue","blue","black","gray","yellow","orange","green")
# }


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
        columns = c(columns,"RMSE_OMEGA","RMSE_PI","RMSE_B")
        # columns = c(columns, "RMSE_SIGMA","RMSE_PI", "RMSE_B")
    }
    if (perf == "computation"){
        if (viz == "poisson" || viz == "proba"){
            columns = c(columns, "moyenne")
        }
        else{
            columns = c(columns,"xscale")
        }
        columns = c(columns,"ELBO","TIME","Reconstruction_error")
    }
    if (perf == "annexe"){
        columns = c(columns,"moyenne","RMSE_OMEGA","RMSE_PI","RMSE_B","RMSE_B0")
    }
    df = subset(read.csv(paste('csv_data/',namedoss, sep = "")), select = -X)
    for (column in colnames(df)){
        df[df == 666] = NA
        }
    df[,"model_name"] = as.factor(df[,"model_name"])
    if ("moyenne" %in% columns){
        df[,"moyenne"] = as.factor(df[,"moyenne"])
    }
    else{
        df[,"xscale"] = as.factor(df[,"xscale"])
    }

    df[,"NBITER"] = as.numeric(df[,"NBITER"])
    df = df[columns]
    # if (perf != "stat" || viz != "samples"){
    # if (!(perf == "stat" & viz == "samples") & perf != "computation"){
        ### If fair_pln and stuff needed only in the RMSE Y,
        ### Maybe put to NAN the needed points. Just like pi in samples stat.
        # df = df[df["model_name"] != "fair_Pln",]
        # df = df[df["model_name"] != "Fair Pln",]
        # df = df[df["model_name"] != "Pln",]
    # }
    return(df)
}

plot_csv = function(namedoss,viz,inflation, list_ylim_moins, list_ylim_plus, perf){
    df = get_df(namedoss,perf,viz)
    third_column = names(df)[3]
    last_column = names(df)[length(names(df))]
    criterions <- names(df[,c(3:(dim(df)[2]))])
    if (viz == "poisson"){
        xlab = TeX('$XB$')
    }
    if (viz == "proba"){
        xlab = TeX('$\\pi')
    }
    if (viz == "samples"){
        xlab = TeX('$n$')
    }
    if (viz == "dims"){
        xlab = TeX('$p$')
    }
    plot_data_column = function(i){
        column = criterions[i]
        y_moins = list_ylim_moins[[i]]
        y_plus = list_ylim_plus[[i]]
        if (viz == "poisson" || viz == "proba"){
            first_col = "moyenne"
        } else{
            first_col = "xscale"
        }

        current_plot <- (ggplot(df, aes(x = df[,first_col], y = df[,column], fill =
                                        as.factor(model_name)) )
                         + geom_point(position = position_jitterdodge(), aes(color =
                                                                     model_name,group
                                                                 = model_name,
                                                                 ),
                                    size = 0.8, alpha = 0.6)
                         + geom_boxplot(lwd = 0.03, outlier.shape = NA)
                         + scale_y_log10()
                         + scale_fill_viridis_d(name = "")
                         # + scale_fill_viridis(name = "")
                         + scale_colour_viridis_d(name = "")
                        # + scale_color_manual(values = colors,name ="")
                        + scale_x_discrete(labels=scaleFUN)
        +guides(fill=guide_legend(nrow=2,byrow=TRUE)) ) + theme_bw()+
         theme(legend.key.size = unit(1.5,"cm"),legend.text = element_text(size=25) )
        if (column != "RMSE_B0" & column != "RMSE_PI"){
            # if (column == "RMSE_PI"){
            #     tmp_y_moins = max(y_moins, 1e-3)
            #     current_plot = current_plot +  scale_y_log10(limits = c(tmp_y_moins,y_plus))
            # }
            # else{
            current_plot = current_plot +  scale_y_log10(limits = c(y_moins,y_plus))
            # }
        }
        if (column == third_column || column == "RMSE_B0"){
            current_plot = current_plot + ggtitle(inflation)+ theme(plot.title = element_text(hjust = 0.5))
        }
        if (column == last_column || column == "RMSE_B"){
            current_plot = current_plot + labs(y = h[[column]],x = xlab)
        }
        else{

            if(inflation == "Non-dependent (3a)"){
                current_plot = current_plot + labs(y=h[[column]], x = NULL)
                # current_plot = current_plot + labs(y=h[[column]], x = NULL)
            }
            else{
                current_plot = current_plot + labs(y= NULL,x = NULL)
            }
        }
        if(column == last_column && inflation != "Non-dependent (3a)"){
            current_plot = current_plot + labs(y=NULL, x = xlab)
        }
        # print('column')
        # print(column)
        return(current_plot)
         #+
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
    criterions <- names(df[,c(3:(dim(df)[2]))])
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
    models = c("Non-dependent (3a)","Column-dependent (3b)","Row-dependent (3c)")
    plots = list()
    for (i in rev(c(1:(length(models))))){
        name_doss = name_dosses[i]
        model = models[i]
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
grid.arrange(grobs = three_plots, as.table = FALSE, align = c("v"),bottom = legend_all,  ncol = 3,top = title, common.legend = TRUE)
dev.off()
