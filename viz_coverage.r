library(ggplot2)
library(grid)
library(gridExtra)
# install.packages("cowplot")
library(cowplot)




filename = "coverage.csv"
df = read.csv(filename)
df$n = as.factor(df$n)
df$p = as.factor(df$p)
df$d = as.factor(df$d)
df$seed = as.factor(df$seed)


ds = unique(df$d)
ns = unique(df$n)
ps = unique(df$p)

remove_legend <- function(myplot){
                return(myplot + theme(legend.position="none"))
}


plot_ns_d_p <- function(d,p){
    selected <- (df$d == d) & (df$p == p)
    sub_df <- df[selected,]
    current_plot <- ggplot(sub_df, aes(x = n, y  = Coverage, fill = method)) + geom_violin(position="dodge", alpha=0.5)
    # current_plot <- current_plot + stat_summary(fun.data = mean_sdl)
    current_plot <- current_plot + ylim(0.1,1) + theme_bw() + scale_fill_viridis_d()
    current_plot <- current_plot + guides(fill=guide_legend(title="Method", nrow = 1, byrow = TRUE))
    current_plot <- current_plot + geom_hline(yintercept=0.95, color = "red")
    if (p == ps[[1]]){
        current_plot <- current_plot + ylab(paste("Coverage, d =",as.character(d)))
    }
    else{
        current_plot <- current_plot + ylab("")
    }

    if (d != ds[[3]]){
        current_plot <- current_plot + xlab("")
    }
    if (d == ds[[1]]){
        print('p:')
        print(p)
        current_plot <- current_plot + ggtitle(paste("p =",as.character(p)))+
              theme(plot.title = element_text(hjust = 0.5))

    }
    return(current_plot)
}

pdf("plots/coverage.pdf")
plots = list()
for (d in rev(levels(df$d))){
    for (p in rev(levels(df$p))){
        current_plot <- plot_ns_d_p(d,p)
        common_legend <- get_legend(current_plot)
        tmp_plot <- remove_legend(current_plot)
        plots <- append(list(tmp_plot), plots)
    }
}
grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE,top=textGrob("Coverage for 10 replicates on B",gp=gpar(fontsize=20,font=3)))

dev.off()
