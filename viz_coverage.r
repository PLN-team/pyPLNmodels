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
nb_replicates = max(df$seed) + 1
print('nb replicates')
print(nb_replicates)
df$seed = as.factor(df$seed)


ds = unique(df$d)
ns = unique(df$n)
ps = unique(df$p)

remove_legend <- function(myplot){
                return(myplot + theme(legend.position="none"))
}
# print('here')
# print('df')
# print(df)
# print(df["method"][df["method"] == "Variational Fisher"])
df["method"][df["method"] == "Variational Fisher"] = "Variational Fisher Information"
df["method"][df["method"] == "Sandwich"] = "Sandwich based estimation"


plot_ns_d_p <- function(d,p){
    selected <- (df$d == d) & (df$p == p)
    sub_df <- df[selected,]
    current_plot <- ggplot(sub_df, aes(x = n, y  = Coverage, fill = method)) + geom_violin(position="dodge", alpha=0.5)
    current_plot <- current_plot + ylim(0.1,1) + theme_bw() + scale_fill_viridis_d()
    current_plot <- current_plot + guides(fill=guide_legend(title="Method", nrow = 1, byrow = TRUE))
    current_plot <- current_plot + geom_hline(yintercept=0.95, color = "red")
    current_plot <- current_plot + theme(plot.margin=unit(c(0.05,0.2,0,0.2), "cm"))
    # current_plot <- current_plot + scale_y_continuous(breaks = c(0.25, 0.5, 0.75, 0.95, 1))
    if (p == ps[[1]]){
        current_plot <- current_plot + ylab(paste("Coverage, m =",as.character(d)))
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

pdf("/home/bastien/These/manuscript/tex/figures/appendix//coverage.pdf")
plots = list()
for (d in rev(levels(df$d))){
    for (p in rev(levels(df$p))){
        current_plot <- plot_ns_d_p(d,p)
        common_legend <- get_legend(current_plot)
        tmp_plot <- remove_legend(current_plot)
        plots <- append(list(tmp_plot), plots)
    }
}
print("nb replicates")
print(nb_replicates)

# title <- paste("Coverage for ", nb_replicates ," replicates on B", sep = "")

grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE)

dev.off()
