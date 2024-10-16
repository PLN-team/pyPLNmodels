library(ggplot2)
library(grid)
library(gridExtra)
# install.packages("cowplot")
library(cowplot)
library(latex2exp)




filename = "coverage_groups.csv"
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

df["method"][df["method"] == "Variational Fisher"] = "Variational Fisher Information"
df["method"][df["method"] == "Sandwich"] = "Sandwich based estimation"


label_equal <- function(variable, value){
    if (variable == "d"){
        variable = "m"
    }
    return(paste(variable, "=", value))
}
plot_ns_d_p_facet <- function(){
    current_plot <- ggplot(df, aes(x = n, y  = Coverage, fill = method)) +
                    geom_violin(position="dodge", alpha=0.5) +
                    ylim(0.1,1) +
                    theme_bw() +
                    scale_fill_viridis_d() +
                    guides(fill=guide_legend(title="Method", nrow = 1, byrow = TRUE)) +
                    geom_hline(yintercept=0.95, color = "red",linetype = "dashed") +
                    theme(plot.margin=unit(c(0.05,0.2,0,0.2), "cm")) +
                    facet_grid(d ~ p, labeller = label_equal) +
                    ylab("Coverage") +
                    xlab(TeX("Number of samples $n$")) +
                    theme(plot.title = element_text(hjust = 0.5), legend.position="bottom", legend.title = element_blank())
    return(current_plot)
}
print("nb replicates")
print(nb_replicates)
pdf("figure_sandwich/coverage.pdf")
plot_ns_d_p_facet()

# title <- paste("Coverage for ", nb_replicates ," replicates on B", sep = "")

# grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE)

dev.off()
