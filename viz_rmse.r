library(ggplot2)
library(grid)
library(gridExtra)
# install.packages("cowplot")
library(cowplot)
library(latex2exp)




filename = "rmse_groups.csv"
df = read.csv(filename)
df$n = as.factor(df$n)
df$p = as.factor(df$p)
df$d = as.factor(df$d)

nb_replicates = max(df$seed) + 1
df$seed = as.factor(df$seed)

print(df)
# df["param"][df["param"] == "coef"] = "B"#TeX("RMSE $B$")
# df["param"][df["param"] == "sigma"] = "test"#TeX("RMSE $\\Sigma$")

ds = unique(df$d)
ns = unique(df$n)
ps = unique(df$p)


label_equal <- function(variable, value){
    if (variable == "d"){
        variable = "m"
    }
    return(paste(variable, "=", value))
}
plot_ns_d_p_facet <- function() {
    current_plot <- ggplot(df, aes(x = n, y  = RMSE, fill = param)) +
                    geom_violin(position="dodge", alpha=0.5) +
                    guides(fill=guide_legend(title="", nrow = 1, byrow = TRUE)) +
                    theme_bw() +
                    scale_fill_viridis_d(labels = unname(TeX(c("RMSE$(\\hat{B} - B^{*})$", "RMSE$(\\hat{\\Sigma} - \\Sigma^{*})$")))) +
                    facet_grid(d ~ p, scales = "free", labeller = label_equal) +
                    ylab("RMSE") +
                    xlab(TeX("Number of samples $n$")) +
                    theme(legend.position="bottom", legend.title = element_blank())+
                    scale_y_continuous(limits = c(0.025,0.21))
    return(current_plot)
}

pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/rmse.pdf")
plot_ns_d_p_facet()
# print(current_plot)
dev.off()

# pdf("figure_sandwich/rmse.pdf")

# grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE,top=textGrob(title,gp=gpar(fontsize=20,font=3)))
# grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE)
