library(ggplot2)
library(grid)
library(gridExtra)
library(dplyr)
library(cowplot)
library(latex2exp)
library(viridis)




filename = "csvs/res_nb_seed_count100_diagonal.csv"

df = read.csv(filename)
df <- subset(df, select = -c(X,Sandwich.based.Information,Variational.Fisher.Information))

# head(df)
# df <- df[df$RMSE_B < 0.13,]
# df <- df[df$RMSE_Sigma < 0.13,]

df$n_samples = as.factor(df$n_samples)
df$p = as.factor(df$p)
df$nb_cov = as.factor(df$nb_cov)

nb_replicates = max(df$seed_param) + 1
df$seed_param = as.factor(df$seed_param)

# df["param"][df["param"] == "coef"] = "B"#TeX("RMSE $B$")
# df["param"][df["param"] == "sigma"] = "test"#TeX("RMSE $\\Sigma$")

ds = unique(df$nb_cov)
ns = unique(df$n_samples)
ps = unique(df$p)


label_equal <- function(variable, value){
    if (variable == "nb_cov"){
        variable = "m"
    }
    return(paste(variable, "=", value))
}

df = reshape2::melt(df, id.vars = c("n_samples","p", "dim_number", "nb_cov", "seed_count", "seed_param"), value.name = "RMSE")


df <- df %>% group_by(n_samples, nb_cov ,variable, p, dim_number, seed_count, seed_param) %>% summarise( RMSE = mean(RMSE)) %>% ungroup()
print('head')
print(head(df))

viridis_colors <- viridis(8)  # Change the number to the number of colors you need
viridis_colors <- viridis_colors[c(2,6)]

plot_ns_d_p_facet <- function() {
    current_plot <- ggplot(df, aes(x = n_samples, y  = RMSE, fill = variable)) +
                    geom_violin(position="dodge", alpha=0.5) +
                    guides(fill=guide_legend(title="", nrow = 1, byrow = TRUE)) +
                    theme_bw() +
                    # scale_fill_viridis_d(labels = unname(TeX(c("RMSE$(\\hat{B} - B^{*})$", "RMSE$(\\hat{\\Sigma} - \\Sigma^{*})$")))) +
                    scale_fill_manual(labels = unname(TeX(c("RMSE$(\\hat{B} - B^{*})$", "RMSE$(\\hat{\\Sigma} - \\Sigma^{*})$"))), values = viridis_colors) +
                    facet_grid(nb_cov ~ p, scales = "free", labeller = label_equal) +
                    ylab("RMSE") +
                    xlab(TeX("Number of samples $n$")) +
                    theme(legend.position="bottom", legend.title = element_blank(), legend.text = element_text(size = rel(4.5)), text = element_text(size = rel(4.5)), strip.text.x = element_text(size = rel(3.5)))+
                    scale_y_continuous(limits = c(0.025,0.13))
    return(current_plot)
}



pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/rmse.pdf", width = 10, height = 8)
plot_ns_d_p_facet()
dev.off()

# pdf("figure_sandwich/rmse.pdf")

# grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE,top=textGrob(title,gp=gpar(fontsize=20,font=3)))
# grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE)
