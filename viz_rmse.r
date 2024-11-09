library(ggplot2)
library(grid)
library(gridExtra)
library(dplyr)
library(cowplot)
library(latex2exp)
library(viridis)

loglog = TRUE

if (loglog == TRUE){
    filename = "csvs/res_nb_seed_count100_dims_[100]diagonal.csv"
}else{
    filename = "csvs/res_nb_seed_count100_diagonal.csv"
}
df = read.csv(filename)
df <- subset(df, select = -c(X,Sandwich.based.Information,Variational.Fisher.Information))
df$theoretical_rmse <- 2.2 / sqrt(df$n_samples)  # Adjust the constant as needed
# df$theoretical_rmse <- 0.13

# head(df)
# df <- df[df$RMSE_B < 0.13,]
# df <- df[df$RMSE_Sigma < 0.13,]
if (loglog == TRUE){
    # df$n_samples = log(df$n_samples)
}
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

df = reshape2::melt(df, id.vars = c("n_samples","p", "dim_number", "nb_cov", "seed_count", "seed_param", "theoretical_rmse"), value.name = "RMSE")
if (loglog == TRUE){
    # df$RMSE = log(df$RMSE)
}


df <- df %>% group_by(n_samples, nb_cov ,variable, p, dim_number, seed_count, seed_param, theoretical_rmse) %>% summarise( RMSE = mean(RMSE)) %>% ungroup()
print('head')
print(head(df))

viridis_colors <- viridis(8)  # Change the number to the number of colors you need
viridis_colors <- viridis_colors[c(2,6)]

plot_ns_d_p_facet <- function() {
    current_plot <- ggplot(df, aes(x = n_samples, y  = RMSE, fill = variable)) +
                    geom_violin(position="dodge", alpha=0.5) +
                    guides(fill=guide_legend(title="", nrow = 1, byrow = TRUE)) +
                    theme_bw() +
                    # scale_x_log10() +
                    # scale_fill_viridis_d(labels = unname(TeX(c("RMSE$(\\hat{B} - B^{*})$", "RMSE$(\\hat{\\Sigma} - \\Sigma^{*})$")))) +
                    scale_fill_manual(labels = unname(TeX(c("RMSE$(\\hat{B} - B^{*})$", "RMSE$(\\hat{\\Sigma} - \\Sigma^{*})$"))), values = viridis_colors) +
                    facet_grid(nb_cov ~ p, scales = "free", labeller = label_equal) +
                    ylab("RMSE") +
                    xlab(TeX("Number of samples $n$")) +
                    theme(legend.position="bottom", legend.title = element_blank(), legend.text = element_text(size = rel(4.5)), text = element_text(size = rel(4.5)), strip.text.x = element_text(size = rel(3.5)))
                    # scale_y_continuous(limits = c(0.025,0.13))
                    # geom_line(aes(y = 0.1, color = "Theoretical RMSE"), linewidth = 1.5, linetype = "dashed")
    if (loglog == TRUE){
        current_plot = current_plot + theme(legend.key.width = unit(4, "line"), legend.box.just = "center")+ geom_line(aes(x = as.numeric(df$n_samples), y = df$theoretical_rmse, color = "Theoretical RMSE"), linewidth = 1, linetype = "dashed")
        current_plot = current_plot + scale_color_manual(values = c("Theoretical RMSE" = "red"), labels = c("Theoretical RMSE" = TeX("$y = 2.2/\\sqrt{n}")))

    }
    return(current_plot)
}


if (loglog == TRUE){
    pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/rmse_loglog.pdf", width = 10, height = 8)
} else{
    pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/rmse.pdf", width = 10, height = 8)
}
plot_ns_d_p_facet()
dev.off()

# pdf("figure_sandwich/rmse.pdf")

# grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE,top=textGrob(title,gp=gpar(fontsize=20,font=3)))
# grid.arrange(grobs = plots, as.table = TRUE, align = c("v"), ncol = 3, bottom = common_legend, common.legend = TRUE)
