library(ggplot2)
library(ggtext)
library(reshape2)
library(dplyr)
library(latex2exp)

get_ks_p_value <- function(vector) {
  ks_result <- ks.test(vector, "pnorm", mean=0, sd=1)
  return(ks_result$p.value)
}

pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/ks.pdf")

filename = "csvs/res_nb_seed_count5_nb_seed_param_1.csv"

df = read.csv(filename)
df = subset(df, select = -c(X))

df$n_samples = as.factor(df$n_samples)
df$p = as.factor(df$p)
df$seed_count = as.factor(df$seed_count)
df$dim_number = as.factor(df$dim_number)
df$seed_param = as.factor(df$seed_param)
df$nb_cov = as.factor(df$nb_cov)



df = reshape2::melt(df, id.vars = c("n_samples","p", "dim_number", "nb_cov", "seed_count", "seed_param", "RMSE_B", "RMSE_Sigma"), value.name = "N01" )

df <- df %>% group_by(n_samples, nb_cov ,variable, p, dim_number) %>% summarise(ks = get_ks_p_value(N01)) %>% ungroup()


# levels(df$n_samples) <- c("500"= TeX("$N = 500 $"), "700"= TeX("$N = 700$"))
levels(df$p) <- c("25"= TeX("$p = 25$"), "50"= TeX("$p = 50$"))
levels(df$nb_cov) <- c("1"= TeX("$d = 1$"), "2"= TeX("$d = 2$"), "3" = TeX("$d = 3$"))
levels(df$variable) <- c("Variational.Fisher.Information" = "Variational Fisher Information", "Sandwich.based.Information" = "Sandwich-based variance")

myqqplot <- ggplot(df, aes(x = n_samples, y = ks,  fill = variable)) +
    geom_boxplot(lwd = 0.1, outlier.shape= NA) +
    facet_grid(nb_cov ~ p, labeller = label_parsed) +
    scale_fill_viridis_d() +
    theme_bw() +
    theme(legend.position="bottom", legend.direction = "horizontal", legend.box = "horizontal", legend.title = element_blank()) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", linewidth = 0.3) +
    xlab(element_text("n")) +
    ylab(element_text("p-value"))

myqqplot


dev.off()
