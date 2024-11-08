library(ggplot2)
library(ggtext)
library(reshape2)
library(dplyr)
library(latex2exp)
library(viridis)

pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/coverage.pdf")

get_ks_p_value <- function(vector) {
  ks_result <- ks.test(vector, "pnorm", mean=0, sd=1)
  return(ks_result$p.value)
}


# filename = "csvs/res_nb_seed_count100_nb_seed_param_1_diagonal.csv"
filename = "csvs/res_nb_seed_count100_diagonal.csv"

df = read.csv(filename)
df = subset(df, select = -c(X))

df$n_samples = as.factor(df$n_samples)
df$p = as.factor(df$p)
df$seed_count = as.factor(df$seed_count)
df$dim_number = as.factor(df$dim_number)
df$seed_param = as.factor(df$seed_param)
df$nb_cov = as.factor(df$nb_cov)


viridis_colors <- viridis(8)  # Change the number to the number of colors you need
viridis_colors <- viridis_colors[c(2,6)]

df = reshape2::melt(df, id.vars = c("n_samples","p", "dim_number", "nb_cov", "seed_count", "seed_param", "RMSE_B", "RMSE_Sigma"), value.name = "N01" )

df <- df %>% group_by(n_samples, nb_cov ,variable, p, dim_number) %>%
        summarise(cover = sum(N01 > qnorm(0.025) & N01 < qnorm(0.975)) / n()) %>% ungroup()

custom_summary <- function(x) {
  m <- mean(x)
  ymin <- m - 0.1  # Adjust this value to control the height
  ymax <- m + 0.1  # Adjust this value to control the height
  return(c(y = m, ymin = ymin, ymax = ymax))
}
summary_stats <- df %>%
  group_by(n_samples, variable, nb_cov, p) %>%
  summarise(custom_summary(cover))

# levels(df$n_samples) <- c("500"= TeX("$N = 500 $"), "700"= TeX("$N = 700$"))
levels(df$p) <- c("50"= TeX("$p = 50$"), "100"= TeX("$p = 100$"), "200"= TeX("$p = 200$"))
levels(df$nb_cov) <- c("1"= TeX("$m = 1$"), "2"= TeX("$m = 2$"), "3" = TeX("$m = 3$"), "4" = TeX("$m = 4$"))
levels(df$variable) <- c("Variational.Fisher.Information" = "Variational Fisher Information", "Sandwich.based.Information" = "Sandwich-based variance")

myqqplot <- ggplot(df, aes(x = n_samples, y = cover,  fill = variable)) +
    geom_violin(alpha = 0.5) +
    stat_summary(fun = mean, geom = "crossbar", size = 0.25, width = 0.5, color = "black", position = position_dodge(width = 0.9), show.legend = FALSE) +
    # geom_errorbar(data = summary_stats, aes(y = y, ymin = ymin, ymax = ymax), width = 0.5, color = "black", position = position_dodge(width = 0.9)) +
    facet_grid(nb_cov ~ p, labeller = label_parsed) +
    # scale_fill_viridis_d() +
    scale_fill_manual(values = viridis_colors) +
    theme_bw() +
    theme(legend.position="bottom", legend.direction = "horizontal", legend.box = "horizontal", legend.title = element_blank()) +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "red", linewidth = 0.3) +
    xlab(element_text("n")) +
    ylab(element_text("p-value"))
myqqplot
dev.off()
