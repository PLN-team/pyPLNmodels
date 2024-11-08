library(ggplot2)
library(ggtext)
library(reshape2)
library(dplyr)
library(latex2exp)
library(viridis)

# filename = "csvs/res_nb_seed_count100_nb_seed_param_1_diagonal.csv"
filename = "csvs/res_nb_seed_count100_diagonal.csv"
p_wanted = 200
wanted_dims = c(1,2,3)
nb_cov_wanted = 3

pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/qqplots.pdf", width = 12, height = 12)

get_df_bound <- function(n_samples) {
  # Set parameters for bounds
  n_sim <- 10000

  # Generate random samples
  all_sim <- matrix(rnorm(n_sim * n_samples), nrow = n_sim, ncol = n_samples)

  # Calculate percentages
  percentages <- (1:n_samples - 0.5) / n_samples * 100

  # Calculate quantiles
  quantiles <- apply(all_sim, 1, function(x) quantile(x, probs = percentages / 100))

  # Calculate lower and upper bounds
  lower_bound <- apply(quantiles, 1, function(x) quantile(x, probs = 0.025))
  upper_bound <- apply(quantiles, 1, function(x) quantile(x, probs = 0.975))

  # Generate a new set of random samples
  theoretical_quantiles <- qnorm(percentages / 100)

  # Create a data frame for plotting bounds
  df_bounds <- data.frame(
    theoretical_quantiles = theoretical_quantiles,
    lower_bound = lower_bound,
    upper_bound = upper_bound
  )
  return(df_bounds)
}


make_qq <- function(dd, x) {
    dd <- dd[order(dd[[x]]), ]
    dd$qq <- qnorm(ppoints(nrow(dd)))
    dd$lower_bound <- df_bounds$lower_bound
    dd$upper_bound <- df_bounds$upper_bound
    return(dd)
}


df = read.csv(filename)
df = subset(df, select = -c(X))
nb_replicates = max(df$seed_count) + 1
df_bounds <- get_df_bound(n_samples = nb_replicates)

df$n_samples = as.factor(df$n_samples)
df$p = as.factor(df$p)
df$seed_count = as.factor(df$seed_count)
df$dim_number = as.factor(df$dim_number)
df$seed_param = as.factor(df$seed_param)
df$nb_cov = as.factor(df$nb_cov)


df <- df[df$dim_number %in% wanted_dims,]
df <- df[df$p  == p_wanted,]
df <- df[df$nb_cov  == nb_cov_wanted,]
print('head')
print(head(df))
df <- droplevels(df)

viridis_colors <- viridis(8)  # Change the number to the number of colors you need
viridis_colors <- viridis_colors[c(2,6)]

df = reshape2::melt(df, id.vars = c("n_samples","p", "dim_number", "nb_cov", "seed_count", "seed_param", "RMSE_B", "RMSE_Sigma"), value.name = "N01" )

df <- df %>% group_by(variable, n_samples, nb_cov, dim_number) %>% do(make_qq(., "N01")) %>% ungroup()

levels(df$dim_number) <- c("1"= TeX("$ B_{1,1}$", bold = TRUE), "2"= TeX("$B_{1,2}$", bold = TRUE), "3"= TeX("$B_{1,3}$", bold = TRUE))
levels(df$n_samples) <- c("1000"= TeX("$n = 1000 $"), "2000"= TeX("$n = 2000$"), "2000"= TeX("$n = 3000$"))
# levels(df$dim_number) <- c("3"= "31", "4"= "32", "5"= "33")

levels(df$variable) <- c("Variational.Fisher.Information" = "Variational Fisher Information", "Sandwich.based.Information" = "Sandwich-based variance")

myqqplot <- ggplot(df, aes(x = qq, y = N01, shape = variable, color = variable)) + geom_point(size = 1.9, alpha = 0.8) +
    facet_grid(dim_number ~ n_samples, labeller = label_parsed) + geom_abline(slope = 1, intercept = 0, linewidth = 0.4) +
    theme_bw() +
    xlab(element_blank()) +
    coord_equal(xlim = c(-3, 3), ylim = c(-3, 3)) +
    theme(legend.position="bottom", legend.direction = "horizontal", legend.box = "horizontal", legend.title = element_blank(), text = element_text(size = rel(4.5)), legend.text = element_text(size = rel(6.5)), strip.text.x = element_text(size = rel(4.5)), strip.text.y = element_text(size = rel(5.5)), axis.text.x = element_text(size = rel(6.5)),axis.text.y = element_text(size = rel(6.5)) ) +
    # scale_color_viridis_d() +
    scale_color_manual(values = viridis_colors) +
    guides(color = guide_legend(override.aes = list(size = 6))) +
    geom_line(aes(y = lower_bound), color = 'red', linetype = "dashed") +
    geom_line(aes(y = upper_bound), color = 'red', linetype = "dashed") +
    ylab(element_blank())

myqqplot
dev.off()
