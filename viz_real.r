library(ggplot2)
library(latex2exp)

df = read.csv("csv_ic/real_ic_n_5000.csv")
df = subset(df, select = -c(X))
print('df')
print(head(df))
df$label <- rep(c(1:as.integer(dim(df)/3)[1]), 3)
df$dimension <- rep(c(1:as.integer(dim(df)/3)[1]), 3)
df$dimension <- as.factor(df$dimension)

df$cell_label <- 0
df$cell_label[df$group == 0] <- "T cells CD8+"
df$cell_label[df$group == 1] <- "T cells CD4+"
df$cell_label[df$group == 2] <- "Macrophages"
df$label <- paste(as.character(df$cell_label), as.character(df$dimensions), sep = ":")
print('head')
print(head(df))
threshold <- 0.6
dims_wanted = df[df$coef <threshold & df$coef > 0 & df$cell_label == "T cells CD8+",]$dimension
print('dims_wanted')
print(dims_wanted)
df <- df[df$dimension %in% dims_wanted,]
df <- df[df$dimension %in% dims_wanted,]

df = df[order(df$coef),]
df$ll <- df$coef + df$ll
df$hh <- df$coef + df$hh

# Add significance column
df$significance <- ifelse(df$ll > 0, "Significantly positive", ifelse(df$hh < 0, "Significantly negative", "Not Significant"))

df$label <- factor(df$label, levels=rev(df$label))
df$groups <- as.factor(df$groups)
levels(df$groups) <- c( "0" = "T cells CD8+","1" = "T cells CD4+", "2" = "Macrophages")

pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/real_coverage.pdf")
fp <- ggplot(data=df, aes(x=dimension, y=coef, ymin=ll, ymax=hh)) +
        theme_bw()+
        coord_flip() +
        theme(axis.text.y = element_text(size = 8), legend.position = "bottom") +
        geom_pointrange(aes(ymin = ll, ymax = hh), size = 0.1, alpha = 0.5) +
        geom_hline(yintercept=0, lty=2) +
        geom_point(aes(color=significance), size=1.5)+
        guides(color = guide_legend(title = NULL)) +
        scale_colour_viridis_d() +
        xlab("Variable number") + ylab(TeX("95% CI of regression parameter $B$", bold = TRUE)) +
        facet_grid(. ~ cell_label, scales = "free_y")
print(fp)
dev.off()
