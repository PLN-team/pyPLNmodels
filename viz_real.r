library(ggplot2)
library(latex2exp)

df = read.csv("csv_ic/real_ic_n_1000.csv")
df = subset(df, select = -c(X))
print('df')
print(head(df))
df$label <- c(1:dim(df)[1])

df$cell_label <- 0
df$cell_label[df$group == 0] <- "CD8+"
df$cell_label[df$group == 1] <- "CD4+"
df$cell_label[df$group == 2] <- "Mac"
df$label <- paste(as.character(df$cell_label), as.character(df$dimensions), sep = ":")

# df$label <- paste(as.character(df$groups),as.character(df$dimensions), sep = ":")
df = df[order(df$coef),]
df$ll <- df$coef + df$ll
df$hh <- df$coef + df$hh

df$label <- factor(df$label, levels=rev(df$label))
df$groups <- as.factor(df$groups)
levels(df$groups) <- c( "0" = "T cells CD8+","1" = "T cells CD4+", "2" = "Macrophages")

pdf("/home/bastien/These/manuscript-sandwich-estimators/figures/real_coverage.pdf")
fp <- ggplot(data=df, aes(x=label, y=coef, ymin=ll, ymax=hh)) +
        theme_bw()+
        coord_flip() +
        theme(axis.text.y = element_text(size = 5), legend.position = "bottom") +
        geom_pointrange(aes(ymin = ll, ymax = hh), size = 0.1, alpha = 0.5) +
        geom_hline(yintercept=0, lty=2) +
        geom_point(aes(color=groups), size=0.5)+
        guides(color = guide_legend(title = NULL)) +
        scale_colour_viridis_d() +
        xlab("Cell type:dimension") + ylab(TeX("95% CI of regression parameter $B$", bold = TRUE))
print(fp)
dev.off()
# fp <- ggplot(data=df, aes(x=label, y=coef, ymin=ll, ymax=hh)) +
#         theme(axis.text.x = element_text(size = 12),  # change x-axis labels size
#               axis.text.y = element_text(size = 12),  # change y-axis labels size
#               axis.title.y = element_text(size = 14, face = "bold")) +
#         geom_pointrange(aes(ymin = ll, ymax = hh), size = 0.1) +
#         geom_hline(yintercept=1, lty=2) +  # add a dotted line at x=1 after flip
#         geom_point(aes(color=groups), size=0.5) +  # add points
#         coord_flip() +  # flip coordinates (puts labels on y axis)
#         xlab("Label") + ylab("95% CI") +
#         theme_bw()  # use a white background
# print(fp)
