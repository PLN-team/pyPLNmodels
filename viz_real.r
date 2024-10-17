library(ggplot2)

df = read.csv("csv_ic/real_ic_n_1000.csv")
df = subset(df, select = -c(X))
print('df')
print(df)
df$label <- c(1:dim(df)[1])
df = df[order(df$coef),]
df$ll <- df$coef + df$ll
df$hh <- df$coef + df$hh

df$label <- factor(df$label, levels=rev(df$label))

fp <- ggplot(data=df, aes(x=label, y=coef, ymin=ll, ymax=hh)) +
        geom_pointrange(aes(ymin = ll, ymax = hh), size = 0.1) +
        geom_hline(yintercept=1, lty=2) +  # add a dotted line at x=1 after flip
        coord_flip() +  # flip coordinates (puts labels on y axis)
        xlab("Label") + ylab("Mean (95% CI)") +
        theme_bw()  # use a white background
print(fp)
