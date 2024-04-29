# install.packages("insight")
library(kableExtra)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(gridExtra)
# library(insight)

viz = "samples"
pdf(paste("figures/",viz,"_elbos.pdf",sep=""), width = 20)
dev.off()


NB_DECIMAL = 3

get_name_computation <- function(viz,formula){
    if (viz == "dims"){
        return(paste(viz,formula,"not_n_or_p_500.csv", sep = "_"))
    }
    else{
        return(paste(viz,formula,"not_n_or_p_150.csv", sep = "_"))
    }
}


if (viz =="samples" || viz == "dims"){
    name_doss_1 <- get_name_computation(viz,"global")
    name_doss_2 <- get_name_computation(viz,"column-wise")
    name_doss_3 <- get_name_computation(viz,"row-wise")
} else{
    name_doss_1 = paste(viz,"_viz_global.csv", sep = "")
    name_doss_2 = paste(viz,"_viz_column-wise.csv", sep = "")
    name_doss_3 = paste(viz,"_viz_row-wise.csv", sep = "")
}


get_df = function(namedoss, viz){
    columns = c("model_name")


    columns = c(columns, "absc")
    columns = c(columns,"ELBO")
    df = subset(read.csv(paste('csv_data/',namedoss, sep = "")), select = -X)
    colnames(df)[2] <- "absc"
    for (column in colnames(df)){
        df[df == 666] = NA
        }
    # df[,"model_name"] = as.factor(df[,"model_name"])


    df["absc"] = as.factor(df[,"absc"])
    df = df[columns]
    return(df)
}
get_merged_df <- function(name_doss_1, name_doss_2, name_doss_3, viz){
    global_df <- get_df(name_doss_1, viz)
    global_df["formula"] = "global"

    column_df <- get_df(name_doss_2, viz)
    column_df["formula"] = "column"

    row_df <- get_df(name_doss_3, viz)
    row_df["formula"] = "row"

    all_df <- rbind(global_df, column_df, row_df)
    return(all_df)
}

merged_df <- get_merged_df(name_doss_1, name_doss_2, name_doss_3, viz)
# print(merged_df)

format_df <- function(df,apply_variance,nb_bootstrap){
    df <- df %>%  pivot_wider(names_from = model_name,
                                    values_from = ELBO)
    df <- data.frame(df)
    rownames(df) <- df[,1]
    df[,1] <- NULL
    if (apply_variance == TRUE){
        df <- 1.96/sqrt(nb_bootstrap)*df
    }
    df <- round(df, NB_DECIMAL)
    df <- df %>% mutate(across(everything(), as.character))
    return(df)

}

get_one_table <-function(merged_df_, oneviz){
    sub_df <- merged_df_[merged_df_["absc"] == oneviz,]
    model_name_formula_df <-data.frame(Map(paste, setNames(sub_df["formula"], paste0("cat_", names(sub_df["formula"]))), sub_df["model_name"], MoreArgs = list(sep = "_")))
    sub_df["model_x_formula"] = model_name_formula_df
    nb_bootstrap <- sum(sub_df["formula"] == "global" & sub_df["model_name"] == "Enhanced")
    agg_df_mean <- aggregate(ELBO ~ model_name + formula, data = sub_df, FUN = mean)
    agg_df_std <- aggregate(ELBO ~ model_name + formula, data = sub_df, FUN = sd)
    # agg_df_std <- 1.96/sqrt(nb_bootstrap)*agg_df_std

    agg_df_mean <- format_df(agg_df_mean, apply_variance = FALSE,nb_bootstrap=NULL)
    agg_df_std <- format_df(agg_df_std, apply_variance = TRUE,nb_bootstrap=nb_bootstrap)
    # print(agg_df_mean)
    end_df <- data.frame(Map(paste, setNames(agg_df_mean, names(agg_df_mean)), agg_df_std,
                        MoreArgs = list(sep = "\u00b1")))
    first_lines <- transpose(data.frame(colnames(end_df)))
    # end_lines <- rbind(first_lines,end_df)
    rownames(end_df) <- c("Non-dependent", "Column-dependent", "row-dependent")
    return(end_df)
    # return()
    # end_df <- paste0(agg_df_mean, "\u00b1", agg_df_std)
    # return(end_df)
    # titles <-

}
first_table <- get_one_table(merged_df,100)
# second_table <- get_one_table(merged_df,200)

first_table %>%
  kbl(caption = "Recreating booktabs style table") %>%
  kable_classic(full_width = F, html_font = "Cambria")
