library(PLNmodels)
install.packages("RcppGSL")
library(gllvm)

df <- read.csv("full_scmark_little.csv")

ns = c(150,250)
ps = c(10,20,30,50)#,100,200,300,500)
time_limit = 100


columns = c(ns,"dim")
get_empty_df <- function(){
    empty_df <- data.frame(matrix(rep(0,length(ps)*(length(ns) +1)),ncol = (length(ns) + 1)))
    colnames(empty_df)[-1] = "dim"
    colnames(empty_df)[1:length(ns)] = ns
    empty_df$dim <- ps
    return(empty_df)
}
df_pln <- get_empty_df()
df_pln$Model <- "RPLN"
df_plnpca <- get_empty_df()
df_plnpca$Model <- "RPLNPCA"
df_gllvm <- get_empty_df()
df_gllvm$Model <- "GLLVM"
# other_df_gllvm <- read.table("csv_res_benchmark/df_gllvm.csv")
# other_df_pln <-
# other_df_plnpca <-


RANK_GLL = 2
RANK_PLN = 5
get_pln_running_time <- function(n,p){
    start = Sys.time()
    Y = df[1:n,1:p]
    covariates = rep(1,n)
    data = prepare_data(Y, covariates)
    pln = PLN("Abundance ~ 1", data)
    return(difftime(Sys.time() , start, units = "secs")[[1]])
}
get_plnpca_running_time <- function(n,p){
    start = Sys.time()
    other_start = Sys.time()
    Y = df[1:n,1:p]
    covariates = rep(1,n)
    data = prepare_data(Y, covariates)
    pln = PLNPCA("Abundance ~ 1", data, ranks = RANK_PLN)
    return(difftime(Sys.time() , start, units = "secs")[[1]])
}

get_gllvm_running_time <- function(n,p){
    start = Sys.time()
    other_start = Sys.time()
    Y = as.matrix(df[1:n,1:p])
    gllvm(Y, family = "poisson", num.lv = RANK_GLL)
    return(difftime(Sys.time() , start, units = "secs")[[1]])
}


for (i in c(1:length(ns))){
    n = ns[i]
    keep_going_gll <- TRUE
    keep_going_pln <- TRUE
    keep_going_plnpca <- TRUE
    print('n:')
    print(n)
    for (j in c(1:length(ps))){
        p = ps[j]
        print('p:')
        print(p)
        if (keep_going_gll == TRUE){
            rt_gllvm <- get_gllvm_running_time(n,p)
            if (rt_gllvm > time_limit){
                keep_going_gll <- FALSE
                print('last rt gll')
                print(rt_gllvm)
            }
        }else{
            rt_gllvm <- NaN

        }
        df_gllvm[j,i] = rt_gllvm

        if (keep_going_plnpca == TRUE){
            rt_plnpca <- get_plnpca_running_time(n,p)
            if (rt_plnpca > time_limit){
                keep_going_plnpca <- FALSE
                print('last rt plnpca')
                print(rt_plnpca)
            }
        }else{
            rt_plnpca <- NaN
        }
        df_plnpca[j,i] = rt_plnpca


        if (keep_going_pln == TRUE){
            rt_pln <- get_pln_running_time(n,p)
            if (rt_pln > time_limit){
                keep_going_pln <- FALSE
                print('last rt pln')
                print(rt_pln)
            }
        }else{
            rt_pln <- NaN
        }
        df_pln[j,i] = rt_pln
    }
}


print('df _gll')
print(df_gllvm)
print("df pln")
print(df_pln)
print("df plnpca")
print(df_plnpca)


write.csv(df_gllvm, "csv_res_benchmark/df_gllvm.csv")
write.csv(df_pln, "csv_res_benchmark/df_pln_r.csv")
write.csv(df_plnpca, "csv_res_benchmark/df_plnpca_r.csv")
