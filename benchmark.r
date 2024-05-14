library(PLNmodels)
library(gllvm)

df <- read.csv("full_scmark_little.csv")

ns = c(200)
ps = c(20,40,80,150)
columns = c(ns,"dim")
get_empty_df <- function(){
    empty_df <- data.frame(matrix(rep(0,length(ps)*(length(ns) +1)),ncol = (length(ns) + 1)))
    empty_df[,-1] <- ps
    colnames(empty_df)[-1] = "dim"
    return(empty_df)
}
df_pln <- get_empty_df()
df_pln$Model <- "RPLN"
df_plnpca <- get_empty_df()
df_plnpca$Model <- "RPLNPCA"
df_gllvm <- get_empty_df()
df_gllvm$Model <- "GLLVM"


RANK = 2
# ?prepare_data
get_pln_running_time <- function(n,p){
    start = Sys.time()
    Y = df[1:n,1:p]
    covariates = rep(1,n)
    data = prepare_data(Y, covariates)
    pln = PLN("Abundance ~ 1", data)
    return(difftime(Sys.time() , start)[[1]])
}
get_plnpca_running_time <- function(n,p){
    start = Sys.time()
    Y = df[1:n,1:p]
    covariates = rep(1,n)
    data = prepare_data(Y, covariates)
    pln = PLNPCA("Abundance ~ 1", data, ranks = RANK)
    return(difftime(Sys.time() , start)[[1]])
}

get_gllvm_running_time <- function(n,p){
    start = Sys.time()
    Y = as.matrix(df[1:n,1:p])
    gllvm(Y, family = "poisson", num.lv = RANK)
    return(difftime(Sys.time() , start)[[1]])
}


for (i in c(1:length(ns))){
    n = ns[i]
    for (j in c(1:length(ps))){
        p = ps[j]
        print('p:')
        print(p)
        print('n:')
        print(n)
        print('i:')
        print(i)
        print('j:')
        print(j)
        rt_gllvm <- get_gllvm_running_time(n,p)
        df_gllvm[j,i] = rt_gllvm
        print('df ij')
        print(df_gllvm[j,i])
        print('df gllvm')
        print(df_gllvm)
        print('gllvm')
        print(rt_gllvm)
        rt_plnpca <- get_plnpca_running_time(n,p)
        # print('plnpca')
        # print(rt_plnpca)
        rt_pln <- get_pln_running_time(n,p)
        # print('pln')
        # print(rt_pln)
    }
}
