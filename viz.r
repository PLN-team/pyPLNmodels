library(ggplot2)


file_pln = "csv_res_benchmark/python_pln.csv"
file_plnpca = "csv_res_benchmark/python_plnpca.csv"


plot_file <- function(filename){
    df = read.csv(filename, header = TRUE, check.names = F)
    df <- data.frame(df[,-1], row.names = df[,1], check.names = F)
    print('df')
    print(df)
}
# ?read.csv
plot_file(file_pln)
