#!/usr/bin/env Rscript
# Match samples using the CEM package.
# 
# Generate and save the list of matched samples between the two classes.
# 
# input_file <- tmp/dump_CEM_table.csv
# output_file <- tmp/dump_matched_samples.csv
#
# This script follows the example 3.1 of the CEM vignette.

main <- function(){
  # Load the CEM library
  library(cem)
  
  # Load the data
  input_file <- "tmp/metformin_CEM_table.csv"
  df <- read.csv(input_file, header = TRUE, row.names = 1)
  
  # Get number of positive and negative
  pos <- which(df$CLASS==1)
  neg <- which(df$CLASS==0)
  n_pos <- length(pos)
  n_neg <- length(neg)
  
  # Measure the dataset imbalance
  imbalance(group = df$CLASS, data = df, drop = 'CLASS')
  
  # Perform CEM
  mat <- cem(treatment = "CLASS", data = df, keep.all = TRUE,
             eval.imbalance = TRUE, k2k = TRUE, drop = 'CLASS')
  print(mat)
  
  # Create matched data frame
  df.matched <- df[mat$matched,]
  
  # Save matched data frame
  write.csv(df.matched, file = 'tmp/matched_CEM_table.csv')
}

# Run the main routine
print('-------------------------------------------------------------------')
print('MBS - PBS 10% dataset utility: matching_step2.R')
print('-------------------------------------------------------------------')

main()
