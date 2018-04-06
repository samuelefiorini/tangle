#!/bin/bash
# Run main scripts

# --- Get population of interest --- #
# Find continuously and consistently concessionals on diabetes
# clear; python scripts/get_population_of_interest.py -r ../../data -o tmp/auxfile

# --- Assign labels --- #
# clear; python scripts/assign_labels.py -r ../../data -s tmp -o tmp/labels.csv

# --- Extract raw data --- #
# clear; python scripts/extract_sequences.py -sic -r ../../data -ep -s tmp/labels.csv -o tmp/item_days

# --- Prepare data for matching with CEM --- #
# clear; python scripts/matching_step1.py -s tmp -o tmp/metformin

# --- Match with CEM --- #
# Rscript scripts/matching_step2.R

# --- Train model --- #
# clear; python scripts/single_train.py -l tmp/1_METONLY_vs_METX/matched_CEM_table.csv -d tmp/item_days_raw_data_.pkl -e tmp/embedding.100d.csv -o tmp/output

# --- Cross validate model -- #
clear; python scripts/cross_validate.py -n 5 -l tmp/1_METONLY_vs_METX/matched_CEM_table.csv -d tmp/item_days_raw_data_.pkl -e tmp/embedding.50d.csv -o tmp/output
