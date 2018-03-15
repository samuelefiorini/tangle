#!/bin/bash
# Run main scripts

# --- Get population of interest --- #
# Find continuously and consistently concessionals on diabetes
clear; python scripts/get_population_of_interest.py -r ../../data -o tmp/auxfile -m -nj 32

# --- Extract raw data --- #
# clear; python scripts/extract_sequences.py -sic -r ../../data -ep -s tmp/dump_2009_class_1.csv -nj 8
# clear; python scripts/extract_sequences.py -sic -r ../../data -ep -s tmp/dump_class_0.csv -nj 8

# --- Prepare data for matching with CEM --- #
# clear; python scripts/matching_step1.py -s tmp -o tmp/metformin

# --- Match with CEM --- #
# clear; Rscript scripts/matching_step2.R
