#!/bin/bash
# Run main scripts

# --- Assign labels --- #
# clear; python scripts/labels_assignment.py -r ../../data -o tmp/dump -cs 50000 -nj 32 -t 2009
# python scripts/labels_assignment.py -r ../../data -o tmp/dump -cs 50000 -nj 32 -t 2010
# python scripts/labels_assignment.py -r ../../data -o tmp/dump -cs 50000 -nj 32 -t 2011
# python scripts/labels_assignment.py -r ../../data -o tmp/dump -cs 50000 -nj 32 -t 2012
# python scripts/labels_assignment.py -r ../../data -o tmp/dump -cs 50000 -nj 32 -t 2013
# python scripts/labels_assignment.py -r ../../data -o tmp/dump -cs 50000 -nj 32 -t 2014

# --- Extract sequences --- #
clear; python scripts/extract_sequences.py -sic -r ../../data -s tmp/dump_class_0.csv # Negative examples
