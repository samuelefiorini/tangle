#!/bin/bash
# Run main scripts

clear; python scripts/labels_assignment.py -s -o tmp/dump -cs 50000 -nj 32 -t 2009
python scripts/labels_assignment.py -s -o tmp/dump -cs 50000 -nj 32 -t 2010
python scripts/labels_assignment.py -s -o tmp/dump -cs 50000 -nj 32 -t 2011
python scripts/labels_assignment.py -s -o tmp/dump -cs 50000 -nj 32 -t 2012
python scripts/labels_assignment.py -s -o tmp/dump -cs 50000 -nj 32 -t 2013
python scripts/labels_assignment.py -s -o tmp/dump -cs 50000 -nj 32 -t 2014
