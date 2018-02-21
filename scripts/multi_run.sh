#!/bin/bash

clear; python scripts/find_concessionals.py -s -o tmp/dump1 -cs 50000 -nj 32 -t 2009
python scripts/find_concessionals.py -s -o tmp/dump1 -cs 50000 -nj 32 -t 2010
python scripts/find_concessionals.py -s -o tmp/dump1 -cs 50000 -nj 32 -t 2011
python scripts/find_concessionals.py -s -o tmp/dump1 -cs 50000 -nj 32 -t 2012
python scripts/find_concessionals.py -s -o tmp/dump1 -cs 50000 -nj 32 -t 2013
