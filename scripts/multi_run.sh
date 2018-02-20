#!/bin/bash

# clear; python scripts/make_xy.py -o tmp/1_df_yearly_w_copayment.pkl
# clear; python scripts/make_xy.py -o tmp/2_df_monthly_w_copayment.pkl -mb
# clear; python scripts/make_xy.py -o tmp/3_df_yearly_no_copayment.pkl -fc
# clear; python scripts/make_xy.py -o tmp/4_df_monthly_no_copayment.pkl -mb -fc

clear; python scripts/find_concessionals.py -s -o tmp/dump1
