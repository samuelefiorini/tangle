# Scripts

These scripts are meant to extract relevant information from the raw MBS-PBS 10% dataset.
To run these scripts you are supposed to have to have organized the MBS-PBS 10% dataset in the folder `../../../data`.

## Usage and details
-----
Usage example:

`$ make_xy.py --root <ROOT> --target_year <YYYY> --output <FILENAME> --filter_copayments --monthly_breakdown`

or, equivalently

`$ make_xy.py -r <ROOT> -t <YYYY> -o <FILENAME> -fc -mb`

This script aims at creating a supervised dataset `D = (X, y)` where `y = 1` for individuals that *started* to take
gliceamia-controlling drugs in the year `<YYYY>`, while `y = 0` for individuals that were never prescribed with
diabetes controlling drugs in the years [2008, 2014].
