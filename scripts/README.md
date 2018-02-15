# Scripts

These scripts are meant to extract relevant information from the raw MBS-PBS 10% dataset.
To run these scripts you are supposed to have to have organized the MBS-PBS 10% dataset in the folder `../../../data`.

## Usage and details
-----
`$ make_xy.py --from <YYYY>`

This script aims at creating a supervised dataset `D = (X, y)` where `y = 1` for individuals that *started* to take
gliceamia-controlling drugs from year `<YYYY>`.
