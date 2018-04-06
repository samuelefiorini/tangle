# Scripts

These scripts are meant to extract relevant information from the raw MBS-PBS 10% dataset.
To run these scripts you are supposed to have to have organized the MBS-PBS 10% dataset in a folder (*e.g.:* `../../../data`).

Scripts developed for Python 2.7.

## Usage and details
### - `labels_assignment.py`

Usage example:

`$ python labels_assignment.py --root <ROOT> --skip_input_check --metformin --output <PATH> --target_year <YYYY> --n_jobs <NNN>`

or, equivalently:

`$ python labels_assignment.py -r <ROOT> -sic -m -o <PATH> -t <YYYY> -nj <NNN>`

This script aims at finding the patient identifiers of the positive and negative
classes, where:
+ positive class (`y = 1`): subjects that continuously and consistently use
  their concessional card in the observation years to buy diabetes drugs,
- negative class (`y = 0`): subjects that continuously and consistently use
  their concessional card but were never prescribed to diabetes control drugs.

 This script also extracts two other labels:

 `MET_ONLY` - *i.e.*: patients that are using metformin ONLY

 `MET_AFTER` - *i.e.*: patients that after a first metformin prescription started to use another diabetes controlling drug.

### - `extract_sequences.py`

Usage example:

`$ python extract_sequences.py -root <ROOT> --skip_input_check --exclude_pregnancy --source <PATH> -n_jobs <NNN>`

or, equivalently:

`$ python extract_sequences.py -r <ROOT> -sic -ep -s <PATH> -nj <NNN>`

This script extracts the raw sequences from the MBS files. An example of
sequence is `1256 0 56489 12 ...` where odd entries are MBS items
and even entries are days between each visit.

### - `matching_step1.py`

Usage example:

`$ python scripts/matching_step1.py --source <PATH-TO-SOURCE> --output <PATH-TO-OUTPUT>`

or, equivalently:

`$ python scripts/matching_step1.py -s <PATH-TO-SOURCE> -o <PATH-TO-OUTPUT>`

This script prepares the data for the actual matching procedure (see `matching_step2.R`).

### - `matching_step2.R`

Usage example:

`$ Rscript scripts/matching_step2.R`

Run the matching algorithm by CEM package and generate a `matched_CEM_table.csv` file.

### - `single_train.py`

Usage example:

`$ python scripts/single_train.py --labels <PATH-TO matched_CEM_table.csv> --data <PATH-TO raw_data_.pkl> --embedding <PATH-TO embedding.100d.csv> --output <PATH-TO-OUTPUT>`

or, equivalently:

`$ python scripts/single_train.py -l <PATH-TO matched_CEM_table.csv> -d <PATH-TO raw_data_.pkl> -e <PATH-TO embedding.100d.csv> -o <PATH-TO-OUTPUT>`

Fit the bidirectional timestamp-guided model on a random training/validation/test split of the matched dataset.

### - `cross_validate.py`

Usage example:

`$ python scripts/cross_validate.py --n_splits N --labels <PATH-TO matched_CEM_table.csv> --data <PATH-TO raw_data_.pkl> --embedding <PATH-TO embedding.100d.csv> --output <PATH-TO-OUTPUT>`

or, equivalently:

`$ python scripts/cross_validate.py -n N -l <PATH-TO matched_CEM_table.csv> -d <PATH-TO raw_data_.pkl> -e <PATH-TO embedding.100d.csv> -o <PATH-TO-OUTPUT>`

Evaluate the average predictive performance of the model on N random training/validation/test splits.
