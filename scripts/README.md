# Scripts

These scripts are meant to extract relevant information from the raw MBS-PBS 10% dataset.
To run these scripts you are supposed to have to have organized the MBS-PBS 10% dataset in a folder (*e.g.:* `../../../data`).

Python 2.7 needed.

## Usage and details
### - `labels_assignment.py`

Usage example:

`$ labels_assignment.py --root <ROOT> --skip_input_check --output <PATH> --target_year <YYYY> --chunk_size <CCC> --n_jobs <NNN>`

or, equivalently:

`$ labels_assignment.py -r <ROOT> -s -o <PATH> -t <YYYY> -cs <CCC> -nj <NNN>`

This script aims at finding the patient identifiers of the positive and negative
classes, where:
+ positive class (`y = 1`): subjects that continuously and consistently use
  their concessional card in the observation years to buy diabetes drugs,
- negative class (`y = 0`): subjects that continuously and consistently use
  their concessional card but were never prescribed to diabetes control drugs.

### - `extract_sequences.py`

Usage example:

`$ extract_sequences.py -root <ROOT> -skip_input_check --source <PATH> -n_jobs <NNN>`

or, equivalently:

`$ extract_sequences.py -r <ROOT> -sic -s <PATH> -nj <NNN>`

This script extracts the raw BTOS sequences from the MBS files. An example of
BTOS sequence is [101, 5, 102, 10, 103, ...] where odd entries are BTOS codes
and even entries are days between each visit.
