#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# Run the data preparation script
echo "running fetch-data.py script to fetch data"
if ! python3 fetch-data.py; then
    log_error "fetch-data.py failed to execute. Please check the script and inputs."
fi

mv WikiRC_StepOne.json "${INPUT_DIR}"/input1/phi35ragRepo/embed-data/

#move output to sharedDir
echo "moving output to next step"

# experimental cmd
find "${INPUT_DIR}"/input1/phi35ragRepo/ -mindepth 1 -maxdepth 1 ! -name 'fetch-data' -exec mv {} "${OUTPUT_DIR}"/output1/ \;

# Final message
echo "Script completed successfully"