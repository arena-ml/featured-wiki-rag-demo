#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "checking required directories exist"
if [[ ! -d ${INPUT_DIR}/input2/output1/embRag/ ]]; then
    log_error "Directory ${INPUT_DIR}/input2/output1/embRag/ does not exist."
fi

if [[ ! -d ${OUTPUT_DIR}/output2/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output2/ does not exist."
fi

echo "# Running the dataEmbed.py"
if ! python3 dataEmbed.py; then
    log_error " dataEmbed.py failed to execute. Please check the script and inputs."
fi

echo "moving output to next step"

# moving files to next stage folder
mv vectorstore_index.faiss ${INPUT_DIR}/input2/output1/embRag/

mv WikiRC.json  ${INPUT_DIR}/input2/output1/embRag/

find ${INPUT_DIR}/input2/output1/ -mindepth 1 -maxdepth 1 ! -name 'dataEmbed' -exec mv {} ${OUTPUT_DIR}/output2/ \;
