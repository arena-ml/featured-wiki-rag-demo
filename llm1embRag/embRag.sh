#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "checking required directories exist"
if [[ ! -d ${OUTPUT_DIR}/output3/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output3/ does not exist."
fi


echo "# Running the embRag.py"
if ! python3 embRag.py; then
    log_error "embRag.py failed to execute. Please check the script and inputs."
fi

mv WikiRC_ES.json ${INPUT_DIR}/input3/output2/llm1oneShotRag/

echo "moving output to next step"

find ${INPUT_DIR}/input3/output2/ -mindepth 1 -maxdepth 1 ! -name 'llm1embRag' -exec mv {} ${OUTPUT_DIR}/output3/ \;
