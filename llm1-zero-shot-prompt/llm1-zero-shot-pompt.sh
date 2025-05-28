#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# echo "checking required directories exist"
# if [[ ! -d ${OUTPUT_DIR}/output4/ ]]; then
#     log_error "Directory ${OUTPUT_DIR}/output4/ does not exist."
# fi


echo "# Running the llm1-zero-shot-pompt.py"
if ! python3 llm1-zero-shot-pompt.py; then
    log_error "llm1-zero-shot-pompt failed to execute. Please check the script and inputs."
fi

mv WikiRC_StepFour.json "${INPUT_DIR}"/json-data-to-llm1/llm1-embRAG-summaries/llm2-zero-shot-pompt/

echo "moving output to next step"

find "${INPUT_DIR}"/json-data-to-llm1/ollm1-embRAG-summaries/ -mindepth 1 -maxdepth 1 ! -name 'llm1-zero-shot-pompt' -exec mv {} "${OUTPUT_DIR}"/llm1-zero-shot-summaries/ \;
