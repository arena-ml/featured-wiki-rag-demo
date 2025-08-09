#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# echo "checking required directories exist"
# if [[ ! -d ${OUTPUT_DIR}/output5/ ]]; then
#     log_error "Directory ${OUTPUT_DIR}/output5/ does not exist."
# fi


echo "# Running the llm4-gen-summaries-zeroshot.py"
if ! python3 llm4-gen-summaries-zeroshot.py; then
    log_error "llm4-gen-summaries-zeroshot.py failed to execute. Please check the script and inputs."
fi

echo "moving output to next step"


#mv WikiRC_StepFive.json "${INPUT_DIR}"/json-data-to-llm4/llm1-summaries-using-zero-shot/auto-rater/

echo "moving output to next step"
mv llm4-summaries-using-zeroshot.json "${OUTPUT_DIR}"/llm4-summaries-using-zeroshot/

#find "${INPUT_DIR}"/json-data-to-llm4/llm1-summaries-using-zero-shot/ -mindepth 1 -maxdepth 1 ! -name 'llm4-gen-summaries-zeroshot' -exec mv {} "${OUTPUT_DIR}"/llm4-summaries-using-zeroshot/ \;
