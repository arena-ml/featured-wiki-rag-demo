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


echo "# Running the llm1-gen-summaries-zeroshot.py"
if ! python3 llm1-gen-summaries-zeroshot.py; then
    log_error "llm1-gen-summaries-zeroshott failed to execute. Please check the script and inputs."
fi

#mv llm1-gen-summaries-zeroshot.json "${INPUT_DIR}"/json-data-to-llm1/llm1-summaries-using-embRAG/llm2-gen-summaries-zeroshot/

echo "moving output to next step"
mv llm1-gen-summaries-zeroshot.json  "${OUTPUT_DIR}"/llm1-summaries-using-zeroshot/

#find "${INPUT_DIR}"/json-data-to-llm1/wiki-edits-as-json/ -mindepth 1 -maxdepth 1 ! -name 'llm1-gen-summaries-zeroshot' -exec mv {} "${OUTPUT_DIR}"/llm1-summaries-using-zeroshot/ \;
