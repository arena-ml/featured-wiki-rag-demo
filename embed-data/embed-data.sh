#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "# Running the embed-data.py"
if ! python3 embed-data.py; then
    log_error " embed-data.py failed to execute. Please check the script and inputs."
fi

echo "moving output to next step"

# moving files to next stage folder
mv vectorstore_index.faiss "${INPUT_DIR}"/json-data-to-embed/raw-data/llm1-rag/

#temp way to indicate this data is from step 2 of pipeline.
mv WikiRC_StepOne.json WikiRC_StepTwo.json

mv WikiRC_StepTwo.json  "${INPUT_DIR}"/json-data-to-embed/raw-data/llm1-rag/

find "${INPUT_DIR}"/json-data-to-embed/raw-data/ -mindepth 1 -maxdepth 1 ! -name 'embed-data' -exec mv {} "${OUTPUT_DIR}"/embedded-data/ \;
