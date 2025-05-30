#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# Run the data preparation script
echo "running fetchWikiData.py script to fetch data"
if ! python3 fetchWikiData.py; then
    log_error "fetchWikiData.py failed to execute. Please check the script and inputs."
fi

cp WikiRC_StepOne.json "${INPUT_DIR}"/src-notebooks/phi35ragRepo/generate-embeddings/
cp WikiRC_StepOne.json "${INPUT_DIR}"/src-notebooks/phi35ragRepo/llm1-gen-summaries-zeroshot/
cp WikiRC_StepOne.json "${INPUT_DIR}"/src-notebooks/phi35ragRepo/llm2-gen-summaries-zeroshot/

#move output to sharedDir
echo "moving output to next step"

# experimental cmd
find "${INPUT_DIR}"/src-notebooks/phi35ragRepo/ -mindepth 1 -maxdepth 1 ! -name 'fetch-wiki-data' -exec mv {} "${OUTPUT_DIR}"/wiki-articles-with-edits-in-json/ \;

# Final message
echo "Script completed successfully"