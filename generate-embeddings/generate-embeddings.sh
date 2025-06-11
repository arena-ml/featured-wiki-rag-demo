#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "# Running the generate-embeddings.py"
if ! python3 generate-embeddings.py; then
    log_error " generate-embeddings.py failed to execute. Please check the script and inputs."
fi

echo "moving output to next step"

# moving files to next stage folder
mv article_embeddings_db "${INPUT_DIR}"/wiki-edits-as-json/wiki-articles-with-edits-in-json/llm1-gen-summaries-via-RAG/

#temp way to indicate this data is from step 2 of pipeline and it's used by emb-RAG step
mv WikiRC_StepOne.json WikiRC_StepTwo.json

mv WikiRC_StepTwo.json  "${INPUT_DIR}"/wiki-edits-as-json/wiki-articles-with-edits-in-json/llm1-gen-summaries-via-RAG/

find "${INPUT_DIR}"/wiki-edits-as-json/wiki-articles-with-edits-in-json/ -mindepth 1 -maxdepth 1 ! -name 'generate-embeddings' -exec mv {} "${OUTPUT_DIR}"/embeddings-of-wiki-articles-with-edits/ \;
