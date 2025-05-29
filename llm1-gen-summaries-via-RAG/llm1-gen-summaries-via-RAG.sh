#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# echo "checking required directories exist"
# if [[ ! -d ${OUTPUT_DIR}/output3/ ]]; then
#     log_error "Directory ${OUTPUT_DIR}/output3/ does not exist."
# fi


echo "# Running the llm1-gen-summaries-via-RAG.py"
if ! python3 llm1-gen-summaries-via-RAG.py; then
    log_error "llm1-gen-summaries-via-RAG.py failed to execute. Please check the script and inputs."
fi

#mv WikiRC_StepThree.json "${INPUT_DIR}"/wiki-edits-embedded/embeddings-of-wiki-articles-with-edits/llm1-gen-summaries-zeroshot/

echo "moving output to next step"
mv llm1-summaries-using-embRAG.json  "${OUTPUT_DIR}"/llm1-summaries-using-embRAG/

find "${INPUT_DIR}"/wiki-edits-embedded/embeddings-of-wiki-articles-with-edits/auto-rate-summaries/ -mindepth 1 -maxdepth 1 ! -name 'llm1-gen-summaries-via-RAG' -exec mv {} "${OUTPUT_DIR}"/llm1-summaries-using-embRAG/ \;
