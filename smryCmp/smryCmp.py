# DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf
import json
import re
import os
import requests
import sys
import logging
from rich.console import Console
from rich.markdown import Markdown
from llama_cpp import Llama


# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console(width=120)

# Paths
articles_file_path = "WikiRC_ESO.json"
output_file_path = "smry_rating.json"

CONST_N_CTX = 30000



# Function to check file existence
def check_path(path):
    if not os.path.exists(path):
        print(f"Error: Path not found at {path}", file=sys.stderr)
        sys.exit(1)

    if not (os.path.isfile(path) or os.path.isdir(path)):
        print(f"Error: Path exists but is neither a file nor a directory: {path}", file=sys.stderr)
        sys.exit(1)

    if not os.access(path, os.R_OK):
        print(f"Error: Path is not readable: {path}", file=sys.stderr)
        sys.exit(1)

    path_type = "File" if os.path.isfile(path) else "Directory"
    print(f"{path_type} verified successfully: {path}")

# Verify required files
# check_path(llm_path)
check_path(articles_file_path)

def initialize_model(model_path):
    """
    Initialize the Llama model with proper error handling
    """
    try:
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_threads=6,
            n_ctx=CONST_N_CTX,
            verbose=True,
        )
        logging.info("Model initialized successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        raise

global model
model = initialize_model("/app/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf")

# Load articles
try:
    with open(articles_file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)
except json.JSONDecodeError:
    logging.error("Failed to decode JSON. Check file format.")
    sys.exit(1)

if not isinstance(articles, list) or not articles:
    logging.error("Invalid or empty articles data.")
    sys.exit(1)


def extract_response(response_text):
    parts = response_text.split('</think>')
    main_part = parts[1].strip() if len(parts) > 1 else response_text
    
    
    return main_part



# Function to generate summaries
def summaryReview(article):
    sections = article.get("content", {}).get("sections", [])        
    main_text = sections[0].get("text", "")
    emb_response = article.get("embResponse", "")
    oneShotReponse = article.get("oneShotSummary","")
    if "No relevant documents" in str(emb_response):
        return "no camparison, embResponse is empty"
    
    prompt = f"""
I have given you an article and two summaries, in JSON format provide score out of ten
for Summary-One and Summary-Two on these four metrics — Coherence, Consistency, Fluency, and Relevance.
Your response should contain no comments, notes, or explanations.

[Article]:  
{main_text}

Summary-One:  
{emb_response}

Summary-Two:  
{oneShotReponse}
"""
    try:
        with console.status("[bold green]Generating response..."):
            inTokens = model.tokenize(prompt.encode("utf-8"))
            if len(inTokens) > CONST_N_CTX:
                return "input above ctx limit"
            output = model.create_completion(
                prompt=prompt,
                max_tokens=5200,
            )
            
        logging.info(f"Raw model output: {output}")
        response = output["choices"][0]["text"].strip()
        
    except Exception as e:
        logging.error(f"Error during LLM response generation: {e}")
        return "An error occurred while generating the model response."

    return response





# Process selected articles and store summaries
for article in articles:
    embResponse = article.get("embResponse", "NULL")
    if  embResponse!= "NULL":
        review = summaryReview(article)
        jsonPart = extract_response(review)
        article["smryReview"] = jsonPart
        article["smryReviewProcess"] = review
        console.print(Markdown(f"### Review for summaries on  {article['title']}\n{jsonPart}"))
        console.print("\n" + "=" * 90 + "\n")

# Save updated articles with summaries
try:
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump(articles, outfile, indent=4, ensure_ascii=False)
    logging.info(f"review  saved to {output_file_path}")
except Exception as e:
    logging.error(f"Failed to save review: {e}")
