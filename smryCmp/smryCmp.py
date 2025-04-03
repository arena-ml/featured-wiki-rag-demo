# DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf
import json
import re
import os
import requests
import sys
import logging
from rich.console import Console
from rich.markdown import Markdown

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console(width=120)

# Paths
articles_file_path = "WikiRC_ESO.json"
output_file_path = "smry_rating.json"



ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
AUTH_TOKEN = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
print(AUTH_TOKEN)

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


# Function to generate summaries
def summaryReview(article):
    sections = article.get("content", {}).get("sections", [])        
    main_text = sections[0].get("text", "")
    emb_response = article.get("embResponse", "")
    oneShotReponse = article.get("oneShotSummary","")
    
    prompt = f"""
I have written two summaries on article below, which summary is better
response in a table format based on different metrics used in comparsion.
[Article]:
{main_text}

SummaryOne:
{emb_response}

SummaryTwo:
{oneShotReponse}
"""
    
    # response = requests.post(
    #     f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    #     headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    #     json={
    #         "messages": [
    #             {"role": "system", "content": "You are a friendly assistant"},
    #             {"role": "user", "content": prompt}
    #         ],
    #         "max_tokens": 3000  
    #     }
    # )
    # jsonResult = response.json()
    # print(jsonResult)
    # response = jsonResult.get("response","")
    # return response
    return ""



# Process selected articles and store summaries
for article in articles:
    review = summaryReview(article)
    article["smryReview"] = review
    console.print(Markdown(f"### Review for summaries on  {article['title']}\n{review}"))
    console.print("\n" + "=" * 90 + "\n")

# Save updated articles with summaries
try:
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump(articles, outfile, indent=4, ensure_ascii=False)
    logging.info(f"review  saved to {output_file_path}")
except Exception as e:
    logging.error(f"Failed to save review: {e}")
