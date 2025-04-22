# DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf
import json
import os
import ollama
import sys
import logging
from rich.console import Console
from rich.markdown import Markdown
# OpenTelemetry Metrics Only
import openlit

openlit.init(collect_gpu_stats=True)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console(width=120)

# Paths
articles_file_path = "WikiRC_StepFive.json"
output_file_path = "WikiRC_StepSix.json"

CONST_N_CTX = 40000
CONST_MAX_CTX=7000



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


def extract_response(response_text):
    parts = response_text.split('</think>')
    main_part = parts[1].strip() if len(parts) > 1 else response_text
    
    
    return main_part



# Function to generate summaries
def summaryReview(article):
    sections = article.get("content", {}).get("sections", [])        
    main_text = sections[0].get("text", "")
    llm1embResponse = article.get("llm1embResponse", "")
    llm1oneShotReponse = article.get("llm1oneShotResponse","")
    llm2oneShotReponse = article.get("llm2oneShotResponse","")
    
    prompt = f"""
I have given you an article and three summaries, in JSON format provide score out of ten
for Summary-One, Summary-Two and Summary-Three on these four metrics — Coherence, Consistency, Fluency, and Relevance.
Your response should contain no comments, notes, or explanations.

[Article]:  
{main_text}

Summary-One:  
{llm1embResponse}

Summary-Two:  
{llm1oneShotReponse}

Summary-Three
{llm2oneShotReponse}
"""
    try:
        with console.status("[bold green]Generating response..."):
            genOpts = {"num_predict":CONST_MAX_CTX,"num_ctx":CONST_N_CTX}
        
            inTokens = ollama.embed(model="hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q8_0",input=prompt)
            if inTokens.prompt_eval_count > CONST_N_CTX:
                return "input above ctx limit"
            
            genOpts = {"num_predict":CONST_MAX_CTX,"num_ctx":CONST_N_CTX,"temperature":0.4}
            output : ollama.ChatResponse = ollama.chat(model='hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q8_0',  messages=[
              {
                'role': 'user',
                'content': prompt,
              },
            ],
            options=genOpts)
            
        logging.info(f"Raw model output: {output.message}")
        response = output.message.content
        
    except Exception as e:
        logging.error(f"Error during LLM response generation: {e}")
        return "An error occurred while generating the model response."

    return response





# Process selected articles and store summaries
for article in articles:
    embResponse = article.get("llm1embResponse", "NULL")
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
