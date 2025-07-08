import json
import re
import sys
import os
import ollama
from ollama import RequestError,ResponseError
import logging
from rich.console import Console
from rich.markdown import Markdown
import openlit

openlit.init(collect_gpu_stats=True, capture_message_content=False)

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
console = Console(width=120)

CONST_N_CTX = 35000
CONST_MAX_CTX = 8200

# Paths

articles_file_path = "WikiRC_StepOne.json"
output_file_path = "llm3-summaries-using-zeroshot.json"
llmModel = os.getenv("MODEL_NAME")


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


# Function to clean and format text
def clean_text(text: str) -> str:
    text = re.sub(r"==\s*(References|External links)\s*==.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\[[0-9]+\]", "", text)  # Remove citation numbers
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

def stripThinkingPart(response_text: str) -> str:
    """Remove <thinking>...</Thinking> part"""
    
    parts = response_text.split("</think>")
    main_part = parts[1].strip() if len(parts) > 1 else response_text.strip()

    return main_part


# Function to generate summaries
def generate_summary(article):
    output = ""
    title = article.get("title", "Unknown Title")
    sections = article.get("content", {}).get("sections", [])

    if not sections:
        logging.warning(f"No sections found for article: {title}")
        return "No content available for summarization."

    main_text = ""
    recenttChange = ""
    for section in sections:
        main_text = f"\nArticle:\n{clean_text(section.get("text", ""))}\n\n"
        changes = section.get("changes", [])
        if changes:
            changesText = "\n".join(
                f"- {chg.get('change_summary', 'No summary')}" for chg in changes
            )
            recenttChange = f"\n[Recent Changes]:\n{changesText}\n\n"

    prompt = f"""
Remember to not explain your actions or make any reference to requests made to you, in your response.
Instruction: 
Write a summary of article given below with an attention-catching start.
Ensure to capture the following segemnts:
- main points
- themes
- key aspects
- historical context
- recent changes made in the article.

Article:
{main_text}

Recent Changes made in the article:
{recenttChange}
/think
    """

    try:
        with console.status("[bold green]Generating summary..."):

            genOpts = {
                "num_predict": CONST_MAX_CTX,
                "num_ctx": CONST_N_CTX,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.8,
                "min_p": 0.05,
            }
            output: ollama.ChatResponse = ollama.chat(
                # model="hf.co/unsloth/Jan-nano-128k-GGUF:BF16",
                model=llmModel,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options=genOpts,
            )

    except RequestError as chatFailed:
        logging.error(f"Request failed: {chatFailed}")
        return "NULL"
    except ResponseError as chatFailed:
        logging.error(f"Response error: {chatFailed}")
        return "NULL"
    except Exception as chatFailed:
        logging.error(f"Failed to load model: {chatFailed}")
        return "NULL"

    response = output.message.content
    response = stripThinkingPart(response)
    return response


# Process selected articles and store summaries
for article in articles:
    summary = generate_summary(article)
    article["llm3oneShotResponse"] = summary
    console.print(Markdown(f"### Summary for {article['title']}\n{summary}"))
    console.print("\n" + "=" * 90 + "\n")

# Save updated articles with summaries
try:
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump(articles, outfile, indent=4, ensure_ascii=False)
    logging.info(f"Summaries saved to {output_file_path}")
except Exception as e:
    logging.error(f"Failed to save summaries: {e}")
