import json
import os
import ollama
import sys
import logging
from rich.console import Console
from rich.markdown import Markdown

# OpenTelemetry Metrics
import openlit

openlit.init(collect_gpu_stats=True, capture_message_content=False)

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console(width=120)

# Paths
articles_file_path = "merged_articles.json"
output_file_path = "WikiRC_StepSix.json"

CONST_N_CTX = 14000
CONST_MAX_CTX = 900

# Get the OTEL Collector endpoint from env
otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT_HTTP")

# Set up OTEL metric exporter
exporter = OTLPMetricExporter(endpoint=otel_endpoint)

reader = PeriodicExportingMetricReader(exporter)
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter("summary_scores")

# Metric instrument (single histogram for all scores)
summary_score_histogram = meter.create_histogram(
    name="summary_score",
    unit="1",
    description="Score per summary per article"
)


def record_score(data, a_id):
    try:
        for summary, scores in data.items():
            for dimension, value in scores.items():
                histogram = meter.create_histogram(
                    name=f"{dimension}_score",
                    unit="1",
                    description=f"{dimension} score for summaries",
                )
                histogram.record(value, attributes={"summary": summary, "article_id": a_id})
    except Exception as e:
        console.print(e)
        sys.exit(1)


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
    # Strip the markdown formatting
    if main_part.startswith("```json"):
        scores = main_part.strip("`")  # remove backticks
        scores = scores.split("\n", 1)[1]  # remove the first line
        scores = scores.rsplit("\n", 1)[0]  # remove the last line
    return scores


# Function to generate summaries
def summaryReview(article):
    sections = article.get("content", {}).get("sections", [])
    main_text = sections[0].get("text", "")
    llm1embResponse = article.get("llm1embResponse", "")
    llm1oneShotReponse = article.get("llm1oneShotResponse", "")
    llm2oneShotReponse = article.get("llm2oneShotResponse", "")

    prompt = f"""
I have given you an article and three summaries, provide score out of ten for llm1-rag-Summary, 
llm1-ZeroShot-summary and llm2-ZeroShot-summary on these four metrics — Coherence, Consistency, Fluency, and Relevance.
Scores Should be in JSON format.
Your response should contain no comments, notes, or explanations.

[Article]:  
{main_text}

[llm1-rag-Summary]:  
{llm1embResponse}

[llm1-ZeroShot-Summary]:  
{llm1oneShotReponse}

[llm2-ZeroShot-Summary]:  
{llm2oneShotReponse}
"""
    try:
        with console.status("[bold green]Generating response..."):
            genOpts = {"num_predict": CONST_MAX_CTX, "num_ctx": CONST_N_CTX}

            inTokens = ollama.embed(model="hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q8_0",
                                    input=prompt)
            if inTokens.prompt_eval_count > CONST_N_CTX:
                return "input above ctx limit"

            genOpts = {"num_predict": CONST_MAX_CTX, "num_ctx": CONST_N_CTX, "temperature": 0.4}
            output: ollama.ChatResponse = ollama.chat(
                model='hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q8_0', messages=[
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
    article_id = article.get("article_id")

    review = summaryReview(article)

    scores = extract_response(review)
    article["smryReview"] = scores

    scores_in_json = json.loads(scores)
    record_score(scores_in_json, article_id)

    article["smryReviewProcess"] = review

    console.print(Markdown(f"### Review for summaries on  {article['title']}\n{scores}"))
    console.print("\n" + "=" * 90 + "\n")

# Save updated articles with summaries
try:
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump(articles, outfile, indent=4, ensure_ascii=False)
    logging.info(f"review  saved to {output_file_path}")
except Exception as e:
    logging.error(f"Failed to save review: {e}")
