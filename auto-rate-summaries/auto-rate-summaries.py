import json
import  re
import os
import time
import ollama
import sys
import logging
from rich.console import Console
from rich.markdown import Markdown
from typing import Dict, Any, Optional

# OpenTelemetry Metrics
import openlit

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

CONST_N_CTX = 40000
CONST_MAX_CTX = 900

class EvaluationMetricsSender:
    def __init__(self,
                 service_name: str = "llm-evaluation-service",
                 handle_missing_metrics: str = "set_zero",  # "set_zero", "skip", "error"
                 value_range: tuple = (0, 10)):
        """
        Initialize OpenTelemetry metrics sender for LLM evaluation data

        Args:
            service_name: Name of the service sending metrics
            handle_missing_metrics: How to handle missing metrics
                - "set_zero": Set missing metrics to 0 (default)
                - "skip": Skip missing metrics entirely
                - "error": Raise error for missing metrics
            value_range: Tuple of (min, max) values for clamping metrics
        """
        self.handle_missing_metrics = handle_missing_metrics
        self.value_range = value_range
        self.otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT_HTTP")
        # if not self.otel_endpoint:
        #     raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT_HTTP environment variable is not set")

        # # Configure OTLP exporter
        # exporter = OTLPMetricExporter(
        #     endpoint=f"{self.otel_endpoint}/v1/metrics",
        #     headers={
        #         "Content-Type": "application/x-protobuf"
        #     }
        # )
        #
        # # Set up metric reader with periodic export
        # reader = PeriodicExportingMetricReader(
        #     exporter=exporter,
        #     export_interval_millis=3000  # Export every 5 seconds
        # )
        #
        # # Create meter provider and meter
        # metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))

        oltp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT_HTTP")
        metrics.set_meter_provider(MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=oltp_endpoint),
                # export_interval_millis=2000 # every 2 seconds
            )]
        ))
        self.meter = metrics.get_meter("summaries_evaluation")

        # Create gauges for each metric type
        self.coherence_gauge = self.meter.create_gauge(
            name="llm_gen_summaries_coherence",
            description="Coherence score for LLM evaluation",
            unit="score"
        )

        self.consistency_gauge = self.meter.create_gauge(
            name="llm_gen_summaries_consistency",
            description="Consistency score for LLM evaluation",
            unit="score"
        )

        self.fluency_gauge = self.meter.create_gauge(
            name="llm_gen_summaries_fluency",
            description="Fluency score for LLM evaluation",
            unit="score"
        )

        self.relevance_gauge = self.meter.create_gauge(
            name="llm_gen_summaries_relevance",
            description="Relevance score for LLM evaluation",
            unit="score"
        )

    def send_evaluation_metrics(self,
                                evaluation_data: Dict[str, Dict[str, int]],
                                additional_attributes: Optional[Dict[str, str]] = None):
        """
        Send evaluation metrics to OpenTelemetry collector

        Args:
            evaluation_data: Dictionary containing evaluation results
            additional_attributes: Optional additional attributes to include
        """

        # Validate input data
        if not evaluation_data:
            print("Warning: Empty evaluation_data provided")
            return

        # Base attributes for all metrics
        base_attributes = {
            "evaluation_type": "llm_gen_summary_quality_assessment"
        }

        if additional_attributes:
            base_attributes.update(additional_attributes)

        # Send metrics for each model configuration
        for config_name, metrics_data in evaluation_data.items():
            # Add configuration-specific attributes
            attributes = base_attributes.copy()
            attributes["method_type"] = config_name

            # Send individual metric values with error handling
            metric_mappings = {
                "Coherence": self.coherence_gauge,
                "Consistency": self.consistency_gauge,
                "Fluency": self.fluency_gauge,
                "Relevance": self.relevance_gauge
            }
            i=0
            for metric_name, gauge in metric_mappings.items():
                if metric_name in metrics_data:
                    i += 1
                    try:
                        # Validate metric value
                        metric_value = metrics_data[metric_name]

                        # Handle different data types
                        if isinstance(metric_value, (int, float)):
                            # Clamp values to configured range
                            min_val, max_val = self.value_range
                            clamped_value = max(min_val, min(max_val, float(metric_value)))
                            gauge.set(clamped_value, attributes)

                        elif isinstance(metric_value, str):
                            # Try to convert string to float
                            try:
                                numeric_value = float(metric_value)
                                min_val, max_val = self.value_range
                                clamped_value = max(min_val, min(max_val, numeric_value))
                                gauge.set(clamped_value, attributes)
                                print(f"Info: Converted string '{metric_value}' to {clamped_value} for {metric_name}")
                            except ValueError:
                                print(f"Error: Cannot convert '{metric_value}' to numeric value for {metric_name}")
                                # Set to min value as fallback
                                gauge.set(self.value_range[0], attributes)

                        elif metric_value is None:
                            print(f"Warning: {metric_name} is None, setting to {self.value_range[0]}")
                            gauge.set(self.value_range[0], attributes)

                        else:
                            print(f"Error: Unexpected data type {type(metric_value)} for {metric_name}: {metric_value}")
                            gauge.set(self.value_range[0], attributes)

                    except Exception as metric_send_error:
                        print(f"Error setting {metric_name} gauge: {metric_send_error}")
                        # Set to min value as fallback to ensure metric is still sent
                        try:
                            gauge.set(self.value_range[0], attributes)
                        except Exception as fallback_error:
                            print(f"Critical error: Could not set fallback value for {metric_name}: {fallback_error}")

                else:
                    # Handle missing metrics based on configuration
                    if self.handle_missing_metrics == "set_zero":
                        print(
                            f"Info: {metric_name} not found in metrics data for {config_name}, setting to {self.value_range[0]}")
                        try:
                            gauge.set(self.value_range[0], attributes)
                        except Exception as metric_send_error:
                            print(f"Error setting default value for missing {metric_name}: {metric_send_error}")

                    elif self.handle_missing_metrics == "skip":
                        print(f"Info: {metric_name} not found in metrics data for {config_name}, skipping")
                        continue

                    elif self.handle_missing_metrics == "error":
                        raise KeyError(f"Required metric '{metric_name}' not found in data for {config_name}")

                    else:
                        print(
                            f"Warning: Unknown handle_missing_metrics option '{self.handle_missing_metrics}', defaulting to skip")
        if i==0:
            sys.exit("No metrics data provided")
            print(metrics_data)
        print(f"Sent evaluation metrics for {len(evaluation_data)} configurations")


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
    # Step 1: Extract the main part after </think>
    parts = response_text.split('</think>')
    main_part = parts[1].strip() if len(parts) > 1 else response_text.strip()

    # Step 2: Remove Markdown backticks if present
    if main_part.startswith("```json"):
        main_part = main_part.strip("`")
        lines = main_part.split("\n")
        main_part = "\n".join(lines[1:-1])  # remove the ```json and closing ```

    # Step 3: Remove trailing commas before } or ]
    main_part = re.sub(r',\s*([\]}])', r'\1', main_part)

    # Step 4: Truncate at last closing brace
    last_brace = max(main_part.rfind("}"), main_part.rfind("]"))
    if last_brace != -1:
        main_part = main_part[:last_brace + 1]

    return main_part


# Function to generate summaries
def summaryReview(article):
    sections = article.get("content", {}).get("sections", [])
    main_text = sections[0].get("text", "")
    llm1_emb_response = article.get("llm1_emb_response", "")
    llm1_zeroshot_reponse = article.get("llm1oneShotResponse", "")
    llm2_zeroshot_reponse = article.get("llm2oneShotResponse", "")

    prompt = f"""
I have given you an article and three summaries, provide score out of ten for llm1-rag-Summary, 
llm1-ZeroShot-summary and llm2-ZeroShot-summary on these four metrics — Coherence, Consistency, Fluency, and Relevance.
Scores Should be in JSON format.
Your response should contain no comments, notes, or explanations.

[Article]:  
{main_text}

[llm1-rag-Summary]:  
{llm1_emb_response}

[llm1-ZeroShot-Summary]:  
{llm1_zeroshot_reponse}

[llm2-ZeroShot-Summary]:  
{llm2_zeroshot_reponse}
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


summary_meterics_sender = EvaluationMetricsSender()
# Process selected articles and store summaries
for article in articles:
    article_id = article.get("article_id")

    review = summaryReview(article)

    scores = extract_response(review)
    article["smryReview"] = scores

    scores_in_json ={}

    try:
        scores_in_json = json.loads(scores)
    except Exception as e:
        console.print(scores)
        console.print(e)

    summary_meterics_sender.send_evaluation_metrics(
        evaluation_data=scores_in_json,
        additional_attributes={
            "article_id": article_id,
        }
    )
    article["smryReviewProcess"] = review

    console.print(f"### Review for summaries on  {article['title']}\n{scores}")
    console.print("\n" + "=" * 90 + "\n")

# Save updated articles with summaries
try:
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump(articles, outfile, indent=4, ensure_ascii=False)
    logging.info(f"review  saved to {output_file_path}")
except Exception as e:
    logging.error(f"Failed to save review: {e}")