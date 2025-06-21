"""
LLM Summary Evaluation System

This module evaluates the quality of LLM-generated summaries using multiple metrics
and sends the results to an OpenTelemetry metrics collector.
"""

import json
import re
import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

import ollama
from rich.console import Console

# OpenTelemetry imports
import openlit

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


@dataclass
class Config:
    """Configuration constants for the evaluation system."""

    ARTICLES_FILE_PATH: str = "merged_articles.json"
    OUTPUT_FILE_PATH: str = "WikiRC_StepSix.json"
    N_CTX: int = 40000
    MAX_CTX: int = 900
    MODEL_NAME: str = "hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q8_0"
    TEMPERATURE: float = 0.4
    CONSOLE_WIDTH: int = 120


class FileValidator:
    """Utility class for file validation."""

    @staticmethod
    def validate_path(path: str) -> None:
        """Validate that a file path exists and is readable."""
        path_obj = Path(path)

        if not path_obj.exists():
            logging.error(f"Path not found: {path}")
            sys.exit(1)

        if not (path_obj.is_file() or path_obj.is_dir()):
            logging.error(f"Path is neither file nor directory: {path}")
            sys.exit(1)

        if not os.access(path, os.R_OK):
            logging.error(f"Path is not readable: {path}")
            sys.exit(1)

        path_type = "File" if path_obj.is_file() else "Directory"
        logging.info(f"{path_type} verified successfully: {path}")


class MetricsExporter:
    """Handles OpenTelemetry metrics export for evaluation data."""

    def __init__(
        self,
        service_name: str = "llm.evaluation.service",
        handle_missing_metrics: str = "set_zero",
    ):
        """
        Initialize the metrics exporter.

        Args:
            service_name: Name of the service
            handle_missing_metrics: How to handle missing metrics ("set_zero", "skip", "error")
        """
        self.handle_missing_metrics = handle_missing_metrics
        self.otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT_HTTP")

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Configure OpenTelemetry metrics."""
        if not self.otel_endpoint:
            logging.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT_HTTP not set, metrics disabled"
            )
            return

        exporter = OTLPMetricExporter(
            endpoint=f"{self.otel_endpoint}/v1/metrics",
            headers={"Content-Type": "application/x-protobuf"},
        )

        reader = PeriodicExportingMetricReader(
            exporter=exporter, export_interval_millis=3000
        )

        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        self.meter = metrics.get_meter("summaries.evaluation")

        # Create metric gauges
        self._create_gauges()

    def _create_gauges(self) -> None:
        """Create gauge metrics for each evaluation dimension."""
        gauge_configs = [
            ("coherence", "Coherence score for LLM evaluation"),
            ("consistency", "Consistency score for LLM evaluation"),
            ("fluency", "Fluency score for LLM evaluation"),
            ("relevance", "Relevance score for LLM evaluation"),
        ]

        self.gauges = {}
        for metric_name, description in gauge_configs:
            self.gauges[metric_name] = self.meter.create_gauge(
                name=f"llm_gen_summaries_{metric_name}",
                description=description,
                unit="score",
            )

    def send_metrics(
        self,
        evaluation_data: Dict[str, Dict[str, int]],
        additional_attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Send evaluation metrics to OpenTelemetry collector."""
        if not hasattr(self, "gauges") or not evaluation_data:
            logging.warning("No metrics to send or gauges not initialized")
            return

        base_attributes = {"evaluation_type": "llm_gen_summary_quality_assessment"}

        if additional_attributes:
            base_attributes.update(additional_attributes)

        metrics_sent = 0
        for config_name, metrics_data in evaluation_data.items():
            attributes = base_attributes.copy()
            attributes["method_type"] = config_name.lower()

            for metric_name, value in metrics_data.items():
                if metric_name.lower() in self.gauges:
                    self._send_single_metric(
                        self.gauges[metric_name.lower()], value, attributes, metric_name
                    )
                    metrics_sent += 1

        logging.debug(f"Sent {metrics_sent} evaluation metrics")

    def _send_single_metric(
        self, gauge, value, attributes: Dict, metric_name: str
    ) -> None:
        """Send a single metric value with error handling."""
        try:
            numeric_value = self._convert_to_numeric(self, value, metric_name)
            gauge.set(numeric_value, attributes)
        except Exception as e:
            logging.error(f"Error sending metric {metric_name}: {e}")

    @staticmethod
    def _convert_to_numeric(self, value: Any, metric_name: str) -> float:
        """Convert value to numeric format."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                logging.error(f"Cannot convert '{value}' to numeric for {metric_name}")
        elif value is None:
            logging.warning(f"{metric_name} is None, returning 0.0")
            return float(0.0)
        else:
            logging.error(f"Unexpected type {type(value)} for {metric_name}")
            return float(0.0)


class ResponseProcessor:
    """Handles processing of LLM responses."""

    @staticmethod
    def extract_json_response(response_text: str) -> str:
        """Extract and clean JSON from LLM response."""
        # Remove thinking tags
        parts = response_text.split("</think>")
        main_part = parts[1].strip() if len(parts) > 1 else response_text.strip()

        # Remove Markdown code blocks
        if main_part.startswith("```json"):
            main_part = main_part.strip("`")
            lines = main_part.split("\n")
            main_part = "\n".join(lines[1:-1])

        # Fix trailing commas
        main_part = re.sub(r",\s*([]}])", r"\1", main_part)

        # Truncate at last closing brace
        last_brace = max(main_part.rfind("}"), main_part.rfind("]"))
        if last_brace != -1:
            main_part = main_part[: last_brace + 1]

        return main_part


class SummaryEvaluator:
    """Main class for evaluating summary quality."""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console(width=config.CONSOLE_WIDTH)
        self.metrics_exporter = MetricsExporter()
        self.response_processor = ResponseProcessor()

    @staticmethod
    def _create_evaluation_prompt(self, article: Dict[str, Any]) -> str:
        """Create the evaluation prompt for the LLM."""
        sections = article.get("content", {}).get("sections", [])
        main_text = sections[0].get("text", "") if sections else ""

        summaries = {
            "llm1_rag": article.get("llm1_emb_response", ""),
            "llm1_zeroshot": article.get("llm1oneShotResponse", ""),
            "llm2_zeroshot": article.get("llm2oneShotResponse", ""),
        }

        return f"""
I have given you an article and three summaries, provide score out of ten for llm1-rag-Summary, 
llm1-ZeroShot-summary and llm2-ZeroShot-summary on these four metrics — Coherence, Consistency, Fluency, and Relevance.
Scores Should be in JSON format.
Your response should contain no comments, notes, or explanations.

[Article]:  
{main_text}

[llm1-rag-Summary]:  
{summaries['llm1_rag']}

[llm1-ZeroShot-Summary]:  
{summaries['llm1_zeroshot']}

[llm2-ZeroShot-Summary]:  
{summaries['llm2_zeroshot']}
"""

    def _validate_context_length(self, prompt: str) -> bool:
        """Check if prompt fits within a context window."""
        try:
            token_info = ollama.embed(model=self.config.MODEL_NAME, input=prompt)
            return token_info.prompt_eval_count <= self.config.N_CTX
        except Exception as e:
            logging.error(f"Error checking context length: {e}")
            return False

    def _generate_evaluation(self, prompt: str) -> str:
        """Generate evaluation using the LLM."""
        try:
            with self.console.status("[bold green]Generating evaluation..."):
                gen_options = {
                    "num_predict": self.config.MAX_CTX,
                    "num_ctx": self.config.N_CTX,
                    "temperature": self.config.TEMPERATURE,
                }

                response = ollama.chat(
                    model=self.config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    options=gen_options,
                )

                return response.message.content
        except Exception as e:
            logging.error(f"Error during LLM evaluation: {e}")
            return "Error occurred during evaluation"

    def evaluate_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate summaries for a single article."""
        article_id = article.get("article_id", "unknown")

        prompt = self._create_evaluation_prompt(self, article)

        if not self._validate_context_length(prompt):
            logging.warning(f"Article {article_id}: Input exceeds context limit")
            return {"error": "input above ctx limit"}

        raw_response = self._generate_evaluation(prompt)
        extracted_response = self.response_processor.extract_json_response(raw_response)

        # Parse JSON scores
        scores_dict = {}
        try:
            scores_dict = json.loads(extracted_response)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON for article {article_id}: {e}")
            self.console.print(f"Raw response: {extracted_response}")

        # Send metrics
        if scores_dict:
            self.metrics_exporter.send_metrics(
                evaluation_data=scores_dict,
                additional_attributes={"article_id": article_id},
            )

        return {
            "scores": extracted_response,
            "raw_response": raw_response,
            "parsed_scores": scores_dict,
        }


class ArticleProcessor:
    """Handles loading and saving article data."""

    @staticmethod
    def load_articles(file_path: str) -> List[Dict[str, Any]]:
        """Load articles from JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                articles = json.load(file)

            if not isinstance(articles, list) or not articles:
                raise ValueError("Invalid or empty articles data")

            logging.info(f"Loaded {len(articles)} articles from {file_path}")
            return articles

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from {file_path}: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading articles: {e}")
            sys.exit(1)

    @staticmethod
    def save_articles(articles: List[Dict[str, Any]], file_path: str) -> None:
        """Save articles to JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(articles, file, indent=4, ensure_ascii=False)
            logging.info(f"Results saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            sys.exit(1)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    """Main execution function."""
    setup_logging()
    config = Config()
    console = Console(width=config.CONSOLE_WIDTH)

    # Validate input files
    FileValidator.validate_path(config.ARTICLES_FILE_PATH)

    # Load articles
    articles = ArticleProcessor.load_articles(config.ARTICLES_FILE_PATH)
    console.print(len(articles))
    # Initialize evaluator
    evaluator = SummaryEvaluator(config)

    # Process each article
    for article in articles:
        article_title = article.get("title", "Unknown")
        console.print(f"\n[bold blue]Processing: {article_title}[/bold blue]")

        evaluation_result = evaluator.evaluate_article(article=article)

        # Store results in article
        article["smryReview"] = evaluation_result["scores"]
        article["smryReviewProcess"] = evaluation_result["raw_response"]

        # Display results
        console.print(f"### Review for summaries on {article_title}")
        console.print(evaluation_result["scores"])
        console.print("\n" + "=" * 90 + "\n")

    # Save results
    ArticleProcessor.save_articles(articles, config.OUTPUT_FILE_PATH)
    console.print(
        f"[bold green]Processing complete! Results saved to {config.OUTPUT_FILE_PATH}[/bold green]"
    )


if __name__ == "__main__":
    openlit.init(collect_gpu_stats=True, capture_message_content=False)
    main()
