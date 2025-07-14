#!/usr/bin/env python3
"""
Article Summarization Pipeline

This script processes Wikipedia articles and generates summaries using a language model.
It reads articles from a JSON file, generates summaries using the Ollama API,
and saves the results to an output file.
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama
from ollama import RequestError, ResponseError
import logging
from rich.console import Console
from rich.markdown import Markdown
import openlit

# Initialize OpenLit for monitoring
openlit.init(collect_gpu_stats=True, capture_message_content=False)


@dataclass
class Config:
    """Configuration settings for the summarization pipeline."""

    # Model parameters
    max_tokens: int = 35000
    TEMPERATURE: float = 0.6
    TOP_K: int = 40
    TOP_P: float = 0.95
    MIN_P: float = 0.05

    # File paths
    ARTICLES_FILE: str = "WikiRC_StepOne.json"
    OUTPUT_FILE: str = "llm2-summaries-using-zeroshot.json"

    # Environment
    MODEL_NAME = os.getenv("MODEL_NAME")

class ArticleSummarizer:
    """Main class for handling article summarization pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console(width=120)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def _validate_files(self) -> None:
        """Validate that required files exist."""
        if not Path(self.config.ARTICLES_FILE).exists():
            raise FileNotFoundError(f"Input file not found: {self.config.ARTICLES_FILE}")

    def _load_articles(self) -> List[Dict]:
        """Load and validate articles from input file."""
        try:
            with open(self.config.ARTICLES_FILE, "r", encoding="utf-8") as file:
                articles = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}")
        except Exception as e:
            raise IOError(f"Failed to read input file: {e}")

        if not isinstance(articles, list) or not articles:
            raise ValueError("Invalid or empty articles data")

        self.logger.info(f"Successfully loaded {len(articles)} articles")
        return articles

    def _clean_text(self, text: str) -> str:
        """Clean and format text content."""
        # Remove references and external links sections
        text = re.sub(r"==\s*(References|External links)\s*==.*", "", text, flags=re.DOTALL)
        # Remove citation numbers
        text = re.sub(r"\[[0-9]+\]", "", text)
        # Normalize whitespace
        text = re.sub(r"\n{2,}", "\n", text).strip()
        return text


    def _extract_thinking_content(self, response_text: str) -> str:
        """Extract content after thinking tags."""
        parts = response_text.split("</think>")
        return parts[1].strip() if len(parts) > 1 else response_text.strip()

    def _build_prompt(self, main_text: str, recent_changes: str) -> str:
        """Build the prompt for summary generation."""
        return f"""
Remember to not explain your actions or make any reference to requests made to you, in your response.
Instruction: 
Return summary of article given below with an attention-catching start.
Ensure to capture the following segments:
- main points
- themes
- key aspects
- historical context
- recent changes made in the article.
NOTE: size of the summary should be roughly 30% of the Article.

Article:
{main_text}

Recent Changes made in the article:
{recent_changes}

/think
"""

    def _extract_article_content(self, article: Dict) -> Tuple[str, str]:
        """Extract main text and recent changes from article."""
        sections = article.get("content", {}).get("sections", [])

        if not sections:
            title = article.get("title", "Unknown Title")
            self.logger.warning(f"No sections found for article: {title}")
            return "", ""

        main_text = ""
        recent_changes = ""

        for section in sections:
            main_text = f"\nArticle:\n{self._clean_text(section.get('text', ''))}\n\n"
            changes = section.get("changes", [])
            if changes:
                change_list = "\n".join(
                    f"- {chg.get('change_summary', 'No summary')}" for chg in changes
                )
                recent_changes = f"\n[Recent Changes]:\n{change_list}\n\n"

        return main_text, recent_changes

    def _generate_summary_with_ollama(self, prompt: str,title :str) -> str:
        """Generate summary using Ollama API."""

        try:
            with self.console.status("[bold green]Generating summary..."):
                gen_options = {
                    "num_predict": self.config.max_tokens,
                    "num_ctx": self.config.max_tokens,
                    "temperature": self.config.TEMPERATURE,
                    "top_k": self.config.TOP_K,
                    "top_p": self.config.TOP_P,
                    "min_p": self.config.MIN_P,
                }

                response: ollama.ChatResponse = ollama.chat(
                    model=self.config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    options=gen_options,
                )

                return self._extract_thinking_content(response.message.content)

        except (RequestError, ResponseError) as e:
            self.logger.error(f"API error for article '{title}': {e}")
            return "NULL"
        except Exception as e:
            self.logger.error(f"Failed to generate summary for article '{title}': {e}")
            return "NULL"

    def _generate_summary(self, article: Dict) -> str:
        """Generate summary for a single article."""
        title = article.get("title", "Unknown Title")
        main_text, recent_changes = self._extract_article_content(article)

        if not main_text:
            return "NULL"

        prompt = self._build_prompt(main_text, recent_changes)
        return self._generate_summary_with_ollama(prompt,title)

    def _save_results(self, articles: List[Dict]) -> None:
        """Save processed articles with summaries to output file."""
        try:
            with open(self.config.OUTPUT_FILE, "w", encoding="utf-8") as outfile:
                json.dump(articles, outfile, indent=4, ensure_ascii=False)
            self.logger.info(f"Summaries saved to {self.config.OUTPUT_FILE}")
        except Exception as e:
            raise IOError(f"Failed to save summaries: {e}")

    def _display_summary(self, title: str, summary: str) -> None:
        """Display summary in the console."""
        self.console.print(Markdown(f"### Summary for {title}\n{summary}"))
        self.console.print("\n" + "=" * 90 + "\n")

    def process_articles(self) -> Tuple[int, int]:
        """Process all articles and generate summaries."""
        self._validate_files()
        articles = self._load_articles()

        successful_summaries = 0
        failed_summaries = 0

        for i, article in enumerate(articles, 1):
            title = article.get("title", f"Article {i}")
            self.logger.info(f"Processing article {i}/{len(articles)}: {title}")

            summary = self._generate_summary(article)
            article["llm2Summary"] = summary

            if summary == "NULL":
                failed_summaries += 1
                self.logger.warning(f"Failed to generate summary for: {title}")
            else:
                successful_summaries += 1
                self._display_summary(title, summary)

        self._save_results(articles)
        return successful_summaries, failed_summaries

    def run(self) -> None:
        """Run the complete summarization pipeline."""
        self.logger.info("Starting summary generation pipeline")

        try:
            successful, failed = self.process_articles()

            self.logger.info(f"Processing complete: {successful} successful, {failed} failed")

            if failed > 0:
                self.logger.warning(f"Pipeline completed with {failed} failures")
                sys.exit(1)
            else:
                self.logger.info("Pipeline completed successfully")
                sys.exit(0)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            sys.exit(1)


def main():
    """Main entry point for the application."""
    try:
        config = Config()
        summarizer = ArticleSummarizer(config)
        summarizer.run()
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()