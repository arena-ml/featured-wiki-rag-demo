#!/usr/bin/env python3
"""
Article Summarization Pipeline

This script processes Wikipedia articles and generates summaries using a language model.
It reads articles from a JSON file, generates summaries using the Ollama API,
and saves the results to an output file.
"""
import os
import sys

# adds the parent directory of the current script to Python’s module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ollama
from ollama import RequestError, ResponseError
import logging
from rich.console import Console
from rich.markdown import Markdown
from summarizer.zero_shot_summarizer import ArticleSummarizer
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
    OUTPUT_FILE: str = "llm4-summaries-using-zeroshot.json"

    # Environment
    MODEL_NAME = os.getenv("MODEL_NAME")
    # Summary Key
    SmryKey = "llm4Summary"


def main():
    """Main entry point for the application."""
    try:
        config = Config()
        summarizer = ArticleSummarizer(config=config)
        summarizer.run()
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()