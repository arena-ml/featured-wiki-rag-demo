#!/usr/bin/env python3
"""
RAG-based Article Summary Generator

This script generates summaries for articles using Retrieval-Augmented Generation (RAG)
with ChromaDB for vector storage and Ollama for LLM inference.
"""

import json
import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

import langchain_chroma
from rich.console import Console
from rich.markdown import Markdown
import ollama
import langchain_ollama
import openlit
import chromadb

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

CONST_SERVICE_NAME = "llm1-gen-summaries-via-RAG"
CONST_SUMMARY_KEY="llm1RagSummary"
CONST_max_tokens = 35000


@dataclass
class Config:
    """Configuration settings for the RAG system."""

    # Service configuration
    service_name: str = "llm1-gen-summaries-via-RAG"

    # Model configuration
    llm_model: str = os.getenv("MODEL_NAME")
    embedding_model: str = os.getenv("EMB_MODEL_NAME")


    # Generation parameters
    temperature: float = 0.6
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05

    # File paths
    vectorstore_path: str = "article_embeddings_db"
    input_path: str = "WikiRC_StepTwo.json"
    output_path: str = "llm1-summaries-using-embRAG.json"

    # Retrieval parameters
    retrieval_k: int = 50
    retrieval_fetch_k: int = 300

    # Logging
    # Handle log level conversion
    log_level_str = os.getenv("LOG_LEVEL", "DEBUG")
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str, logging.DEBUG)


class TelemetrySetup:
    """Handles OpenTelemetry logging and OpenLIT monitoring setup."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logging_provider = None
        self.logging_handler = None

    def setup(self) -> logging.Handler:
        """Initialize telemetry and return logging handler."""
        # Setup OpenTelemetry logging
        self.logging_provider = LoggerProvider(
            shutdown_on_exit=True,
            resource=Resource.create({"service.name": self.service_name})
        )
        set_logger_provider(self.logging_provider)

        # Setup OTLP exporter if endpoint is configured
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otel_endpoint:
            otlp_exporter = OTLPLogExporter(endpoint=otel_endpoint, insecure=True)
            self.logging_provider.add_log_record_processor(
                BatchLogRecordProcessor(otlp_exporter)
            )

        self.logging_handler = LoggingHandler(
            level=logging.NOTSET,
            logger_provider=self.logging_provider
        )

        return self.logging_handler


class RAGSummaryGenerator:
    """Main class for generating article summaries using RAG."""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console(width=90)
        self.embeddings = None
        self.vectorstore = None
        self.telemetry = TelemetrySetup(config.service_name)

        self._setup_logging()
        self._setup_monitoring()

    def _setup_logging(self) -> None:
        """Configure logging with telemetry."""
        logging.basicConfig(
            level=self.config.log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Add telemetry handler
        telemetry_handler = self.telemetry.setup()
        logging.getLogger().addHandler(telemetry_handler)
    @staticmethod
    def _setup_monitoring() -> None:
        """Initialize OpenLIT monitoring."""
        openlit.init(collect_gpu_stats=True, capture_message_content=False,application_name=CONST_SERVICE_NAME)

    def _initialize_embeddings(self) -> None:
        """Initialize the embeddings model."""
        try:
            self.embeddings = langchain_ollama.OllamaEmbeddings(
                model=self.config.embedding_model
            )
            logging.info(f"Initialized embeddings model: {self.config.embedding_model}")
        except Exception as e:
            logging.error(f"Failed to initialize embeddings: {e}")
            sys.exit(1)

    def _initialize_vectorstore(self) -> None:
        """Initialize the ChromaDB vectorstore."""
        try:
            chroma_client_settings = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=self.config.vectorstore_path,
                anonymized_telemetry=False,
            )

            self.vectorstore = langchain_chroma.Chroma(
                persist_directory=self.config.vectorstore_path,
                embedding_function=self.embeddings,
                collection_name="article_embeddings",
                client_settings=chroma_client_settings,
            )
            logging.info(f"Initialized vectorstore at: {self.config.vectorstore_path}")
        except Exception as e:
            logging.error(f"Failed to initialize vectorstore: {e}")
            sys.exit(1)

    def _load_articles(self) -> List[Dict[str, Any]]:
        """Load articles from the input file."""
        try:
            input_file = Path(self.config.input_path)
            if not input_file.exists():
                logging.error(f"Input file does not exist: {self.config.input_path}")
                sys.exit(1)

            with open(input_file, "r", encoding="utf-8") as file:
                articles = json.load(file)

            if not articles:
                logging.error(f"Input file does not contain valid JSON: {self.config.input_path}")
                sys.exit(1)

            logging.info(f"Loaded {len(articles)} articles from {self.config.input_path}")
            return articles

        except Exception as e:
            logging.error(f"Failed to load articles: {e}")
            sys.exit(1)

    def _save_results(self, articles: List[Dict[str, Any]]) -> None:
        """Save the processed articles to the output file."""
        try:
            output_file = Path(self.config.output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(articles, file, indent=4, ensure_ascii=False)

            logging.info(f"Results saved to {self.config.output_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            sys.exit(1)

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Remove underscores from title."""
        return title.replace("_", " ")

    def _getSummaryLength(self,prompt: str) -> int:
        token_info = ollama.embed(model=self.config.llm_model, input=prompt)
        total_tokens = token_info.prompt_eval_count

        if total_tokens is None:
            logging.error(f"token count none {token_info}.")
            sys.exit(1)

        thirty_percent = total_tokens * 0.3

        return min(thirty_percent, CONST_max_tokens)

    @staticmethod
    def _construct_prompt(self, context: str) -> str:
        """Construct the prompt for the language model."""
        return f"""
Follow these three instructions:
1. Go through the given article then provide a catchy summary, 
   consider historical context, significance, key aspects and recent changes made in the article.
2. Your responses should be strictly from the article provided nothing else.
3. Do not mention that it's a summary, and also do not mention anything about instructions given to you.

Article:
{context}
"""

    def _retrieve_documents(self, query: str, article_id: str) -> List[Any]:
        """Retrieve relevant documents from the vectorstore."""
        try:
            docs = self.vectorstore.max_marginal_relevance_search(
                query=query,
                filter={"articleID": article_id},
                k=self.config.retrieval_k,
                fetch_k=self.config.retrieval_fetch_k
            )

            logging.info(f"Retrieved {len(docs)} documents for article {article_id}")
            return docs

        except Exception as e:
            logging.error(f"Error during document retrieval: {e}")
            sys.exit(1)

    def _generate_response(self, context: str) -> str:
        """Generate response using the LLM."""
        try:
            prompt = self._construct_prompt(self=self,context=context)
            max_ouput_token = self._getSummaryLength(prompt=prompt)

            gen_options = {
                "num_predict": max_ouput_token,
                "num_ctx": CONST_max_tokens,
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "min_p": self.config.min_p,
            }

            with self.console.status("[bold green]Generating response..."):
                response = ollama.chat(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options=gen_options,
                )

            result = response.message.content
            logging.info("Successfully generated response")
            return result

        except Exception as e:
            logging.error(f"Error during LLM response generation: {e}")
            return "An error occurred while generating the model response."

    def _process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single article to generate its summary."""
        try:
            title = self._normalize_title(str(article["title"]))
            article_id = str(article["article_id"])

            # Create retrieval query
            retrieval_query = f"{title} {article_id} {title} {article_id}"

            self.console.print(Markdown(f"### Retrieval Query:\n {retrieval_query}"))

            # Retrieve documents
            docs = self._retrieve_documents(retrieval_query, article_id)

            if not docs:
                logging.warning(f"No documents retrieved for article {article_id}")
                article[CONST_SUMMARY_KEY] = "NULL"
                return article

            # Combine context and generate response
            context = "\n".join(doc.page_content for doc in docs)
            response = self._generate_response(context)

            # Update article with results
            article[CONST_SUMMARY_KEY] = response

            if response != "NULL":
                retrieved_docs = " ".join(str(doc) for doc in docs)
                article["retrivedDocs"] = retrieved_docs

            self.console.print(Markdown(f"### Response:\n{response}"))
            self.console.print("\n" + "=" * 50 + "\n")

            return article

        except Exception as e:
            logging.error(f"Error processing article {article.get('article_id', 'unknown')}: {e}")
            article["llm1embResponse"] = "An error occurred while processing this article."
            return article

    @contextmanager
    def _resource_manager(self):
        """Context manager for resource cleanup."""
        try:
            self._initialize_embeddings()
            self._initialize_vectorstore()
            yield
        finally:
            # Cleanup resources
            if hasattr(self, 'embeddings'):
                del self.embeddings
            logging.info("Resources cleaned up")

    def run(self) -> None:
        """Main execution method."""
        try:
            with self._resource_manager():
                # Load articles
                articles = self._load_articles()

                # Process each article
                updated_articles = []
                for i, article in enumerate(articles, 1):
                    logging.info(f"Processing article {i}/{len(articles)}")
                    processed_article = self._process_article(article)
                    updated_articles.append(processed_article)

                # Save results
                self._save_results(updated_articles)

                logging.info("Summary generation completed successfully")

        except Exception as e:
            logging.error(f"Fatal error in main execution: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    try:
        config = Config()
        generator = RAGSummaryGenerator(config)
        generator.run()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()