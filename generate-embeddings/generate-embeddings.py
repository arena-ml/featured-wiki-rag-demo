#!/usr/bin/env python3
"""
Embedding Generator for WikiRC Articles

This module processes WikiRC articles from JSON format and generates embeddings
using Ollama, storing them in a ChromaDB vector database.
"""

import sys
import os
import time
import json
import logging
import signal
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager

import langchain
import langchain.schema
import langchain.text_splitter
from tqdm import tqdm
import langchain_ollama
from langchain_chroma import Chroma
import chromadb
import openlit
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

CONST_SERVICE_NAME = "generate.embeddings"


@dataclass
class Config:
    """Configuration settings for the embedding generator."""

    json_file_path: str = "WikiRC_StepOne.json"
    vector_store_path: str = "article_embeddings_db"
    collection_name: str = "article_embeddings"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 2500
    chunk_overlap: int = 50
    log_level: int = logging.INFO
    log_file: str = "indexing.log"

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            json_file_path=os.getenv("JSON_FILE_PATH", cls.json_file_path),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", cls.vector_store_path),
            collection_name=os.getenv("COLLECTION_NAME", cls.collection_name),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            log_level=getattr(logging, os.getenv("LOG_LEVEL",cls.log_level)),
            log_file=os.getenv("LOG_FILE", cls.log_file),
        )


class GracefulShutdown:
    """Handles graceful shutdown on SIGINT and SIGTERM."""

    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        logging.warning("Shutdown signal received. Finishing current batch before exiting...")
        self.shutdown_requested = True

    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_requested


class TelemetrySetup:
    """Sets up OpenTelemetry logging and OpenLIT monitoring."""

    def __init__(self):
        self.logger_provider = None
        self.handler = None

    def setup(self) -> logging.Handler:
        """Initialize telemetry and return logging handler."""
        # Initialize OpenLIT
        openlit.init(collect_gpu_stats=True, capture_message_content=False,application_name=CONST_SERVICE_NAME)

        # Setup OpenTelemetry logging
        self.logger_provider = LoggerProvider(
            shutdown_on_exit=True,
            resource=Resource.create({"service.name": CONST_SERVICE_NAME})
        )
        set_logger_provider(self.logger_provider)

        # Setup OTLP exporter if endpoint is configured
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otel_endpoint:
            otlp_exporter = OTLPLogExporter(endpoint=otel_endpoint, insecure=True)
            self.logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(otlp_exporter)
            )

        self.handler = LoggingHandler(level=logging.NOTSET, logger_provider=self.logger_provider)
        return self.handler


class DocumentProcessor:
    """Processes WikiRC JSON documents and splits them into chunks."""

    def __init__(self, config: Config, shutdown_handler: GracefulShutdown):
        self.config = config
        self.shutdown_handler = shutdown_handler
        self.text_splitter = langchain.text_splitter.TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def parse_json(self, file_path: str) -> Generator[langchain.schema.Document, None, None]:
        """Parse JSON file and yield documents."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                logging.debug(f"Successfully loaded JSON with {len(data)} articles")

                for article_idx, article in enumerate(data):
                    if self.shutdown_handler.should_shutdown:
                        logging.error("Shutdown requested during JSON parsing. Exiting.")
                        break

                    yield from self._process_article(article_idx, article)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error parsing JSON: {str(e)}")
            raise

    def _process_article(self, article_idx: int, article: Dict[str, Any]) -> Generator[
        langchain.schema.Document, None, None]:
        """Process a single article and yield document chunks."""
        article_id = article.get("article_id", "")
        title = str(article.get("title", ""))
        logging.debug(f"Parsing article {article_idx} - {title}")

        for section in article.get("content", {}).get("sections", []):
            content = self._build_content(article_id, title, section)
            document = langchain.schema.Document(
                page_content=content,
                metadata={
                    "source": "https://api.wikimedia.org",
                    "articleID": article_id,
                    "articleTitle": title,
                }
            )
            yield from self._split_document(document)

    def _build_content(self, article_id: str, title: str, section: Dict[str, Any]) -> str:
        """Build content string from article section."""
        text = section.get("text", "")
        changes_content = self._format_changes(section.get("changes", []))

        return (
            f"[Article Title: {title}]\n"
            f"[ID: {article_id}]\n"
            f"Full Text: {text}\n"
            f"Changes:\n{changes_content}\n"
        )

    def _format_changes(self, changes: List[Dict[str, Any]]) -> str:
        """Format changes into a readable string."""
        if not changes:
            return "No changes in this section."

        return "\n".join([
            f"Change Summary: {change['change_summary']}\nDiff: {change['diff']}\n"
            for change in changes
        ])

    def _split_document(self, document: langchain.schema.Document) -> Generator[langchain.schema.Document, None, None]:
        """Split document into chunks."""
        try:
            splits = self.text_splitter.split_text(document.page_content)
            for chunk in splits:
                yield langchain.schema.Document(
                    page_content=chunk,
                    metadata=document.metadata
                )
        except Exception as e:
            logging.error(f"Failed to split document: {str(e)}")
            raise


class VectorStoreManager:
    """Manages the ChromaDB vector store operations."""

    def __init__(self, config: Config):
        self.config = config
        self.vectorstore: Optional[Chroma] = None
        self.chroma_settings = chromadb.config.Settings(
            is_persistent=True,
            persist_directory=config.vector_store_path,
            anonymized_telemetry=False,
        )

    @contextmanager
    def get_vectorstore(self):
        """Context manager for vector store operations."""
        try:
            embeddings = langchain_ollama.OllamaEmbeddings(model=self.config.embedding_model)

            self.vectorstore = Chroma(
                collection_name=self.config.collection_name,
                client_settings=self.chroma_settings,
                embedding_function=embeddings,
                persist_directory=self.config.vector_store_path,
            )
            yield self.vectorstore
        except Exception as e:
            logging.error(f"Error initializing vector store: {str(e)}")
            raise
        finally:
            if self.vectorstore:
                # ChromaDB persists automatically
                logging.info("Vector store operations completed")

    def add_documents(self, documents: List[langchain.schema.Document]) -> None:
        """Add documents to the vector store."""
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized")

        try:
            self.vectorstore.add_documents(documents)
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {str(e)}")
            raise


class EmbeddingGenerator:
    """Main class for generating embeddings from WikiRC articles."""

    def __init__(self, config: Config):
        self.config = config
        self.shutdown_handler = GracefulShutdown()
        self.telemetry = TelemetrySetup()
        self.document_processor = DocumentProcessor(config, self.shutdown_handler)
        self.vector_store_manager = VectorStoreManager(config)

    def setup_logging(self) -> None:
        """Configure logging with telemetry."""
        logging.basicConfig(
            level=self.config.log_level,
            format="%(levelname)s | %(asctime)s | %(message)s",
            handlers=[
                # logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ],
        )

        # Add telemetry handler
        telemetry_handler = self.telemetry.setup()
        logging.getLogger().addHandler(telemetry_handler)

    def process_and_index(self) -> None:
        """Process documents and create embeddings."""
        if not Path(self.config.json_file_path).exists():
            raise FileNotFoundError(f"JSON file not found: {self.config.json_file_path}")

        logging.debug(f"Starting embedding generation for {self.config.json_file_path}")

        with self.vector_store_manager.get_vectorstore() as vectorstore:
            documents = list(self.document_processor.parse_json(self.config.json_file_path))
            total_documents = len(documents)
            logging.info(f"Total documents to process: {total_documents}")

            with tqdm(total=total_documents, desc="Processing documents") as pbar:
                for idx, doc in enumerate(documents):
                    if self.shutdown_handler.should_shutdown:
                        logging.info("Shutdown requested. Exiting.")
                        break

                    try:
                        self.vector_store_manager.add_documents([doc])
                        logging.debug(f"Processed document {idx + 1}/{total_documents}")
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error processing document {idx + 1}: {str(e)}")
                        raise

        logging.info("Embedding generation completed successfully")

    def run(self) -> None:
        """Main execution method."""
        self.setup_logging()
        openlit.logger.info("Generating embeddings start time: %.2f", time.time())

        try:
            logging.info("Starting embedding generation process")
            self.process_and_index()
        except Exception as e:
            logging.error(f"Error occurred during embedding generation: {str(e)}")
            sys.exit(1)
        finally:
            logging.info("Finished embedding generation")


def main():
    """Entry point for the embedding generator."""
    try:
        config = Config.from_env()
        generator = EmbeddingGenerator(config)
        generator.run()
    except KeyboardInterrupt:
        logging.error("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()