import os
import sys
import json
import time
import logging
# import psutil
import signal
# import gc
from tqdm import tqdm
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings

# OpenTelemetry Metrics Only
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

OTEL_COLLECTOR_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

metrics.set_meter_provider(MeterProvider(
    metric_readers=[PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTEL_COLLECTOR_ENDPOINT),
        export_interval_millis=5000  # every 5 seconds
    )]
))
meter = metrics.get_meter("featuredwikirag.data.embed")

documents_processed = meter.create_counter("documents.processed", unit="1", description="Total documents processed")
processing_time = meter.create_histogram("documents.processing_time", unit="s", description="Processing time per document")
errors_count = meter.create_counter("errors.count", unit="1", description="Number of errors")

document_to_chunks = meter.create_histogram("document.to.chunks",description="number of chunks per document")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("indexing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.warning("Shutdown signal received. Finishing current batch before exiting...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


json_file_path = "WikiRC.json"
saveVectorStoreTo = "vectorstore_index.faiss"

def parse_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            logging.info(f"Successfully loaded JSON with {len(data)} articles")

            for article_idx, article in enumerate(data):
                if shutdown_requested:
                    logging.info("Shutdown requested during JSON parsing. Exiting.")
                    break

                start_time = time.time()
                article_id = article.get("article_id", "")
                title = str(article.get("title", ""))
                logging.info(f"Parsing article {article_idx} - {title}")

                for section in article.get("content", {}).get("sections", []):
                    text = section.get("text", "")
                    changes_content = "\n".join(
                        [f"Change Summary: {c['change_summary']}\nDiff: {c['diff']}\n" 
                         for c in section.get("changes", [])]
                    ) if section.get("changes") else "No changes in this section."

                    content = (
                        f"[Article Title: {title}]\n"
                        f"[ID: {article_id}]\n"
                        f"Full Text: {text}\n"
                        f"Changes:\n{changes_content}\n"
                    )

                    processing_time.record(time.time() - start_time)
                    documents_processed.add(1)

                    yield Document(
                        page_content=content,
                        metadata={"articleID": article_id, "articleTitle": title},
                    )
    except Exception as e:
        logging.error(f"Error parsing JSON: {str(e)}")
        errors_count.add(1)
        sys.exit(1)

def split_documents(documents):
    try:
        text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=50)
        for doc in documents:
            yield from text_splitter.split_documents([doc])
    except Exception as e:
        logging.error(f"Failed to split documents: {str(e)}")
        errors_count.add(1)
        sys.exit(1)

def process_and_index():
    vectorstore = None
    start_time = time.time()
    
    try:
        # embeddings = initialize_embeddings()
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        documents = list(parse_json(json_file_path))
        total_documents = len(documents)
        logging.info(f"Total documents to process: {total_documents}")

        with tqdm(total=total_documents, desc="Processing documents") as pbar:
            for idx, doc in enumerate(documents):
                if shutdown_requested:
                    logging.info("Shutdown requested. Exiting.")
                    break

                try:
                    texts = list(split_documents([doc]))
                    document_to_chunks.record(len(texts))
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(texts, embeddings)
                    else:
                        vectorstore.add_documents(texts)

                    logging.info(f"Processed {idx+1}/{total_documents} documents")
                except Exception as e:
                    logging.error(f"Error processing document: {str(e)}")
                    errors_count.add(1)
                    sys.exit(1)

                pbar.update(1)

        if vectorstore:
            vectorstore.save_local(saveVectorStoreTo)
            logging.info("Vector store saved successfully")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        errors_count.add(1)
        sys.exit(1)
    finally:
        logging.info(f"Process completed in {time.time() - start_time:.2f} seconds")

def main():
    try:
        logging.info("Starting indexing process")
        process_and_index()
        logging.info("Indexing process completed successfully")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        errors_count.add(1)
        sys.exit(1)

if __name__ == "__main__":
    main()
