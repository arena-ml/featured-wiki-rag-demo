import sys
import os
import time
import json
import logging
import signal
import langchain
import langchain.schema
import langchain.text_splitter
from tqdm import tqdm
import langchain_ollama
from langchain_chroma import Chroma
import chromadb

import openlit

openlit.init(collect_gpu_stats=True, capture_message_content=False)


from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
#  from opentelemetry.sdk._logs.export import ConsoleLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

logger_provider = LoggerProvider(shutdown_on_exit=True,resource=Resource.create(
        {
            "service.name": "generate.embeddings",
        }
    ),)

set_logger_provider(logger_provider)
OTEL_COLLECTOR_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_RSRC_ENDPOINT = os.getenv("OTEL_RESOURCE_ATTRIBUTE")
otlp_exporter = OTLPLogExporter(endpoint=OTEL_COLLECTOR_ENDPOINT, insecure=True)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_exporter))

#  console_exporter = ConsoleLogExporter()
#  logger_provider.add_log_record_processor(BatchLogRecordProcessor(console_exporter))

handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("indexing.log"), logging.StreamHandler(sys.stdout)],
)

shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    logging.warning(
        "Shutdown signal received. Finishing current batch before exiting..."
    )
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


json_file_path = "WikiRC_StepOne.json"
saveVectorStoreTo = "article_embeddings_db"

chroma_client_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory=saveVectorStoreTo,
    anonymized_telemetry=False,
)


def parse_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            logging.info(f"Successfully loaded JSON with {len(data)} articles")

            for article_idx, article in enumerate(data):
                if shutdown_requested:
                    logging.info("Shutdown requested during JSON parsing. Exiting.")
                    break

                # start_time = time.time()
                article_id = article.get("article_id", "")
                title = str(article.get("title", ""))
                logging.info(f"Parsing article {article_idx} - {title}")

                for section in article.get("content", {}).get("sections", []):
                    text = section.get("text", "")
                    changes_content = (
                        "\n".join(
                            [
                                f"Change Summary: {c['change_summary']}\nDiff: {c['diff']}\n"
                                for c in section.get("changes", [])
                            ]
                        )
                        if section.get("changes")
                        else "No changes in this section."
                    )

                    content = (
                        f"[Article Title: {title}]\n"
                        f"[ID: {article_id}]\n"
                        f"Full Text: {text}\n"
                        f"Changes:\n{changes_content}\n"
                    )

                    # processing_time.record(time.time() - start_time)
                    # documents_processed.add(1)

                    yield langchain.schema.Document(
                        page_content=content,
                        metadata={
                            "source": "https://api.wikimedia.org",
                            "articleID": article_id,
                            "articleTitle": title,
                        },
                    )
    except Exception as e:
        logging.error(f"Error parsing JSON: {str(e)}")
        # errors_count.add(1)
        sys.exit(1)


def split_documents(documents):
    try:
        text_splitter = langchain.text_splitter.TokenTextSplitter(
            chunk_size=2000, chunk_overlap=50
        )
        for doc in documents:
            splits = text_splitter.split_text(doc.page_content)
            for chunk in splits:
                yield langchain.schema.Document(
                    page_content=chunk, metadata=doc.metadata  # ✅ preserve metadata
                )
    except Exception as e:
        logging.error(f"Failed to split documents: {str(e)}")
        sys.exit(1)


def process_and_index():
    vectorstore = None
    start_time = time.time()

    try:
        embeddings = langchain_ollama.OllamaEmbeddings(model="nomic-embed-text")

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
                    # document_to_chunks.record(len(texts))
                    if vectorstore is None:
                        # vectorstore = langchain_community.vectorstores.FAISS.from_documents(texts, embeddings)
                        vectorstore = Chroma(
                            collection_name="article_embeddings",
                            client_settings=chroma_client_settings,
                            embedding_function=embeddings,
                            persist_directory=saveVectorStoreTo,
                            # Where to save data locally, remove if not necessary
                        )
                        vectorstore.add_documents(texts)
                    else:
                        vectorstore.add_documents(texts)

                    logging.info(f"Processed {idx+1}/{total_documents} documents")
                except Exception as e:
                    logging.error(f"Error processing document: {str(e)}")
                    # errors_count.add(1)
                    sys.exit(1)

                pbar.update(1)

        # if vectorstore:
        #     vectorstore.save_local(saveVectorStoreTo)
        #     logging.info("Vector store saved successfully")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        # errors_count.add(1)
        sys.exit(1)
    finally:
        logging.info(f"Process completed in {time.time() - start_time:.2f} seconds")


def main():
    logging.getLogger().addHandler(handler)
    main_logger = logging.getLogger("generate.embeddings.main")
    main_logger.setLevel(logging.INFO)
    openlit.logger.info("Generating embeddings start time",time=time.time())
    process_and_index()
    main_logger.info("start of generate.embeddings: %s",time.time() )
    try:
        logging.info("Starting indexing process")
        process_and_index()
        logging.info("Indexing process completed successfully")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        main_logger.info("end of generate.embeddings: %s",time.time())
        openlit.logger.info("End of generate.embeddings: %s",time.time())
        # errors_count.add(1)
        sys.exit(1)
        


if __name__ == "__main__":
    main()
