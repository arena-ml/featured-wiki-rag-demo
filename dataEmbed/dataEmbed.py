from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import torch
import json
import os
import logging
import sys
import time
import psutil
import traceback
from tqdm import tqdm
import signal
import gc

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("indexing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.warning("Shutdown signal received. Finishing current batch before exiting...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_path(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Path is not readable: {path}")
        logging.info(f"Path verified: {path}")
        return True
    except Exception as e:
        logging.error(f"Error checking path {path}: {str(e)}")
        return False

# Paths
embpath = "/app/jinv3"
modelCachePath = "/app/jinv3/modelCache"
json_file_path = "WikiRC.json"
saveVectorStoreTo = "vectorstore_index.faiss"

# Verify important paths before proceeding
path_checks = [
    check_path(embpath),
    check_path(json_file_path),
]

if not all(path_checks):
    logging.error("Path verification failed. Exiting.")
    sys.exit(1)

CONST_cuda = "cuda"
CONST_mps = "mps"
CONST_cpu="cpu"

def set_device():
    try:
        if torch.cuda.is_available():
            device = torch.device(CONST_cuda)
            logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device(CONST_mps)
            logging.info("MPS available")
            return device
        else:
            device = torch.device(CONST_cpu)
            logging.info("Only CPU available")
            return device
    except Exception as e:
        logging.error(f"Error detecting device: {str(e)}")
        logging.info("Falling back to CPU")
        return torch.device("cpu")

# Resource monitoring
class ResourceMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.last_check = time.time()
        self.device = set_device()
        self.device = torch.device(CONST_cpu)
        self.is_cuda = self.device.type == CONST_cuda
        self.start_time = time.time()
        self.log_header()
        
    def log_header(self):
        header = "TIME(s) | CPU(%) | RAM(GB) | RAM(%)"
        if self.is_cuda:
            header += " | GPU(%) | GPU MEM(GB)"
        logging.info(f"Resource Monitor: {header}")
        
    def log_usage(self, force=False):
        current_time = time.time()
        if force or (current_time - self.last_check) >= self.interval:
            # Calculate elapsed time
            elapsed = current_time - self.start_time
            
            # CPU and RAM
            cpu_percent = psutil.cpu_percent()
            ram_used = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
            ram_percent = psutil.virtual_memory().percent
            
            log_msg = f"{elapsed:.1f}s | CPU: {cpu_percent:.1f}% | RAM: {ram_used:.2f}GB ({ram_percent:.1f}%)"
            
            # GPU if available
            if self.is_cuda:
                try:
                    gpu_percent = torch.cuda.utilization(0)
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
                    log_msg += f" | GPU: {gpu_percent}% | GPU MEM: {gpu_mem_allocated:.2f}GB"
                except Exception as e:
                    log_msg += f" | GPU: Error ({str(e)})"
            
            logging.info(f"Resource usage: {log_msg}")
            self.last_check = current_time

# Initialize the resource monitor
monitor = ResourceMonitor()

# Set the device
deviceDetected = set_device()
# deviceDetected = "cpu"
emb_model_kwargs = {"device": deviceDetected, "local_files_only": True, "trust_remote_code": True}

# Initialize Embeddings with error handling
def initialize_embeddings():
    try:
        logging.info(f"Initializing embeddings model from {embpath}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embpath, 
            model_kwargs=emb_model_kwargs, 
            cache_folder=modelCachePath
        )
        logging.info("Embeddings model initialized successfully")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

# JSON Parsing with better error handling
def parse_json(file_path):
    try:
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
                logging.info(f"Successfully loaded JSON with {len(data)} articles")
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON file format: {str(e)}")
                sys.exit(1)
            
            total_articles = len(data)
            for article_idx, article in enumerate(data):
                if shutdown_requested:
                    logging.info("Shutdown requested during JSON parsing. Exiting.")
                    break
                    
                article_id = article.get("article_id", "")
                title = str(article.get("title", ""))

                
                logging.info(f"Parsing article {article_idx}/{total_articles} - {title}")
                monitor.log_usage()
                
                try:
                    for section in article.get("content", {}).get("sections", []):
                        # section_title = section.get("section_title", "")
                        text = section.get("text", "")
                        
                        #  a list to store all changes 
                        all_changes = []
                        
                        # Process each change 
                        for change in section.get("changes", []):
                            change_summary = change.get("change_summary", "")
                            diff = change.get("diff", "")
                            
                            # Append each change to the list
                            all_changes.append(
                                f"Change Summary: {change_summary}\n"
                                f"Diff: {diff}\n\n"
                            )
                        
                        # Combine all changes 
                        changes_content = "\n".join(all_changes) if all_changes else "No changes in this section."

                        content=(
                            f"[Article Title: {title}]\n"
                            f"[ID:{article_id}]\n"
                            f"Full Text: {text}\n"
                            f"Changes:\n{changes_content}\n"
                        )
                        
                        yield Document(
                            page_content=content,
                            metadata={"articleID": article_id,
                                      "articleTitle":title},
                        )
                except Exception as e:
                    logging.warning(f"Error processing article {article_id}: {str(e)}")
                    # continue  # Skip problematic articles instead of exiting
                    sys.exit(1)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except PermissionError:
        logging.error(f"Permission denied when accessing: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error reading file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

# Optimized Text Splitter with error handling
def split_documents(documents):
    try:
        text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=50)
        for doc in documents:
            try:
                split_docs = text_splitter.split_documents([doc])
                yield from split_docs
            except Exception as e:
                logging.warning(f"Error splitting document: {str(e)}")
                # continue  # Skip problematic documents instead of exiting
                sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to initialize text splitter: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)


def process_and_index():
    vectorstore = None
    total_processed = 0
    start_time = time.time()
    
    try:
        # Initialize embeddings
        embeddings = initialize_embeddings()
        
        # Create a collection of documents
        logging.info("Starting document processing")
        documents = list(parse_json(json_file_path))
        total_documents = len(documents)
        logging.info(f"Total documents to process: {total_documents}")
        
        # Process documents with progress bar
        with tqdm(total=total_documents, desc="Processing documents") as pbar:
            for idx, doc in enumerate(documents):
                if shutdown_requested:
                    logging.info("Shutdown requested. Finishing current document and exiting.")
                    break
                
                logging.info(f"Processing document {idx+1}/{total_documents}")
                monitor.log_usage()
                
                try:
                    texts = list(split_documents([doc]))
                    logging.info(f"Split into {len(texts)} chunks")
                    
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(texts, embeddings)
                        logging.info("Created new FAISS vectorstore")
                    else:
                        vectorstore.add_documents(texts)
                        logging.info(f"Added document to existing vectorstore")
                    
                    total_processed += 1
                    logging.info(f"Progress: {total_processed}/{total_documents} documents processed")
                    
                    # Calculate and log performance metrics
                    elapsed = time.time() - start_time
                    docs_per_second = total_processed / elapsed if elapsed > 0 else 0
                    logging.info(f"Performance: {docs_per_second:.2f} docs/second")
                    
                    # Force garbage collection to free memory
                    # gc.collect()
                    # if deviceDetected.type == CONST_cuda:
                    #     torch.cuda.empty_cache()
                    
                except Exception as e:
                    logging.error(f"Error processing document: {str(e)}")
                    logging.error(traceback.format_exc())
                    sys.exit(1)
                
                pbar.update(1)
        
        # Save the vectorstore
        if vectorstore:
            try:
                logging.info(f"Saving vector store to {saveVectorStoreTo}")
                vectorstore.save_local(saveVectorStoreTo)
                logging.info(f"Vector store saved successfully")
            except Exception as e:
                logging.error(f"Failed to save vector store: {str(e)}")
                logging.error(traceback.format_exc())
                sys.exit(1)
        else:
            logging.warning("No vector store created. Nothing to save.")
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        # Save partial progress if possible
        if vectorstore:
            try:
                partial_path = f"{saveVectorStoreTo}_partial"
                logging.info(f"Saving partial progress to {partial_path}")
                vectorstore.save_local(partial_path)
                logging.info(f"Partial progress saved")
            except Exception as e:
                logging.error(f"Failed to save partial progress: {str(e)}")
    
    except Exception as e:
        logging.error(f"Unexpected error in process_and_index: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # Log final resource usage
        monitor.log_usage(force=True)
        
        # Log completion information
        elapsed = time.time() - start_time
        logging.info(f"Process completed in {elapsed:.2f} seconds")
        logging.info(f"Total documents processed: {total_processed}")

def main():
    try:
        logging.info("Starting indexing process")
        process_and_index()
        logging.info("Indexing process completed successfully")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()