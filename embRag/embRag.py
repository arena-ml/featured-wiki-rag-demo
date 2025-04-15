import json
import os
import sys
import logging
import time
from rich.console import Console
from rich.markdown import Markdown
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ollama import chat,ChatResponse
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

meter = metrics.get_meter("featuredwikirag.emb_rag")

doc_retrieval_time = meter.create_histogram("documents.retrieval_time", unit="s", description="document retriveal time per query")

empty_results_counter = meter.create_counter(
    "zero_docs_retrieved.count",
    description="Count of queries with zero documents retrieved",
)

summary_generation_time = meter.create_histogram("summary.generation.time",unit="s",description="time taken by llm to generate summary")


# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

console = Console(width=90)
CONST_N_CTX=35000
CONST_MAX_CTX=8200




# Paths
vectorstore_path = (
    "vectorstore_index.faiss"  # .faiss is not a not a file so don't check this
)
inputPath="WikiRC.json"
output_file_path = "WikiRC_ES.json"




def remove_underscores(input_string):
    return input_string.replace("_", " ")



def construct_prompt(context: str) -> str:
    """
    Constructs the prompt for the language model.
    
    Args:
        context (str): The retrieved context to base the answer on
        question (str): The user's question
        
    Returns:
        str: The formatted prompt
    """
    return f"""
Follow this three instructions:
1.Go through the given article then provide a catchy summary, 
consider historical context, significance, key aspects and recent changes made in the article.
3.Your responses should be strictly from the article provided nothing else.
4.Do not mention that it's a summary, and also do not mention anything about instructions given to you.

Article :
{context}
"""

def PhiQnA(query: str, aID: str, retriever) -> tuple[str, list]:
    """
    Perform Q&A based on the context retrieved from the vectorstore.

    Args:
        question (str): The user's question.
        retriever: The initialized retriever object.

    Returns:
        tuple: Response string and the retrieved documents.
    """
    docs = []
    
    # Step 1: Document Retrieval
    try:
        start_time = time.time()

        docs = retriever.max_marginal_relevance_search(query,filter={"articleID": aID},k=40,fetch_k=200)
        logging.info(f"Type of retriever output: {type(docs)}, len_docs: {len(docs)}")    
        if not docs:
            empty_results_counter.add(1, {"query": query, "queryType": "max_marginal_relevance_search"})
            
            logging.warning("No documents retrieved for the question")
            return "NULL", []
        
        doc_retrieval_time.record(time.time() - start_time)
    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        logging.error(f"Retriever type: {type(retriever)}")  # Add this to check retriever type
        return "An error occurred while retrieving relevant documents.", []

    # Step 2: Context Combination and Prompt Construction
    try:
        context = "\n".join(doc.page_content for doc in docs)
        prompt = construct_prompt(context)
    except Exception as e:
        logging.error(f"Error during prompt construction: {e}")
        return "An error occurred while preparing the response.", docs

    # Step 3: LLM Response Generation
    try:
        with console.status("[bold green]Generating response..."):
            start_time = time.time()

            genOpts = {"num_predict":CONST_MAX_CTX,"num_ctx":CONST_N_CTX,"temperature":0.4}

            response: ChatResponse = chat(model='phi3.5:3.8b-mini-instruct-q8_0', messages=[
              {
                'role': 'user',
                'content': prompt,
              },
            ],
            options=genOpts)

            summary_generation_time.record(time.time() - start_time)

        logging.info(f"Raw model output: {response}")

        response = response.message.content
    except Exception as e:
        logging.error(f"Error during LLM response generation: {e}")
        return "An error occurred while generating the model response.", docs

    return response, docs

def main():
    # Initialize embeddings Model
    try:
        global embeddings
        embeddings = embeddings = OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {e}")
    
    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.error(f"Failed to load vectorstore: {e}")
        sys.exit(1)
    
    try:
        with open(inputPath, "r") as file:
            articles = json.load(file)
        if not articles:
            logging.error("The queries file is empty.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to read queries file: {e}")
        sys.exit(1)
    
    try:
        updatedArticle =[]
        for article in articles:
            aTitle = remove_underscores(str(article["title"]))
            aID = str(article["article_id"])
            retrieverQuery =   aTitle  + " " + aID +  " " + aTitle + " " + aID 

            console.print(Markdown(f"### Retriver Query:\n {retrieverQuery}"))

            response, retrivedDocsList = PhiQnA(retrieverQuery, aID, vectorstore)
            article["embResponse"] = response

            if response != "NULL":
                retrivedDocs= " ".join(str(doc) for doc in retrivedDocsList)
                article["retrivedDocs"]=retrivedDocs

            updatedArticle.append(article)
            
            console.print(Markdown(f"### Response:\n{response}"))
            console.print("\n" + "=" * 50 + "\n")
            sys.exit(0)
    except Exception as e:
        logging.error(f"Summary Generation failed:{e}")

    del(embeddings)

    try:
        with open(output_file_path, "w") as outfile:
            json.dump(updatedArticle, outfile, indent=4)
        logging.info(f"Responses saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Failed to write output file: {e}")
        sys.exit(1)
    
    

if __name__ == "__main__":
    main()
