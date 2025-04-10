import json
import os
import sys
import logging
from rich.console import Console
from rich.markdown import Markdown
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

console = Console(width=90)


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



# Paths
llm_path = "/app/Phi-3.5-mini-instruct-Q6_K.gguf"
embpath = "/app/jinv3"
modelCachePath="/app/jinv3/modelCache"

vectorstore_path = (
    "vectorstore_index.faiss"  # .faiss is not a not a file so don't check this
)
inputPath="WikiRC.json"
output_file_path = "WikiRC_ES.json"

check_path(llm_path)
check_path(embpath)


def remove_underscores(input_string):
    return input_string.replace("_", " ")

# Additional helper function for model initialization
def initialize_model(model_path):
    """
    Initialize the Llama model with proper error handling
    """
    try:
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_threads=6,
            n_ctx=32000,
            verbose=True,
        )
        logging.info("Model initialized successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        raise

def construct_prompt(context: str,instruction) -> str:
    """
    Constructs the prompt for the language model.
    
    Args:
        context (str): The retrieved context to base the answer on
        question (str): The user's question
        
    Returns:
        str: The formatted prompt
    """
    return f"""
<|system|>
Your objective is follow instructions given by the user.
<|end|>

<|user|>
instruction : {instruction}
{context}
<|end|>

<|assistant|>
"""

def PhiQnA(query: str, aID: str, instruction: str, retriever) -> tuple[str, list]:
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
        docs = retriever.max_marginal_relevance_search(query,filter={"articleID": aID},k=20,fetch_k=120)
        logging.info(f"Type of retriever output: {type(docs)}")    
        if not docs:
            logging.warning("No documents retrieved for the question")
            return "No relevant documents found.", []
    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        logging.error(f"Retriever type: {type(retriever)}")  # Add this to check retriever type
        return "An error occurred while retrieving relevant documents.", []

    # Step 2: Context Combination and Prompt Construction
    try:
        context = "\n".join(doc.page_content for doc in docs)
        prompt = construct_prompt(context,instruction)
    except Exception as e:
        logging.error(f"Error during prompt construction: {e}")
        return "An error occurred while preparing the response.", docs

    # Step 3: LLM Response Generation
    try:
        with console.status("[bold green]Generating response..."):
            output = model.create_completion(
                prompt=prompt,
                max_tokens=5200,
                stop=["<|end|>"],
                temperature=0.4,
            )
            
        logging.info(f"Raw model output: {output}")
        response = output["choices"][0]["text"].strip()
        
    except Exception as e:
        logging.error(f"Error during LLM response generation: {e}")
        return "An error occurred while generating the model response.", docs

    return response, docs

def main():
    # Initialize embeddings Model
    try:
        global embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embpath,cache_folder=modelCachePath)
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {e}")
    
    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        # Initialize model
        global model
        model = initialize_model(llm_path)

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
            instruction=f"""
            Go through the given context.
            Your objective is to provide a well-structured and accurate summary.
            Consider historical context, significance and key aspects.
            Given context might have recent changes section; if they are meaningful, incorporate them into your summary. 
            Your responses should be strictly from the context provided nothing else.
            Do not mention that it's a summary, and also do not mention anything about instructions given to you.
            """

            aTitle = remove_underscores(str(article["title"]))
            aID = str(article["article_id"])
            retrieverQuery =   aTitle  + " " + aID +  " " + aTitle + " " + aID 

            console.print(Markdown(f"### Retriver Query:\n {retrieverQuery} \n Prompt:\n{instruction}"))

            response, retrivedDocsList = PhiQnA(retrieverQuery,aID,instruction, vectorstore)

            article["embResponse"] = response

            retrivedDocs= " ".join(str(doc) for doc in retrivedDocsList)
            article["retrivedDocs"]=retrivedDocs

            updatedArticle.append(article)
            
            console.print(Markdown(f"### Response:\n{response}"))
            console.print("\n" + "=" * 50 + "\n")
    except Exception as e:
        logging.error(f"Summary Generation failed:{e}")

    del(model)
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
