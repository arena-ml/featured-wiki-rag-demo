import json
import os
import sys
import logging
import torch
from sentence_transformers import util
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Tuple, Any, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("factscore_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONST_CUDA = "cuda"
CONST_MPS = "mps"
MAX_TOKENS = 2000
TEMPERATURE = 0.4

class FactScoreEvaluator:
    def __init__(self, config: Dict[str, str]) -> None:
        """
        Initialize the FactScore evaluator with the provided configuration.
        
        Args:
            config: Dictionary containing paths and configuration parameters
        """
        self.console = Console(width=120)
        self.config = config
        self.validate_paths()
        self.device = self.set_device()
        self.embeddings = self.initialize_embeddings()
        self.llm = self.initialize_llm()
        self.results = []
        
    def validate_paths(self) -> None:
        """Validate all required paths exist and are readable."""
        required_paths = [
            "embpath", 
            "inputFile", 
            "llm_path"
        ]
        
        try:
            for path_key in required_paths:
                path = self.config[path_key]
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Path not found: {path}")
                if not os.access(path, os.R_OK):
                    raise PermissionError(f"Path is not readable: {path}")
                logger.info(f"Path verified: {path}")
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            sys.exit(1)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(str(e))
            sys.exit(1)
            
    def set_device(self) -> torch.device:
        """Determine the best available device for computation."""
        try:
            if torch.cuda.is_available():
                device = torch.device(CONST_CUDA)
                self.console.print("[bold green]CUDA available and enabled[/bold green]")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device(CONST_MPS)
                self.console.print("[bold yellow]MPS available and enabled[/bold yellow]")
            else:
                device = torch.device("cpu")
                self.console.print("[bold yellow]Only CPU available[/bold yellow]")
            return device
        except Exception as e:
            logger.error(f"Error setting device: {e}")
            logger.error(traceback.format_exc())
            self.console.print("[bold red]Error setting device, falling back to CPU[/bold red]")
            return torch.device("cpu")
            
    def initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize the embedding model."""
        try:
            emb_model_kwargs = {
                "device": self.device, 
                "local_files_only": True, 
                "trust_remote_code": True
            }
            
            embed_model = HuggingFaceEmbeddings(
                model_name=self.config["embpath"],
                model_kwargs=emb_model_kwargs,
                cache_folder=self.config.get("modelCachePath", "./modelCache"),
                encode_kwargs={"convert_to_tensor": True}
            )
            logger.info("Embedding model initialized successfully")
            return embed_model
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
    def initialize_llm(self) -> Llama:
        """Initialize the LLM model."""
        try:
            llm = Llama(
                model_path=self.config["llm_path"], 
                n_gpu_layers=-1, 
                n_threads=4,
                n_ctx=32000,
                verbose=True
            )
            logger.info("LLM initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
    def load_articles(self) -> List[Dict[str, Any]]:
        """Load articles from the input file."""
        try:
            with open(self.config["inputFile"], "r", encoding="utf-8") as file:
                articles = json.load(file)
            logger.info(f"Loaded {len(articles)} articles from {self.config['inputFile']}")
            return articles
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load articles: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
    def generate_questions(self, main_text: str) -> List[str]:
        """Generate questions from the main text using LLM."""
        try:
            prompt = f"""
<|system|>
Generate 10 or less concise factual questions from given context.
Ensure questions target key facts. Do not mention anything about instructions given to you.
<|end|>

<|user|>
Summary:
{main_text}
<|end|>

<|assistant|>
"""
            output = self.llm.create_completion(
                prompt=prompt, 
                max_tokens=MAX_TOKENS, 
                stop=["<|end|>"],
                temperature=TEMPERATURE
            )
            
            questions = output["choices"][0]["text"].strip().split("\n")
            valid_questions = [q for q in questions if q and len(q.strip()) > 0]
            
            logger.info(f"Generated {len(valid_questions)} questions")
            return valid_questions
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def answer_from_content(self, content: str, questions: List[str]) -> Dict[str, str]:
        """Generate answers to questions based on the provided content."""
        reference_answers = {}
        
        try:
            for idx, question in enumerate(questions, 1):
                prompt = f"""
<|system|>
Provide a factual and concise response to the question based on the given content only, no outside/training Knowledge allowed.
Do not mention anything about instructions given to you.
If the content is not enough to answer the question, reply with NULL
<|end|>

<|user|>
Content:
{content}

Question: 
{question}
Answer:
<|end|>

<|assistant|>
"""
                output = self.llm.create_completion(
                    prompt=prompt, 
                    max_tokens=MAX_TOKENS, 
                    stop=["<|end|>"],
                    temperature=TEMPERATURE
                )
                answer = output["choices"][0]["text"].strip()
                reference_answers[question] = answer
                
                # Log progress
                if idx % 5 == 0 or idx == len(questions):
                    logger.info(f"Processed {idx}/{len(questions)} questions")
                    
            return reference_answers
        except Exception as e:
            logger.error(f"Failed to generate answers: {e}")
            logger.error(traceback.format_exc())
            return reference_answers
            
    def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between embeddings of two texts."""
        try:
            # Generate embeddings
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)
            
            # Convert to tensors
            tensor1 = torch.tensor(emb1)
            tensor2 = torch.tensor(emb2)
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(tensor1, tensor2).item()
            return similarity
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            logger.error(traceback.format_exc())
            return 0.0
            
    def compute_factscore(self, generated_answers: Dict[str, str], 
                         reference_answers: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        """
        Compute FactScore by comparing generated answers with reference answers.
        
        Returns:
            Tuple containing overall FactScore and per-question similarity scores
        """
        scores = {}
        
        self.console.print("\n[bold]FactScore Evaluation:[/bold]")
        
        try:
            for question, gen_answer in generated_answers.items():
                ref_answer = reference_answers.get(question, "")
                if not ref_answer or ref_answer == "NULL":
                    continue
                
                # Compute similarity
                similarity = self.compute_embedding_similarity(gen_answer, ref_answer)
                scores[question] = similarity
                
                # Print details
                self.console.print(f"\n[cyan]Question:[/cyan] {question}")
                self.console.print(f"[magenta]Generated Answer:[/magenta] {gen_answer}")
                self.console.print(f"[green]Reference Answer:[/green] {ref_answer}")
                self.console.print(f"[yellow]Cosine Similarity:[/yellow] {similarity:.4f}")
            
            # Compute overall FactScore
            factscore = sum(scores.values()) / len(scores) if scores else 0
            self.console.print(f"\n[bold green]Overall FactScore: {factscore:.4f}[/bold green]\n")
            
            return factscore, scores
        except Exception as e:
            logger.error(f"Failed to compute FactScore: {e}")
            logger.error(traceback.format_exc())
            return 0.0, {}
            
    def create_results_table(self, title: str, questions: List[str], 
                           generated_answers: Dict[str, str],
                           reference_answers: Dict[str, str], 
                           similarity_scores: Dict[str, float]) -> Table:
        """Create a Rich table for displaying results."""
        table = Table(title=f"FactScore Evaluation - {title}")
        table.add_column("Question", style="cyan")
        table.add_column("Generated Answer", style="magenta")
        table.add_column("Reference Answer", style="green")
        table.add_column("Similarity", style="yellow")
        
        for question in questions:
            if question in similarity_scores:
                sim_score = similarity_scores[question]
                table.add_row(
                    question,
                    generated_answers.get(question, "-"),
                    reference_answers.get(question, "-"),
                    f"{sim_score:.2f}"
                )
                
        return table
            
    def save_results(self, output_file: Optional[str] = None) -> None:
        """Save evaluation results to a JSON file."""
        try:
            output_path = output_file or self.config.get("outputFile", "factscore_results.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4)
            
            self.console.print(f"\n[bold green]FactScore evaluation completed! Results saved to {output_path}[/bold green]")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            logger.error(traceback.format_exc())
            self.console.print(f"[bold red]Failed to save results: {e}[/bold red]")
            
    def evaluate_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single article and return results."""
        title = article.get("title", "Unknown Title")
        result = {"title": title, "FactScore": 0.0, "error": None}
        
        try:
            sections = article.get("content", {}).get("sections", [])
            if not sections:
                logger.warning(f"No sections found for article: {title}")
                result["error"] = "No sections found"
                return result
            
            main_text = sections[0].get("text", "")
            if not main_text:
                logger.warning(f"No main text found for article: {title}")
                result["error"] = "No main text found"
                return result
            
            emb_response = article.get("embResponse", "")
            if not emb_response:
                logger.warning(f"No embResponse found for article: {title}")
                result["error"] = "No embResponse found"
                return result
            
            self.console.print(f"\n[bold]Processing:[/bold] {title}")
            
            # Generate questions
            questions = self.generate_questions(main_text)
            if not questions:
                result["error"] = "Failed to generate questions"
                return result
            
            # Get reference answers
            reference_answers = self.answer_from_content(main_text, questions)
            
            # Get generated answers
            generated_answers = self.answer_from_content(emb_response, questions)
            
            # Compute FactScore
            factscore, similarity_scores = self.compute_factscore(
                generated_answers, reference_answers
            )
            
            # Create and display table
            table = self.create_results_table(
                title, questions, generated_answers, reference_answers, similarity_scores
            )
            self.console.print(table)
            
            result["FactScore"] = factscore
            result["error"] = None
            result["table"] = {
                "questions": questions,
                "generated_answers": generated_answers,
                "reference_answers": reference_answers,
                "similarity_scores": similarity_scores,
            }
            
            return result
        except Exception as e:
            logger.error(f"Error evaluating article '{title}': {e}")
            logger.error(traceback.format_exc())
            result["error"] = str(e)
            return result
            
    def run(self) -> None:
        """Run the evaluation process on all articles."""
        try:
            articles = self.load_articles()
            total_articles = len(articles)
            
            for idx, article in enumerate(articles, 1):
                self.console.print(f"\n[bold blue]Processing article {idx}/{total_articles}[/bold blue]")
                result = self.evaluate_article(article)
                self.results.append(result)
                
            self.save_results(self.config.get("outputFile"))
        except Exception as e:
            logger.error(f"Failed during evaluation: {e}")
            logger.error(traceback.format_exc())
            self.console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        finally:
            try:
                if hasattr(self, 'llm'):
                    del(self.llm)
                    del(self.embeddings)
                    logger.info("LLM closed successfully")
            except Exception as e:
                logger.error(f"Error closing LLM: {e}")


def main():
    """Main entry point for the program."""
    try:
        # Configuration
        config = {
            "embpath": "/app/jinv3",
            "modelCachePath": "/app/jinv3/modelCache",
            "inputFile": "WikiRC_ESO.json",
            "llm_path": "/app/Phi-3.5-mini-instruct-Q6_K.gguf",
            "outputFile": "factScore.json"
        }
        
        # Create and run evaluator
        evaluator = FactScoreEvaluator(config)
        evaluator.run()
        
        
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        logger.critical(traceback.format_exc())
        Console().print(f"[bold red]Critical error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()