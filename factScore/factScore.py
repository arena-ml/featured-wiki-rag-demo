import json
import os
import sys
import logging
import time
import ollama
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Tuple, Any, Optional
import traceback
import random
import numpy as np
from langchain_ollama import OllamaEmbeddings
# OpenTelemetry Metrics Only
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
import openlit

openlit.init()

OTEL_COLLECTOR_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

metrics.set_meter_provider(MeterProvider(
    metric_readers=[PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTEL_COLLECTOR_ENDPOINT),
        export_interval_millis=5000  # every 5 seconds
    )]
))

meter = metrics.get_meter("featuredwikirag.fact_score")

questionGenerationTime = meter.create_histogram("question.generation.time", unit="s", description="time to generate question from given article")
factScoreEvaluationTime = meter.create_histogram("factscore.evaluation.time",unit="s",description="time taken to compute factscore per article")

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
CONST_MAX_TOKENS = 4000
CONST_LLM_TEMPERATURE = 0.4
CONST_N_CTX=14000

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class FactScoreEvaluator:
    def __init__(self, config: Dict[str, str]) -> None:
        """
        Initialize the FactScore evaluator with the provided configuration.
        
        Args:
            config: Dictionary containing paths and configuration parameters
        """
        self.console = Console(width=120)
        self.config = config
        self.embeddings = self.initialize_embeddings()
        self.results = []
        

    def initialize_embeddings(self) -> OllamaEmbeddings:
        """Initialize the embedding model."""
        try:
            embed_model = OllamaEmbeddings(model="nomic-embed-text",)
            return embed_model
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
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
            Give a numbered list of concise questions, targeting key facts seperated by a newline, from given context.
            Do not mention anything about instructions given to you.
            <|end|>

            <|user|>
            Context:
            {main_text}
            <|end|>

            <|assistant|>
            """
            
            genOpts = {"num_predict":CONST_MAX_TOKENS,"num_ctx":CONST_N_CTX,"temperature":CONST_LLM_TEMPERATURE}
            output = ollama.generate(model='phi3.5:3.8b-mini-instruct-q8_0', prompt=prompt,options=genOpts)

            selected_questions = []
            questions = output['response'].strip().split("\n")

            valid_questions = [q for q in questions if q and len(q.strip()) > 0]
            # If there are more than 10, pick 10 randomly
            if len(valid_questions) > 10:
                selected_questions = random.sample(valid_questions, 10)
            else:
                selected_questions = valid_questions
            

            logger.info(f"Generated {len(selected_questions)} questions")
            return selected_questions
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
                Provide a factual and concise response to the question based on the given content ONLY!.
                If the content is not enough to answer the question , Your response should be just one word: "NULL".
                Do not mention anything about instructions given to you.
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
                genOpts = {"num_predict":CONST_MAX_TOKENS,"num_ctx":CONST_N_CTX,"temperature":CONST_LLM_TEMPERATURE}
                output = ollama.generate(model='phi3.5:3.8b-mini-instruct-q8_0', prompt=prompt,options=genOpts)
                answer = output['response']
                reference_answers[question] = answer
                
                # Log progress
                if idx % 5 == 0 or idx == len(questions):
                    logger.info(f"Processed {idx}/{len(questions)} questions")
                    
            return reference_answers
        except Exception as e:
            logger.error(f"Failed to generate answers: {e}")
            logger.error(traceback.format_exc())
            return reference_answers
            
    def compute_embedding_similarity(self, refAns: str, genAns: str) -> float:
        """Compute cosine similarity between embeddings of two texts."""
        if "NULL" in genAns:
            return 0.0
        try:
            # Generate embeddings
            emb1 = self.embeddings.embed_query(refAns)
            emb2 = self.embeddings.embed_query(genAns)
            
            similarity =  cosine_similarity(emb1, emb2)
            return similarity
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            logger.error(traceback.format_exc())
            return 0.0
            
    def compute_factscore(self,reference_answers: Dict[str, str], generated_answers: Dict[str, str], 
                         ) -> Tuple[float, Dict[str, float]]:
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
                similarity = self.compute_embedding_similarity(ref_answer,gen_answer)
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
            start_time = time.time()
            questions = self.generate_questions(main_text)
            if not questions:
                result["error"] = "Failed to generate questions"
                return result
            questionGenerationTime.record(time.time() - start_time)

            # Get reference answers
            reference_answers = self.answer_from_content(main_text, questions)
            
            # Get generated answers
            generated_answers = self.answer_from_content(emb_response, questions)
            
            # Compute FactScore
            factscore, similarity_scores = self.compute_factscore(
               reference_answers, generated_answers
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
                
                embResponse = article.get("embResponse", "NULL")
    
                if  embResponse!= "NULL":
                    start_time = time.time()

                    result = self.evaluate_article(article)
                    self.results.append(result)

                    factScoreEvaluationTime.record(time.time() - start_time)
                
            self.save_results(self.config.get("outputFile"))
        except Exception as e:
            logger.error(f"Failed during evaluation: {e}")
            logger.error(traceback.format_exc())
            self.console.print(f"[bold red]Evaluation failed: {e}[/bold red]")


def main():
    """Main entry point for the program."""
    try:
        # Configuration
        config = {
            "inputFile": "WikiRC_ESO.json",
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