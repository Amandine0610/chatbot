"""
Evaluation module for healthcare chatbot.
Implements various NLP metrics including BLEU, F1-score, and perplexity.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json
import os
from collections import Counter
import re

# Import evaluation metrics
from sacrebleu import BLEU
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareEvaluator:
    """
    Evaluator for healthcare chatbot models.
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run evaluation on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        
        # Initialize scorers
        self.bleu_scorer = BLEU()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load the fine-tuned model for evaluation.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        logger.info(f"Loading model from {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully for evaluation")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """
        Calculate perplexity of the model on given texts.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            Average perplexity
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Calculating perplexity...")
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model outputs
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Accumulate loss and token count
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BLEU scores for predictions against references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with BLEU scores
        """
        logger.info("Calculating BLEU scores...")
        
        # Calculate corpus-level BLEU using sacrebleu
        corpus_bleu = self.bleu_scorer.corpus_score(predictions, [references])
        
        # Calculate sentence-level BLEU scores
        sentence_bleus = []
        smoothing = SmoothingFunction().method1
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]  # BLEU expects list of reference lists
            
            # Calculate sentence BLEU
            try:
                bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                sentence_bleus.append(bleu)
            except:
                sentence_bleus.append(0.0)
        
        results = {
            "corpus_bleu": corpus_bleu.score,
            "avg_sentence_bleu": np.mean(sentence_bleus),
            "bleu_1": corpus_bleu.precisions[0],
            "bleu_2": corpus_bleu.precisions[1] if len(corpus_bleu.precisions) > 1 else 0,
            "bleu_3": corpus_bleu.precisions[2] if len(corpus_bleu.precisions) > 2 else 0,
            "bleu_4": corpus_bleu.precisions[3] if len(corpus_bleu.precisions) > 3 else 0
        }
        
        logger.info(f"BLEU scores calculated: {results}")
        return results
    
    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores for predictions against references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        logger.info("Calculating ROUGE scores...")
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            
            for metric in rouge_scores.keys():
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Calculate averages
        results = {}
        for metric, scores in rouge_scores.items():
            results[f"{metric}_precision"] = np.mean([self.rouge_scorer.score(ref, pred)[metric].precision 
                                                     for pred, ref in zip(predictions, references)])
            results[f"{metric}_recall"] = np.mean([self.rouge_scorer.score(ref, pred)[metric].recall 
                                                  for pred, ref in zip(predictions, references)])
            results[f"{metric}_f1"] = np.mean(scores)
        
        logger.info(f"ROUGE scores calculated: {results}")
        return results
    
    def calculate_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate semantic similarity using simple token overlap.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Average semantic similarity score
        """
        logger.info("Calculating semantic similarity...")
        
        similarities = []
        
        for pred, ref in zip(predictions, references):
            # Simple token-based similarity
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                similarity = 1.0
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                similarity = 0.0
            else:
                intersection = len(pred_tokens.intersection(ref_tokens))
                union = len(pred_tokens.union(ref_tokens))
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        logger.info(f"Average semantic similarity: {avg_similarity:.4f}")
        return avg_similarity
    
    def generate_response(self, question: str, max_length: int = 200, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate response for a given question.
        
        Args:
            question: Input question
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format input
        input_text = f"<|startoftext|>User: {question}<|endoftext|>Assistant:"
        
        # Tokenize
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def evaluate_on_dataset(self, dataset_path: str, num_samples: int = None) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset_path: Path to evaluation dataset
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating model on dataset: {dataset_path}")
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if num_samples:
            data = data[:num_samples]
        
        logger.info(f"Evaluating on {len(data)} samples")
        
        # Generate predictions
        predictions = []
        references = []
        questions = []
        
        for item in data:
            question = item["question"]
            reference = item["answer"]
            
            # Generate prediction
            try:
                prediction = self.generate_response(question)
                predictions.append(prediction)
                references.append(reference)
                questions.append(question)
            except Exception as e:
                logger.warning(f"Error generating response for question: {question[:50]}... Error: {e}")
                continue
        
        if not predictions:
            raise ValueError("No successful predictions generated")
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Calculate metrics
        results = {}
        
        # BLEU scores
        try:
            bleu_results = self.calculate_bleu_score(predictions, references)
            results.update(bleu_results)
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
        
        # ROUGE scores
        try:
            rouge_results = self.calculate_rouge_scores(predictions, references)
            results.update(rouge_results)
        except Exception as e:
            logger.error(f"Error calculating ROUGE: {e}")
        
        # Semantic similarity
        try:
            similarity = self.calculate_semantic_similarity(predictions, references)
            results["semantic_similarity"] = similarity
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
        
        # Perplexity on references
        try:
            perplexity = self.calculate_perplexity(references)
            results["perplexity"] = perplexity
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
        
        # Response length statistics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        results.update({
            "avg_prediction_length": np.mean(pred_lengths),
            "avg_reference_length": np.mean(ref_lengths),
            "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        })
        
        # Store sample predictions for qualitative analysis
        results["sample_predictions"] = [
            {
                "question": q,
                "reference": r,
                "prediction": p
            }
            for q, r, p in zip(questions[:5], references[:5], predictions[:5])
        ]
        
        logger.info("Evaluation completed")
        return results
    
    def qualitative_evaluation(self, test_questions: List[str]) -> List[Dict[str, str]]:
        """
        Perform qualitative evaluation with custom test questions.
        
        Args:
            test_questions: List of test questions
            
        Returns:
            List of question-response pairs
        """
        logger.info("Performing qualitative evaluation...")
        
        results = []
        
        for question in test_questions:
            try:
                response = self.generate_response(question)
                results.append({
                    "question": question,
                    "response": response
                })
                logger.info(f"Q: {question}")
                logger.info(f"A: {response}")
                logger.info("-" * 50)
            except Exception as e:
                logger.error(f"Error generating response for: {question}. Error: {e}")
                results.append({
                    "question": question,
                    "response": f"Error: {str(e)}"
                })
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")


def main():
    """
    Main function to demonstrate evaluation.
    """
    # Test questions for qualitative evaluation
    test_questions = [
        "What are the symptoms of diabetes?",
        "How can I prevent heart disease?",
        "What should I do if I have a fever?",
        "What are the side effects of aspirin?",
        "How much sleep do I need?",
        "What is the best way to lose weight?",  # Out of domain test
        "Can you help me with my math homework?"  # Out of domain test
    ]
    
    # Note: This would require a trained model
    print("Evaluation module created successfully!")
    print("To use the evaluator:")
    print("1. Train a model using the fine_tuning module")
    print("2. Initialize evaluator with the trained model path")
    print("3. Run evaluation on your test dataset")


if __name__ == "__main__":
    main()