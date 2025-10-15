#!/usr/bin/env python3
"""
Main training script for healthcare chatbot.
Orchestrates the entire training pipeline from data preprocessing to model evaluation.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import HealthcareDataPreprocessor
from src.model_manager import HealthcareModelManager
from src.fine_tuning import HealthcareFinetuner
from src.evaluation import HealthcareEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Healthcare Chatbot Training Pipeline")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, 
                       default="./data/healthcare_qa_dataset.json",
                       help="Path to the training dataset")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data for training")
    
    # Model arguments
    parser.add_argument("--model_key", type=str, default="dialogpt-medium",
                       choices=["gpt2", "gpt2-medium", "distilgpt2", 
                               "dialogpt-small", "dialogpt-medium", "t5-small"],
                       help="Model to use for training")
    parser.add_argument("--output_dir", type=str, 
                       default="./models/healthcare_chatbot",
                       help="Directory to save the trained model")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    
    # Pipeline control
    parser.add_argument("--skip_hyperparameter_search", action="store_true",
                       help="Skip hyperparameter search and use provided parameters")
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only evaluate an existing model")
    parser.add_argument("--model_path_for_eval", type=str,
                       help="Path to model for evaluation (if evaluate_only)")
    
    # Evaluation arguments
    parser.add_argument("--eval_samples", type=int, default=None,
                       help="Number of samples for evaluation (None for all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log training configuration
    logger.info("=" * 60)
    logger.info("HEALTHCARE CHATBOT TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Model: {args.model_key}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training split: {args.train_split}")
    
    if args.evaluate_only:
        # Evaluation only mode
        logger.info("Running evaluation only...")
        
        model_path = args.model_path_for_eval or os.path.join(args.output_dir, "final_model")
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        # Initialize evaluator
        evaluator = HealthcareEvaluator(model_path)
        
        # Evaluate on dataset
        results = evaluator.evaluate_on_dataset(
            args.dataset_path, 
            num_samples=args.eval_samples
        )
        
        # Save evaluation results
        eval_results_path = os.path.join(args.output_dir, "evaluation_results.json")
        evaluator.save_evaluation_results(results, eval_results_path)
        
        # Print key metrics
        logger.info("Evaluation Results:")
        logger.info(f"  BLEU Score: {results.get('corpus_bleu', 'N/A'):.4f}")
        logger.info(f"  ROUGE-L F1: {results.get('rougeL_f1', 'N/A'):.4f}")
        logger.info(f"  Semantic Similarity: {results.get('semantic_similarity', 'N/A'):.4f}")
        logger.info(f"  Perplexity: {results.get('perplexity', 'N/A'):.4f}")
        
        # Qualitative evaluation
        test_questions = [
            "What are the symptoms of diabetes?",
            "How can I prevent heart disease?",
            "What should I do if I have a fever?",
            "How much water should I drink daily?",
            "What are healthy snack options?"
        ]
        
        qualitative_results = evaluator.qualitative_evaluation(test_questions)
        
        logger.info("\nQualitative Evaluation:")
        for result in qualitative_results:
            logger.info(f"Q: {result['question']}")
            logger.info(f"A: {result['response']}")
            logger.info("-" * 40)
        
        return
    
    # Full training pipeline
    try:
        # Step 1: Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        preprocessor = HealthcareDataPreprocessor()
        
        # Load and prepare dataset
        raw_data = preprocessor.load_dataset(args.dataset_path)
        dataset = preprocessor.prepare_dataset(raw_data, args.train_split)
        
        # Get data statistics
        stats = preprocessor.get_data_statistics(raw_data)
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Step 2: Model Setup
        logger.info("Step 2: Model Setup")
        model_manager = HealthcareModelManager(args.model_key, args.output_dir)
        model, tokenizer = model_manager.load_model_and_tokenizer()
        
        # Log model info
        model_info = model_manager.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        # Step 3: Fine-tuning
        logger.info("Step 3: Fine-tuning")
        finetuner = HealthcareFinetuner(args.model_key, args.output_dir)
        
        if args.skip_hyperparameter_search:
            # Use provided hyperparameters
            hyperparameters = {
                "learning_rate": args.learning_rate,
                "per_device_train_batch_size": args.batch_size,
                "num_train_epochs": args.epochs,
                "warmup_steps": args.warmup_steps
            }
            logger.info(f"Using provided hyperparameters: {hyperparameters}")
        else:
            # Perform hyperparameter search
            logger.info("Performing hyperparameter search...")
            hyperparameters = None  # Will trigger search in fine_tune method
        
        # Fine-tune the model
        training_results = finetuner.fine_tune(dataset, hyperparameters)
        
        logger.info("Training completed!")
        logger.info(f"Final validation loss: {training_results['eval_loss']:.4f}")
        logger.info(f"Model saved to: {training_results['model_path']}")
        
        # Step 4: Evaluation
        logger.info("Step 4: Model Evaluation")
        
        # Initialize evaluator with trained model
        evaluator = HealthcareEvaluator(training_results['model_path'])
        
        # Evaluate on validation set
        eval_results = evaluator.evaluate_on_dataset(
            args.dataset_path, 
            num_samples=args.eval_samples
        )
        
        # Save evaluation results
        eval_results_path = os.path.join(args.output_dir, "final_evaluation_results.json")
        evaluator.save_evaluation_results(eval_results, eval_results_path)
        
        # Print evaluation summary
        logger.info("Final Evaluation Results:")
        logger.info(f"  BLEU Score: {eval_results.get('corpus_bleu', 'N/A'):.4f}")
        logger.info(f"  ROUGE-L F1: {eval_results.get('rougeL_f1', 'N/A'):.4f}")
        logger.info(f"  Semantic Similarity: {eval_results.get('semantic_similarity', 'N/A'):.4f}")
        logger.info(f"  Perplexity: {eval_results.get('perplexity', 'N/A'):.4f}")
        
        # Qualitative evaluation
        test_questions = [
            "What are the symptoms of diabetes?",
            "How can I prevent heart disease?",
            "What should I do if I have a fever?",
            "How much water should I drink daily?",
            "What are healthy snack options?",
            "Can you help me with my homework?",  # Out of domain
            "What's the weather like?"  # Out of domain
        ]
        
        qualitative_results = evaluator.qualitative_evaluation(test_questions)
        
        # Save qualitative results
        qualitative_path = os.path.join(args.output_dir, "qualitative_evaluation.json")
        with open(qualitative_path, 'w') as f:
            json.dump(qualitative_results, f, indent=2)
        
        logger.info("\nQualitative Evaluation Sample:")
        for i, result in enumerate(qualitative_results[:3]):
            logger.info(f"Q{i+1}: {result['question']}")
            logger.info(f"A{i+1}: {result['response']}")
            logger.info("-" * 40)
        
        # Step 5: Create summary report
        logger.info("Step 5: Creating Summary Report")
        
        summary_report = {
            "training_config": {
                "model_key": args.model_key,
                "dataset_path": args.dataset_path,
                "train_split": args.train_split,
                "hyperparameters": training_results.get("hyperparameters", {}),
                "training_time": training_results.get("train_runtime", 0)
            },
            "dataset_stats": stats,
            "model_info": model_info,
            "training_results": {
                "final_train_loss": training_results.get("train_loss", 0),
                "final_eval_loss": training_results.get("eval_loss", 0)
            },
            "evaluation_metrics": {
                "bleu_score": eval_results.get("corpus_bleu", 0),
                "rouge_l_f1": eval_results.get("rougeL_f1", 0),
                "semantic_similarity": eval_results.get("semantic_similarity", 0),
                "perplexity": eval_results.get("perplexity", 0)
            },
            "model_path": training_results["model_path"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary report
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Final success message
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Trained model available at: {training_results['model_path']}")
        logger.info("To test the chatbot:")
        logger.info(f"  python -m src.chatbot {training_results['model_path']}")
        logger.info("To launch web interface:")
        logger.info(f"  python -m src.web_interface --model_path {training_results['model_path']}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()