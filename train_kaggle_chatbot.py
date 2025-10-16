#!/usr/bin/env python3
"""
Training script for healthcare chatbot using Kaggle ai-medical-chatbot dataset.
Handles dataset loading, preprocessing, and training with the Kaggle data.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.kaggle_data_loader import KaggleMedicalDataLoader
from src.data_preprocessing import HealthcareDataPreprocessor
from src.model_manager import HealthcareModelManager
from src.fine_tuning import HealthcareFinetuner
from src.evaluation import HealthcareEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kaggle_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline for Kaggle dataset."""
    parser = argparse.ArgumentParser(description="Healthcare Chatbot Training with Kaggle Dataset")
    
    # Dataset arguments
    parser.add_argument("--kaggle_dataset_path", type=str, required=True,
                       help="Path to the Kaggle ai-medical-chatbot dataset file")
    parser.add_argument("--processed_dataset_path", type=str, 
                       default="./data/kaggle_medical_dataset.json",
                       help="Path to save/load processed dataset")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use (for testing)")
    
    # Model arguments
    parser.add_argument("--model_key", type=str, default="dialogpt-medium",
                       choices=["gpt2", "gpt2-medium", "distilgpt2", 
                               "dialogpt-small", "dialogpt-medium", "t5-small"],
                       help="Model to use for training")
    parser.add_argument("--output_dir", type=str, 
                       default="./models/kaggle_healthcare_chatbot",
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
    parser.add_argument("--skip_preprocessing", action="store_true",
                       help="Skip data preprocessing if processed file exists")
    parser.add_argument("--skip_hyperparameter_search", action="store_true",
                       help="Skip hyperparameter search and use provided parameters")
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only evaluate an existing model")
    parser.add_argument("--model_path_for_eval", type=str,
                       help="Path to model for evaluation (if evaluate_only)")
    
    # Evaluation arguments
    parser.add_argument("--eval_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("HEALTHCARE CHATBOT TRAINING WITH KAGGLE DATASET")
    logger.info("=" * 70)
    logger.info(f"Kaggle dataset: {args.kaggle_dataset_path}")
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
        
        # Load processed dataset for evaluation
        if not os.path.exists(args.processed_dataset_path):
            logger.error(f"Processed dataset not found at {args.processed_dataset_path}")
            logger.info("Please run training first to process the dataset")
            return
        
        # Initialize evaluator
        evaluator = HealthcareEvaluator(model_path)
        
        # Evaluate on dataset
        results = evaluator.evaluate_on_dataset(
            args.processed_dataset_path, 
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
        
        return
    
    # Full training pipeline
    try:
        # Step 1: Load and Process Kaggle Dataset
        logger.info("Step 1: Loading and Processing Kaggle Dataset")
        
        if args.skip_preprocessing and os.path.exists(args.processed_dataset_path):
            logger.info(f"Skipping preprocessing, using existing file: {args.processed_dataset_path}")
            with open(args.processed_dataset_path, 'r') as f:
                processed_data = json.load(f)
        else:
            # Load Kaggle dataset
            logger.info(f"Loading Kaggle dataset from: {args.kaggle_dataset_path}")
            kaggle_loader = KaggleMedicalDataLoader(args.kaggle_dataset_path)
            
            # Process the data
            processed_data = kaggle_loader.process_data()
            
            # Apply sample limit if specified
            if args.max_samples and len(processed_data) > args.max_samples:
                logger.info(f"Limiting dataset to {args.max_samples} samples")
                processed_data = processed_data[:args.max_samples]
            
            # Get and display statistics
            stats = kaggle_loader.get_data_statistics()
            logger.info("Kaggle Dataset Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: {list(value.keys())[:5]}...")  # Show first 5 categories
                elif isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
            
            # Save processed data
            kaggle_loader.save_processed_data(args.processed_dataset_path)
            
            # Show preview
            preview = kaggle_loader.preview_data(2)
            logger.info("Data Preview:")
            for i, item in enumerate(preview, 1):
                logger.info(f"  Sample {i}:")
                logger.info(f"    Q: {item['question'][:80]}...")
                logger.info(f"    A: {item['answer'][:80]}...")
        
        # Step 2: Data Preprocessing for Training
        logger.info("Step 2: Preparing Data for Training")
        
        # Use the healthcare preprocessor with the processed Kaggle data
        preprocessor = HealthcareDataPreprocessor()
        dataset = preprocessor.prepare_dataset(processed_data, args.train_split)
        
        logger.info(f"Training samples: {len(dataset['train'])}")
        logger.info(f"Validation samples: {len(dataset['validation'])}")
        
        # Step 3: Model Setup
        logger.info("Step 3: Model Setup")
        model_manager = HealthcareModelManager(args.model_key)
        model, tokenizer = model_manager.load_model_and_tokenizer()
        
        # Log model info
        model_info = model_manager.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        # Step 4: Fine-tuning
        logger.info("Step 4: Fine-tuning")
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
            # Perform hyperparameter search with smaller grid for large datasets
            logger.info("Performing hyperparameter search...")
            hyperparameters = None  # Will trigger search in fine_tune method
        
        # Fine-tune the model
        training_results = finetuner.fine_tune(dataset, hyperparameters)
        
        logger.info("Training completed!")
        logger.info(f"Final validation loss: {training_results['eval_loss']:.4f}")
        logger.info(f"Model saved to: {training_results['model_path']}")
        
        # Step 5: Evaluation
        logger.info("Step 5: Model Evaluation")
        
        # Initialize evaluator with trained model
        evaluator = HealthcareEvaluator(training_results['model_path'])
        
        # Evaluate on validation set
        eval_results = evaluator.evaluate_on_dataset(
            args.processed_dataset_path, 
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
        
        # Qualitative evaluation with medical questions
        test_questions = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "What causes chest pain?",
            "How to prevent heart disease?",
            "What are the side effects of aspirin?",
            "When should I see a doctor for headaches?",
            "What is the treatment for pneumonia?",
            "How is COVID-19 diagnosed?",
            "What foods should diabetics avoid?",
            "How long does it take to recover from surgery?"
        ]
        
        qualitative_results = evaluator.qualitative_evaluation(test_questions)
        
        # Save qualitative results
        qualitative_path = os.path.join(args.output_dir, "qualitative_evaluation.json")
        with open(qualitative_path, 'w') as f:
            json.dump(qualitative_results, f, indent=2)
        
        logger.info("\nQualitative Evaluation Sample:")
        for i, result in enumerate(qualitative_results[:3]):
            logger.info(f"Q{i+1}: {result['question']}")
            logger.info(f"A{i+1}: {result['response'][:150]}...")
            logger.info("-" * 50)
        
        # Step 6: Create Summary Report
        logger.info("Step 6: Creating Summary Report")
        
        summary_report = {
            "training_config": {
                "model_key": args.model_key,
                "kaggle_dataset_path": args.kaggle_dataset_path,
                "processed_dataset_path": args.processed_dataset_path,
                "train_split": args.train_split,
                "max_samples": args.max_samples,
                "hyperparameters": training_results.get("hyperparameters", {}),
                "training_time": training_results.get("train_runtime", 0)
            },
            "dataset_stats": {
                "total_samples": len(processed_data),
                "training_samples": len(dataset['train']),
                "validation_samples": len(dataset['validation'])
            },
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
        summary_path = os.path.join(args.output_dir, "kaggle_training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Final success message
        logger.info("=" * 70)
        logger.info("KAGGLE DATASET TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Trained model available at: {training_results['model_path']}")
        logger.info(f"Dataset size: {len(processed_data)} samples")
        logger.info("To test the chatbot:")
        logger.info(f"  python -m src.chatbot {training_results['model_path']}")
        logger.info("To launch web interface:")
        logger.info(f"  python -m src.web_interface --model_path {training_results['model_path']}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()