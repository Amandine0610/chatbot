"""
Fine-tuning pipeline for healthcare chatbot.
Implements training loop with hyperparameter tuning and monitoring.
"""

import torch
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .model_manager import HealthcareModelManager
    from .data_preprocessing import HealthcareDataPreprocessor
    from .evaluation import HealthcareEvaluator
except ImportError:
    from model_manager import HealthcareModelManager
    from data_preprocessing import HealthcareDataPreprocessor
    from evaluation import HealthcareEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareFinetuner:
    """
    Fine-tuning pipeline for healthcare chatbot models.
    """
    
    def __init__(self, model_key: str = "dialogpt-medium", 
                 output_dir: str = "./models/healthcare_chatbot"):
        """
        Initialize the fine-tuner.
        
        Args:
            model_key: Key for the model to fine-tune
            output_dir: Directory to save fine-tuned models
        """
        self.model_key = model_key
        self.output_dir = output_dir
        self.model_manager = HealthcareModelManager(model_key)
        self.preprocessor = HealthcareDataPreprocessor(
            self.model_manager.model_config["model_name"]
        )
        self.evaluator = None
        
        # Training history
        self.training_history = []
        self.best_model_path = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data(self, dataset_path: str, train_split: float = 0.8):
        """
        Prepare data for training.
        
        Args:
            dataset_path: Path to the dataset JSON file
            train_split: Fraction of data for training
            
        Returns:
            Prepared dataset dictionary
        """
        logger.info("Preparing data for training...")
        
        # Load and preprocess data
        raw_data = self.preprocessor.load_dataset(dataset_path)
        dataset = self.preprocessor.prepare_dataset(raw_data, train_split)
        
        # Get data statistics
        stats = self.preprocessor.get_data_statistics(raw_data)
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return dataset
    
    def hyperparameter_search(self, dataset, 
                            param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            dataset: Prepared dataset
            param_grid: Grid of hyperparameters to search
            
        Returns:
            Best hyperparameters found
        """
        if param_grid is None:
            param_grid = {
                "learning_rate": [3e-5, 5e-5, 1e-4],
                "per_device_train_batch_size": [2, 4],
                "num_train_epochs": [2, 3, 4],
                "warmup_steps": [50, 100, 200]
            }
        
        logger.info("Starting hyperparameter search...")
        logger.info(f"Parameter grid: {param_grid}")
        
        best_score = float('inf')
        best_params = None
        results = []
        
        # Generate all combinations
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        logger.info(f"Testing {len(combinations)} hyperparameter combinations...")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")
            
            try:
                # Train with current parameters
                result = self._train_with_params(dataset, params, trial_name=f"trial_{i+1}")
                results.append({**params, **result})
                
                # Check if this is the best so far
                if result['eval_loss'] < best_score:
                    best_score = result['eval_loss']
                    best_params = params
                    logger.info(f"New best parameters found: {params} (eval_loss: {best_score:.4f})")
                
            except Exception as e:
                logger.error(f"Error in trial {i+1}: {e}")
                results.append({**params, "eval_loss": float('inf'), "error": str(e)})
        
        # Save results
        results_df = pd.DataFrame(results)
        results_path = os.path.join(self.output_dir, "hyperparameter_search_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Hyperparameter search results saved to {results_path}")
        
        # Plot results
        self._plot_hyperparameter_results(results_df)
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation loss: {best_score:.4f}")
        
        return best_params
    
    def _train_with_params(self, dataset, params: Dict[str, Any], 
                          trial_name: str = "trial") -> Dict[str, float]:
        """
        Train model with specific hyperparameters.
        
        Args:
            dataset: Training dataset
            params: Hyperparameters to use
            trial_name: Name for this trial
            
        Returns:
            Training results
        """
        # Load fresh model for each trial
        model, tokenizer = self.model_manager.load_model_and_tokenizer()
        
        # Prepare training arguments
        trial_output_dir = os.path.join(self.output_dir, trial_name)
        training_args = TrainingArguments(
            output_dir=trial_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=params["num_train_epochs"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params.get("per_device_eval_batch_size", 4),
            warmup_steps=params["warmup_steps"],
            learning_rate=params["learning_rate"],
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            save_total_limit=1  # Only keep the best model
        )
        
        # Get data collator
        data_collator = self.model_manager.get_data_collator()
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        # Clean up to save memory
        del trainer
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "train_runtime": train_result.metrics["train_runtime"],
            "eval_runtime": eval_result["eval_runtime"]
        }
    
    def fine_tune(self, dataset, hyperparameters: Optional[Dict[str, Any]] = None,
                  use_early_stopping: bool = True) -> Dict[str, Any]:
        """
        Fine-tune the model with given hyperparameters.
        
        Args:
            dataset: Prepared dataset
            hyperparameters: Hyperparameters to use (will search if None)
            use_early_stopping: Whether to use early stopping
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting fine-tuning process...")
        
        # Use provided hyperparameters or search for best ones
        if hyperparameters is None:
            logger.info("No hyperparameters provided, starting search...")
            hyperparameters = self.hyperparameter_search(dataset)
        
        logger.info(f"Fine-tuning with hyperparameters: {hyperparameters}")
        
        # Load model and tokenizer
        model, tokenizer = self.model_manager.load_model_and_tokenizer()
        
        # Prepare training arguments
        final_output_dir = os.path.join(self.output_dir, "final_model")
        training_args = TrainingArguments(
            output_dir=final_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=hyperparameters["num_train_epochs"],
            per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
            per_device_eval_batch_size=hyperparameters.get("per_device_eval_batch_size", 4),
            warmup_steps=hyperparameters["warmup_steps"],
            learning_rate=hyperparameters["learning_rate"],
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(final_output_dir, "logs"),
            report_to=None,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            save_total_limit=2
        )
        
        # Get data collator
        data_collator = self.model_manager.get_data_collator()
        
        # Prepare callbacks
        callbacks = []
        if use_early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(final_output_dir)
        
        self.best_model_path = final_output_dir
        
        # Prepare results
        results = {
            "hyperparameters": hyperparameters,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "train_runtime": train_result.metrics["train_runtime"],
            "eval_runtime": eval_result["eval_runtime"],
            "model_path": final_output_dir,
            "training_history": trainer.state.log_history
        }
        
        # Save training results
        results_path = os.path.join(final_output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Plot training curves
        self._plot_training_curves(trainer.state.log_history, final_output_dir)
        
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Final model saved to: {final_output_dir}")
        logger.info(f"Final validation loss: {eval_result['eval_loss']:.4f}")
        
        return results
    
    def _plot_hyperparameter_results(self, results_df: pd.DataFrame):
        """
        Plot hyperparameter search results.
        
        Args:
            results_df: DataFrame with hyperparameter search results
        """
        try:
            plt.figure(figsize=(15, 10))
            
            # Filter out failed trials
            valid_results = results_df[results_df['eval_loss'] != float('inf')]
            
            if len(valid_results) == 0:
                logger.warning("No valid results to plot")
                return
            
            # Plot 1: Learning rate vs eval loss
            plt.subplot(2, 3, 1)
            sns.boxplot(data=valid_results, x='learning_rate', y='eval_loss')
            plt.title('Learning Rate vs Eval Loss')
            plt.xticks(rotation=45)
            
            # Plot 2: Batch size vs eval loss
            plt.subplot(2, 3, 2)
            sns.boxplot(data=valid_results, x='per_device_train_batch_size', y='eval_loss')
            plt.title('Batch Size vs Eval Loss')
            
            # Plot 3: Epochs vs eval loss
            plt.subplot(2, 3, 3)
            sns.boxplot(data=valid_results, x='num_train_epochs', y='eval_loss')
            plt.title('Epochs vs Eval Loss')
            
            # Plot 4: Warmup steps vs eval loss
            plt.subplot(2, 3, 4)
            sns.boxplot(data=valid_results, x='warmup_steps', y='eval_loss')
            plt.title('Warmup Steps vs Eval Loss')
            
            # Plot 5: Best combinations
            plt.subplot(2, 3, 5)
            top_10 = valid_results.nsmallest(min(10, len(valid_results)), 'eval_loss')
            plt.barh(range(len(top_10)), top_10['eval_loss'])
            plt.yticks(range(len(top_10)), [f"Config {i+1}" for i in range(len(top_10))])
            plt.xlabel('Eval Loss')
            plt.title('Top 10 Configurations')
            
            # Plot 6: Training time vs performance
            plt.subplot(2, 3, 6)
            plt.scatter(valid_results['train_runtime'], valid_results['eval_loss'])
            plt.xlabel('Training Runtime (seconds)')
            plt.ylabel('Eval Loss')
            plt.title('Training Time vs Performance')
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "hyperparameter_search_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Hyperparameter search plots saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting hyperparameter results: {e}")
    
    def _plot_training_curves(self, log_history: List[Dict], output_dir: str):
        """
        Plot training curves.
        
        Args:
            log_history: Training log history
            output_dir: Directory to save plots
        """
        try:
            # Extract training and validation losses
            train_logs = [log for log in log_history if 'loss' in log]
            eval_logs = [log for log in log_history if 'eval_loss' in log]
            
            if not train_logs or not eval_logs:
                logger.warning("Insufficient data for plotting training curves")
                return
            
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Training and validation loss
            plt.subplot(1, 2, 1)
            train_steps = [log['step'] for log in train_logs]
            train_losses = [log['loss'] for log in train_logs]
            eval_steps = [log['step'] for log in eval_logs]
            eval_losses = [log['eval_loss'] for log in eval_logs]
            
            plt.plot(train_steps, train_losses, label='Training Loss', alpha=0.7)
            plt.plot(eval_steps, eval_losses, label='Validation Loss', alpha=0.7)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Learning rate schedule
            plt.subplot(1, 2, 2)
            lr_logs = [log for log in log_history if 'learning_rate' in log]
            if lr_logs:
                lr_steps = [log['step'] for log in lr_logs]
                lr_values = [log['learning_rate'] for log in lr_logs]
                plt.plot(lr_steps, lr_values, label='Learning Rate')
                plt.xlabel('Steps')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "training_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")


def main():
    """
    Main function to demonstrate fine-tuning pipeline.
    """
    # Initialize fine-tuner
    finetuner = HealthcareFinetuner("dialogpt-medium")
    
    # Prepare data
    dataset = finetuner.prepare_data("/workspace/data/healthcare_qa_dataset.json")
    
    # Define hyperparameters for quick testing
    hyperparameters = {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 2,
        "num_train_epochs": 2,
        "warmup_steps": 100
    }
    
    # Fine-tune the model
    results = finetuner.fine_tune(dataset, hyperparameters)
    
    print("Fine-tuning completed!")
    print(f"Final validation loss: {results['eval_loss']:.4f}")
    print(f"Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()