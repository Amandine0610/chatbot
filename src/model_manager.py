"""
Model management module for healthcare chatbot.
Handles model selection, loading, and configuration.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Dict, Any, Optional, Tuple
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareModelManager:
    """
    Manages pre-trained models for healthcare chatbot fine-tuning.
    """
    
    SUPPORTED_MODELS = {
        "gpt2": {
            "model_class": GPT2LMHeadModel,
            "tokenizer_class": GPT2Tokenizer,
            "model_name": "gpt2",
            "type": "causal_lm"
        },
        "gpt2-medium": {
            "model_class": GPT2LMHeadModel,
            "tokenizer_class": GPT2Tokenizer,
            "model_name": "gpt2-medium",
            "type": "causal_lm"
        },
        "distilgpt2": {
            "model_class": GPT2LMHeadModel,
            "tokenizer_class": GPT2Tokenizer,
            "model_name": "distilgpt2",
            "type": "causal_lm"
        },
        "dialogpt-small": {
            "model_class": AutoModelForCausalLM,
            "tokenizer_class": AutoTokenizer,
            "model_name": "microsoft/DialoGPT-small",
            "type": "causal_lm"
        },
        "dialogpt-medium": {
            "model_class": AutoModelForCausalLM,
            "tokenizer_class": AutoTokenizer,
            "model_name": "microsoft/DialoGPT-medium",
            "type": "causal_lm"
        },
        "t5-small": {
            "model_class": T5ForConditionalGeneration,
            "tokenizer_class": T5Tokenizer,
            "model_name": "t5-small",
            "type": "seq2seq"
        }
    }
    
    def __init__(self, model_key: str = "dialogpt-medium", device: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            model_key: Key for the model to use
            device: Device to load model on (auto-detected if None)
        """
        self.model_key = model_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = self.SUPPORTED_MODELS.get(model_key)
        
        if not self.model_config:
            raise ValueError(f"Unsupported model: {model_key}. "
                           f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized ModelManager with {model_key} on {self.device}")
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Load the pre-trained model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading {self.model_key} model and tokenizer...")
        
        try:
            # Load tokenizer
            tokenizer_class = self.model_config["tokenizer_class"]
            model_name = self.model_config["model_name"]
            
            self.tokenizer = tokenizer_class.from_pretrained(model_name)
            
            # Add special tokens if not present
            special_tokens = {}
            if self.tokenizer.pad_token is None:
                special_tokens["pad_token"] = "<|pad|>"
            if self.tokenizer.eos_token is None:
                special_tokens["eos_token"] = "<|endoftext|>"
            if self.tokenizer.bos_token is None:
                special_tokens["bos_token"] = "<|startoftext|>"
            if self.tokenizer.unk_token is None:
                special_tokens["unk_token"] = "<|unk|>"
            
            if special_tokens:
                self.tokenizer.add_special_tokens(special_tokens)
            
            # Load model
            model_class = self.model_config["model_class"]
            self.model = model_class.from_pretrained(model_name)
            
            # Resize token embeddings if we added new tokens
            if special_tokens:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"Successfully loaded {self.model_key}")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            "model_key": self.model_key,
            "model_name": self.model_config["model_name"],
            "model_type": self.model_config["type"],
            "device": self.device,
            "num_parameters": self.model.num_parameters(),
            "vocabulary_size": len(self.tokenizer) if self.tokenizer else 0,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'N/A'),
            "hidden_size": getattr(self.model.config, 'hidden_size', 'N/A'),
            "num_layers": getattr(self.model.config, 'num_layers', 
                                getattr(self.model.config, 'n_layer', 'N/A'))
        }
        
        return info
    
    def prepare_for_training(self, learning_rate: float = 5e-5,
                           per_device_train_batch_size: int = 4,
                           per_device_eval_batch_size: int = 4,
                           num_train_epochs: int = 3,
                           warmup_steps: int = 100,
                           logging_steps: int = 10,
                           save_steps: int = 500,
                           output_dir: str = "./models/healthcare_chatbot") -> TrainingArguments:
        """
        Prepare training arguments for fine-tuning.
        
        Args:
            learning_rate: Learning rate for training
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            num_train_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            save_steps: Model saving frequency
            output_dir: Directory to save the model
            
        Returns:
            TrainingArguments object
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb for now
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
        
        return training_args
    
    def get_data_collator(self):
        """
        Get appropriate data collator for the model type.
        
        Returns:
            Data collator for training
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        if self.model_config["type"] == "causal_lm":
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal language modeling
            )
        else:
            # For seq2seq models like T5
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
    
    def save_model(self, save_path: str):
        """
        Save the fine-tuned model and tokenizer.
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model and tokenizer saved to {save_path}")
    
    def load_fine_tuned_model(self, model_path: str):
        """
        Load a fine-tuned model from disk.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        try:
            model_class = self.model_config["model_class"]
            tokenizer_class = self.model_config["tokenizer_class"]
            
            self.model = model_class.from_pretrained(model_path)
            self.tokenizer = tokenizer_class.from_pretrained(model_path)
            
            self.model.to(self.device)
            
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise


def main():
    """
    Main function to demonstrate model management.
    """
    # Initialize model manager
    model_manager = HealthcareModelManager("dialogpt-medium")
    
    # Load model and tokenizer
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Get model info
    info = model_manager.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Prepare training arguments
    training_args = model_manager.prepare_for_training(
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=2
    )
    
    print(f"\nTraining arguments prepared for output directory: {training_args.output_dir}")


if __name__ == "__main__":
    main()