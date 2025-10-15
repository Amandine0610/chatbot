"""
Data preprocessing module for healthcare chatbot dataset.
Handles tokenization, normalization, and data preparation for training.
"""

import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareDataPreprocessor:
    """
    Preprocessor for healthcare conversational data.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the preprocessor with a tokenizer.
        
        Args:
            model_name: Name of the pre-trained model for tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if not present
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
            "unk_token": "<|unk|>"
        }
        
        # Add tokens that don't exist
        tokens_to_add = []
        for token_type, token in special_tokens.items():
            if getattr(self.tokenizer, token_type) is None:
                tokens_to_add.append(token)
                setattr(self.tokenizer, token_type, token)
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": tokens_to_add
            })
        
        self.max_length = 512
        
    def load_dataset(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to the JSON dataset file
            
        Returns:
            List of question-answer pairs
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"['']", "'", text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def create_conversational_format(self, question: str, answer: str) -> str:
        """
        Format question-answer pair for conversational training.
        
        Args:
            question: User question
            answer: Bot response
            
        Returns:
            Formatted conversation string
        """
        # Clean both question and answer
        question = self.clean_text(question)
        answer = self.clean_text(answer)
        
        # Create conversational format
        conversation = f"<|startoftext|>User: {question}<|endoftext|>Assistant: {answer}<|endoftext|>"
        
        return conversation
    
    def tokenize_data(self, conversations: List[str]) -> Dict[str, Any]:
        """
        Tokenize conversations for model training.
        
        Args:
            conversations: List of formatted conversations
            
        Returns:
            Tokenized data dictionary
        """
        logger.info("Tokenizing conversations...")
        
        # Tokenize all conversations
        tokenized = self.tokenizer(
            conversations,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        logger.info(f"Tokenized {len(conversations)} conversations")
        return tokenized
    
    def prepare_dataset(self, data: List[Dict[str, str]], 
                       train_split: float = 0.8) -> DatasetDict:
        """
        Prepare the complete dataset for training.
        
        Args:
            data: List of question-answer pairs
            train_split: Fraction of data for training
            
        Returns:
            DatasetDict with train and validation splits
        """
        logger.info("Preparing dataset...")
        
        # Create conversations
        conversations = []
        for item in data:
            conversation = self.create_conversational_format(
                item["question"], item["answer"]
            )
            conversations.append(conversation)
        
        # Create DataFrame
        df = pd.DataFrame({
            'conversation': conversations,
            'question': [item["question"] for item in data],
            'answer': [item["answer"] for item in data]
        })
        
        # Split data
        train_size = int(len(df) * train_split)
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['conversation'],
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Set labels for language modeling
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        train_dataset = train_dataset.map(add_labels, batched=True)
        val_dataset = val_dataset.map(add_labels, batched=True)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        logger.info(f"Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        
        return dataset_dict
    
    def get_data_statistics(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            data: List of question-answer pairs
            
        Returns:
            Dictionary with dataset statistics
        """
        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]
        
        # Calculate statistics
        stats = {
            "total_samples": len(data),
            "avg_question_length": np.mean([len(q.split()) for q in questions]),
            "avg_answer_length": np.mean([len(a.split()) for a in answers]),
            "max_question_length": max([len(q.split()) for q in questions]),
            "max_answer_length": max([len(a.split()) for a in answers]),
            "min_question_length": min([len(q.split()) for q in questions]),
            "min_answer_length": min([len(a.split()) for a in answers])
        }
        
        return stats


def main():
    """
    Main function to demonstrate data preprocessing.
    """
    # Initialize preprocessor
    preprocessor = HealthcareDataPreprocessor()
    
    # Load dataset
    data = preprocessor.load_dataset("/workspace/data/healthcare_qa_dataset.json")
    
    # Get statistics
    stats = preprocessor.get_data_statistics(data)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Prepare dataset
    dataset = preprocessor.prepare_dataset(data)
    
    print(f"\nDataset splits:")
    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    
    # Show sample
    sample = dataset['train'][0]
    print(f"\nSample conversation:")
    print(f"  Original: {sample['conversation']}")
    print(f"  Tokenized length: {len(sample['input_ids'])}")


if __name__ == "__main__":
    main()