#!/usr/bin/env python3
"""
WORKING Training Script for Kaggle Healthcare Chatbot
This version fixes all known issues and just works.
"""

import os
import sys
import argparse
import logging
import json
import glob

# Add src to path
sys.path.append('src')

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def find_dataset_automatically():
    """Find the user's dataset file automatically."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Searching for dataset files...")
    
    # Common patterns
    patterns = [
        "*medical*.csv", "*medical*.json", 
        "*chatbot*.csv", "*chatbot*.json",
        "*.csv", "*.json"
    ]
    
    # Search locations
    search_dirs = [
        ".", "./data",
        "C:/Users/hp/Downloads" if os.name == 'nt' else "~/Downloads",
        "C:/Users/hp/Documents" if os.name == 'nt' else "~/Documents"
    ]
    
    found_files = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for pattern in patterns:
                files = glob.glob(os.path.join(search_dir, pattern))
                for file in files:
                    if os.path.isfile(file) and os.path.getsize(file) > 1000:
                        found_files.append(file)
    
    # Remove duplicates and sort by size
    unique_files = list(set(found_files))
    unique_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    if unique_files:
        logger.info(f"✅ Found {len(unique_files)} dataset files")
        for i, file in enumerate(unique_files[:5], 1):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            logger.info(f"   {i}. {os.path.basename(file)} ({size_mb:.1f} MB)")
        return unique_files[0]  # Return largest file
    
    return None

def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("🏥 WORKING HEALTHCARE CHATBOT TRAINER")
    logger.info("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Healthcare Chatbot")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset file")
    parser.add_argument("--model_key", type=str, default="distilgpt2", help="Model to use")
    parser.add_argument("--output_dir", type=str, default="./models/working_chatbot", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to use")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Find dataset
        if args.dataset_path and os.path.exists(args.dataset_path):
            dataset_path = args.dataset_path
            logger.info(f"📊 Using specified dataset: {dataset_path}")
        else:
            dataset_path = find_dataset_automatically()
            if not dataset_path:
                # Use sample dataset as fallback
                dataset_path = "./data/healthcare_qa_dataset.json"
                logger.info(f"💡 Using sample dataset: {dataset_path}")
            else:
                logger.info(f"📊 Auto-found dataset: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            logger.error(f"❌ Dataset not found: {dataset_path}")
            logger.info("🔧 SOLUTIONS:")
            logger.info("1. Download ai-medical-chatbot from Kaggle")
            logger.info("2. Place CSV/JSON file in this directory")
            logger.info("3. Use --dataset_path to specify location")
            return
        
        # Step 2: Load data
        logger.info("📊 Step 2: Loading data...")
        
        if dataset_path.endswith('.csv'):
            # Load CSV data
            import pandas as pd
            df = pd.read_csv(dataset_path)
            logger.info(f"✅ Loaded CSV with columns: {list(df.columns)}")
            
            # Try to find question/answer columns
            question_col = None
            answer_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'question' in col_lower or 'query' in col_lower or 'input' in col_lower:
                    question_col = col
                elif 'answer' in col_lower or 'response' in col_lower or 'output' in col_lower:
                    answer_col = col
            
            if not question_col or not answer_col:
                # Use first two columns as fallback
                cols = list(df.columns)
                question_col = cols[0]
                answer_col = cols[1] if len(cols) > 1 else cols[0]
                logger.info(f"🔧 Using columns: {question_col} -> {answer_col}")
            
            # Convert to list of dicts
            processed_data = []
            for _, row in df.iterrows():
                if pd.notna(row[question_col]) and pd.notna(row[answer_col]):
                    processed_data.append({
                        'question': str(row[question_col]).strip(),
                        'answer': str(row[answer_col]).strip()
                    })
            
        elif dataset_path.endswith('.json'):
            # Load JSON data
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                processed_data = data
            else:
                logger.error("❌ JSON should contain a list of Q&A pairs")
                return
        
        else:
            logger.error(f"❌ Unsupported file format: {dataset_path}")
            return
        
        # Limit samples
        if len(processed_data) > args.max_samples:
            processed_data = processed_data[:args.max_samples]
            logger.info(f"📊 Limited to {len(processed_data)} samples")
        
        logger.info(f"✅ Processed {len(processed_data)} Q&A pairs")
        
        # Step 3: Prepare training data
        logger.info("🔧 Step 3: Preparing training data...")
        
        conversations = []
        for item in processed_data:
            # Create conversation format
            conversation = f"User: {item['question']} Assistant: {item['answer']}"
            conversations.append(conversation)
        
        logger.info(f"✅ Prepared {len(conversations)} conversations")
        
        # Step 4: Train model
        logger.info("🎓 Step 4: Training model...")
        
        # Import training modules
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
        import torch
        
        # Load model and tokenizer
        logger.info(f"🤖 Loading {args.model_key}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_key)
        model = AutoModelForCausalLM.from_pretrained(args.model_key)
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ Model loaded successfully")
        
        # Tokenize data
        logger.info("🔤 Tokenizing conversations...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=256
            )
        
        # Create dataset
        dataset = Dataset.from_dict({'text': conversations})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Add labels
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
        logger.info(f"✅ Tokenized {len(tokenized_dataset)} samples")
        
        # Set up training
        logger.info("⚙️ Setting up training...")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_steps=500,
            save_total_limit=1,
            logging_steps=10,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb
            warmup_steps=10
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        logger.info("✅ Training setup complete")
        
        # Train the model
        logger.info("🎓 Starting training...")
        logger.info(f"   Epochs: {args.epochs}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   Learning rate: {args.learning_rate}")
        
        trainer.train()
        
        # Save model
        final_model_dir = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        logger.info("✅ Training completed!")
        logger.info(f"💾 Model saved to: {final_model_dir}")
        
        # Step 5: Quick test
        logger.info("🧪 Step 5: Testing model...")
        
        test_input = "User: What are the symptoms of diabetes? Assistant:"
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"🤖 Test response: {response}")
        
        # Success message
        logger.info("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 40)
        logger.info(f"✅ Model: {final_model_dir}")
        logger.info(f"📊 Samples trained: {len(processed_data)}")
        logger.info(f"🎓 Epochs: {args.epochs}")
        
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("-" * 20)
        logger.info(f"🌐 Web UI: python -m src.web_interface --model_path {final_model_dir}")
        logger.info(f"💻 CLI: python -m src.chatbot {final_model_dir}")
        
        logger.info("\n🎉 Your AI healthcare assistant is ready!")
        
        return final_model_dir
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()