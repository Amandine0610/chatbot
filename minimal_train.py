#!/usr/bin/env python3
"""
Minimal training script that avoids complex features and just works.
"""

import os
import sys
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Add src to path
sys.path.append('src')

def minimal_train(dataset_path, output_dir="./models/minimal_chatbot"):
    """Minimal training function that just works."""
    print("🏥 MINIMAL HEALTHCARE CHATBOT TRAINING")
    print("=" * 50)
    
    # Step 1: Load data
    print("📊 Loading dataset...")
    
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    else:
        print("❌ This minimal script only supports JSON files")
        print("💡 Use the sample dataset: ./data/healthcare_qa_dataset.json")
        return False
    
    print(f"✅ Loaded {len(data)} samples")
    
    # Limit to small number for quick training
    if len(data) > 50:
        data = data[:50]
        print(f"📊 Limited to {len(data)} samples for quick training")
    
    # Step 2: Prepare conversations
    print("🔧 Preparing conversations...")
    
    conversations = []
    for item in data:
        conversation = f"User: {item['question']} Assistant: {item['answer']}"
        conversations.append(conversation)
    
    print(f"✅ Prepared {len(conversations)} conversations")
    
    # Step 3: Load model and tokenizer
    print("🤖 Loading model...")
    
    model_name = "distilgpt2"  # Use smallest model for reliability
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Loaded {model_name}")
    
    # Step 4: Tokenize data
    print("🔤 Tokenizing data...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=256  # Shorter for quick training
        )
    
    # Create dataset
    dataset = Dataset.from_dict({'text': conversations})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Add labels for language modeling
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
    
    print(f"✅ Tokenized {len(tokenized_dataset)} samples")
    
    # Step 5: Set up training
    print("⚙️ Setting up training...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Just 1 epoch for quick test
        per_device_train_batch_size=1,  # Small batch size
        save_steps=1000,
        save_total_limit=1,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None  # Disable wandb
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
    
    print("✅ Training setup complete")
    
    # Step 6: Train
    print("🎓 Starting training...")
    print("   (This may take a few minutes)")
    
    try:
        trainer.train()
        print("✅ Training completed!")
        
        # Save model
        final_model_dir = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        print(f"💾 Model saved to: {final_model_dir}")
        
        # Step 7: Quick test
        print("🧪 Quick test...")
        
        # Simple generation test
        test_input = "User: What are diabetes symptoms? Assistant:"
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
        print(f"🤖 Test response: {response}")
        
        print("\n🎉 SUCCESS!")
        print(f"Your model is ready at: {final_model_dir}")
        
        return final_model_dir
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def main():
    """Main function."""
    print("🏥 MINIMAL HEALTHCARE CHATBOT TRAINER")
    print("=" * 50)
    
    # Check for dataset argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        dataset_path = sys.argv[1]
    else:
        # Use sample dataset
        dataset_path = "./data/healthcare_qa_dataset.json"
        print("💡 No dataset specified, using sample data")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("\n🔧 SOLUTIONS:")
        print("1. Use sample data: python minimal_train.py")
        print("2. Specify your file: python minimal_train.py your-dataset.json")
        print("3. Convert CSV to JSON first")
        return
    
    print(f"📊 Using dataset: {dataset_path}")
    
    # Train the model
    model_path = minimal_train(dataset_path)
    
    if model_path:
        print("\n🚀 NEXT STEPS:")
        print("-" * 20)
        print(f"🌐 Web interface: python -m src.web_interface --model_path {model_path}")
        print(f"💻 CLI chat: python -m src.chatbot {model_path}")
        print("\n🎉 Your AI healthcare assistant is ready!")
    else:
        print("\n❌ Training failed. Please check the errors above.")

if __name__ == "__main__":
    main()