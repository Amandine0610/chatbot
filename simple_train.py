#!/usr/bin/env python3
"""
Simple training script that handles common issues automatically.
"""

import os
import sys
import glob
import json

# Add src to path
sys.path.append('src')

def find_dataset_file():
    """Find the user's dataset file automatically."""
    print("🔍 Looking for your dataset file...")
    
    # Common patterns for medical datasets
    patterns = [
        "*medical*.csv",
        "*medical*.json", 
        "*chatbot*.csv",
        "*chatbot*.json",
        "*.csv",
        "*.json"
    ]
    
    # Search locations
    search_dirs = [
        ".",
        "./data",
        "C:/Users/hp/Downloads" if os.name == 'nt' else "~/Downloads",
        "C:/Users/hp/Documents" if os.name == 'nt' else "~/Documents"
    ]
    
    found_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for pattern in patterns:
                files = glob.glob(os.path.join(search_dir, pattern))
                for file in files:
                    if os.path.isfile(file) and os.path.getsize(file) > 1000:  # At least 1KB
                        found_files.append(file)
    
    # Remove duplicates and sort by size (larger files first)
    unique_files = list(set(found_files))
    unique_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    return unique_files

def train_with_dataset(dataset_path):
    """Train the chatbot with the given dataset."""
    print(f"🎓 Training chatbot with: {os.path.basename(dataset_path)}")
    print("=" * 50)
    
    try:
        from kaggle_data_loader import KaggleMedicalDataLoader
        from data_preprocessing import HealthcareDataPreprocessor
        from fine_tuning import HealthcareFinetuner
        
        # Step 1: Load and process data
        print("📊 Step 1: Loading and processing data...")
        loader = KaggleMedicalDataLoader(dataset_path)
        processed_data = loader.process_data()
        
        print(f"✅ Processed {len(processed_data)} Q&A pairs")
        
        # Limit to small number for quick testing
        if len(processed_data) > 100:
            processed_data = processed_data[:100]
            print(f"📊 Limited to {len(processed_data)} samples for quick training")
        
        # Step 2: Prepare for training
        print("🔧 Step 2: Preparing training data...")
        preprocessor = HealthcareDataPreprocessor()
        dataset = preprocessor.prepare_dataset(processed_data, train_split=0.8)
        
        print(f"✅ Training: {len(dataset['train'])} samples")
        print(f"✅ Validation: {len(dataset['validation'])} samples")
        
        # Step 3: Train model
        print("🎓 Step 3: Training model...")
        print("   Using distilgpt2 for quick training...")
        
        finetuner = HealthcareFinetuner("distilgpt2", "./models/simple_healthcare_chatbot")
        
        # Use simple hyperparameters
        hyperparameters = {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 1,  # Small batch for compatibility
            "num_train_epochs": 1,  # Just 1 epoch for quick test
            "warmup_steps": 10
        }
        
        print(f"⚙️ Hyperparameters: {hyperparameters}")
        
        # Train the model
        training_results = finetuner.fine_tune(dataset, hyperparameters, use_early_stopping=False)
        
        print("\n🎉 TRAINING COMPLETED!")
        print("=" * 30)
        print(f"✅ Model saved to: {training_results['model_path']}")
        print(f"📊 Final loss: {training_results.get('eval_loss', 'N/A'):.4f}")
        
        # Step 4: Quick test
        print("\n🧪 Step 4: Quick test...")
        from chatbot import HealthcareChatbot
        
        chatbot = HealthcareChatbot(training_results['model_path'])
        test_response = chatbot.chat("What are the symptoms of diabetes?")
        
        print(f"🤖 Test Response: {test_response['response'][:100]}...")
        
        print("\n🚀 SUCCESS! Your chatbot is ready!")
        print(f"💻 Test CLI: python -m src.chatbot {training_results['model_path']}")
        print(f"🌐 Test Web: python -m src.web_interface --model_path {training_results['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def main():
    """Main function."""
    print("🏥 SIMPLE HEALTHCARE CHATBOT TRAINER")
    print("=" * 50)
    
    # Step 1: Find dataset
    datasets = find_dataset_file()
    
    if not datasets:
        print("❌ No dataset files found!")
        print("\n🔧 SOLUTIONS:")
        print("1. Download ai-medical-chatbot dataset from Kaggle")
        print("2. Place the CSV/JSON file in this directory")
        print("3. Or try with sample data:")
        print("   python simple_train.py --use_sample")
        return
    
    print(f"✅ Found {len(datasets)} potential dataset(s):")
    for i, dataset in enumerate(datasets, 1):
        size_mb = os.path.getsize(dataset) / (1024 * 1024)
        print(f"   {i}. {os.path.basename(dataset)} ({size_mb:.1f} MB)")
    
    # Use the first (largest) dataset
    selected_dataset = datasets[0]
    print(f"\n🎯 Using: {os.path.basename(selected_dataset)}")
    
    # Check if user wants to use sample data instead
    if len(sys.argv) > 1 and '--use_sample' in sys.argv:
        selected_dataset = './data/healthcare_qa_dataset.json'
        print(f"🧪 Using sample dataset: {selected_dataset}")
    
    # Train the model
    success = train_with_dataset(selected_dataset)
    
    if success:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("Your healthcare chatbot is ready to use!")
    else:
        print("\n❌ Training failed. Please check the errors above.")

if __name__ == "__main__":
    main()