#!/usr/bin/env python3
"""
Setup script for Kaggle ai-medical-chatbot dataset.
Helps users upload, validate, and prepare their Kaggle dataset.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.kaggle_data_loader import KaggleMedicalDataLoader


def print_banner():
    """Print setup banner."""
    print("=" * 70)
    print("🏥 KAGGLE AI-MEDICAL-CHATBOT DATASET SETUP")
    print("=" * 70)
    print("This script helps you set up your Kaggle dataset for training.")
    print("Supported formats: CSV, JSON, JSONL, or directory with multiple files")
    print("=" * 70)
    print()


def check_dataset_exists():
    """Check if dataset already exists in common locations."""
    common_paths = [
        "./data/kaggle_medical_dataset.json",
        "./ai-medical-chatbot.csv",
        "./ai-medical-chatbot.json",
        "./data/ai-medical-chatbot.csv",
        "./data/ai-medical-chatbot.json",
    ]
    
    existing_datasets = []
    for path in common_paths:
        if os.path.exists(path):
            existing_datasets.append(path)
    
    if existing_datasets:
        print("📁 Found existing datasets:")
        for i, path in enumerate(existing_datasets, 1):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  {i}. {path} ({size:.1f} MB)")
        print()
        
        choice = input("Use existing dataset? Enter number (or 'n' for new): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(existing_datasets):
            return existing_datasets[int(choice) - 1]
    
    return None


def get_dataset_path():
    """Get dataset path from user."""
    print("📂 Dataset Location Options:")
    print("1. Upload file to this directory")
    print("2. Provide path to existing file")
    print("3. Use Kaggle CLI (if installed)")
    print()
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        print("\n📤 Please upload your ai-medical-chatbot dataset file to this directory")
        print("Common filenames: ai-medical-chatbot.csv, medical_data.csv, etc.")
        print("After uploading, press Enter to continue...")
        input()
        
        # Look for uploaded files
        current_dir = Path(".")
        csv_files = list(current_dir.glob("*.csv"))
        json_files = list(current_dir.glob("*.json"))
        
        all_files = csv_files + json_files
        if all_files:
            print("Found files:")
            for i, file in enumerate(all_files, 1):
                size = file.stat().st_size / (1024 * 1024)  # MB
                print(f"  {i}. {file.name} ({size:.1f} MB)")
            
            file_choice = input("Select file number: ").strip()
            if file_choice.isdigit() and 1 <= int(file_choice) <= len(all_files):
                return str(all_files[int(file_choice) - 1])
        
        print("❌ No suitable files found. Please upload your dataset and try again.")
        return None
    
    elif choice == "2":
        path = input("Enter full path to your dataset file: ").strip()
        if os.path.exists(path):
            return path
        else:
            print(f"❌ File not found: {path}")
            return None
    
    elif choice == "3":
        print("\n📦 Kaggle CLI Download:")
        print("If you have Kaggle CLI installed, you can download with:")
        print("  kaggle datasets download -d <dataset-name>")
        print("Then unzip and provide the path to the CSV/JSON file.")
        return None
    
    else:
        print("❌ Invalid choice")
        return None


def validate_dataset(dataset_path):
    """Validate and preview the dataset."""
    try:
        print(f"\n🔍 Validating dataset: {dataset_path}")
        
        # Load and process data
        loader = KaggleMedicalDataLoader(dataset_path)
        
        # Detect format
        format_type = loader.detect_format()
        print(f"✅ Detected format: {format_type}")
        
        # Load data
        raw_data = loader.load_data()
        print(f"✅ Successfully loaded data")
        
        # Process data
        processed_data = loader.process_data()
        print(f"✅ Processed {len(processed_data)} Q&A pairs")
        
        # Get statistics
        stats = loader.get_data_statistics()
        print("\n📊 Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} categories")
                # Show top categories
                top_categories = list(value.items())[:3]
                for cat, count in top_categories:
                    print(f"    - {cat}: {count}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Preview data
        print("\n👀 Data Preview:")
        preview = loader.preview_data(3)
        for i, item in enumerate(preview, 1):
            print(f"\nSample {i}:")
            print(f"  Q: {item['question'][:100]}{'...' if len(item['question']) > 100 else ''}")
            print(f"  A: {item['answer'][:100]}{'...' if len(item['answer']) > 100 else ''}")
            if 'category' in item:
                print(f"  Category: {item['category']}")
        
        return loader, processed_data
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return None, None


def save_processed_dataset(loader, processed_data):
    """Save processed dataset for training."""
    output_path = "./data/kaggle_medical_dataset.json"
    
    try:
        loader.save_processed_data(output_path)
        print(f"\n💾 Processed dataset saved to: {output_path}")
        print(f"📊 Total samples: {len(processed_data)}")
        return output_path
    except Exception as e:
        print(f"❌ Error saving processed dataset: {e}")
        return None


def show_next_steps(processed_dataset_path):
    """Show next steps for training."""
    print("\n🚀 Next Steps:")
    print("=" * 40)
    print("1. Quick Training (for testing):")
    print(f"   python train_kaggle_chatbot.py \\")
    print(f"     --kaggle_dataset_path <your-original-file> \\")
    print(f"     --epochs 1 \\")
    print(f"     --max_samples 100 \\")
    print(f"     --skip_hyperparameter_search")
    print()
    print("2. Full Training:")
    print(f"   python train_kaggle_chatbot.py \\")
    print(f"     --kaggle_dataset_path <your-original-file> \\")
    print(f"     --epochs 3")
    print()
    print("3. Training with specific model:")
    print(f"   python train_kaggle_chatbot.py \\")
    print(f"     --kaggle_dataset_path <your-original-file> \\")
    print(f"     --model_key distilgpt2 \\")
    print(f"     --batch_size 2")
    print()
    print("4. After training, test your chatbot:")
    print("   python -m src.chatbot ./models/kaggle_healthcare_chatbot/final_model")
    print()
    print("5. Launch web interface:")
    print("   python -m src.web_interface --model_path ./models/kaggle_healthcare_chatbot/final_model")


def main():
    """Main setup function."""
    print_banner()
    
    # Check for existing datasets
    existing_path = check_dataset_exists()
    if existing_path:
        dataset_path = existing_path
        print(f"✅ Using existing dataset: {dataset_path}")
    else:
        # Get new dataset path
        dataset_path = get_dataset_path()
        if not dataset_path:
            print("❌ Setup cancelled. Please provide a valid dataset path.")
            return
    
    # Validate dataset
    loader, processed_data = validate_dataset(dataset_path)
    if not loader or not processed_data:
        print("❌ Dataset validation failed. Please check your file format and content.")
        return
    
    # Check if we should proceed
    proceed = input(f"\n✅ Dataset looks good! Proceed with setup? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Setup cancelled.")
        return
    
    # Save processed dataset
    processed_path = save_processed_dataset(loader, processed_data)
    if not processed_path:
        print("❌ Failed to save processed dataset.")
        return
    
    # Show next steps
    show_next_steps(processed_path)
    
    print("\n🎉 Setup completed successfully!")
    print("Your Kaggle dataset is ready for training.")
    print(f"Original dataset: {dataset_path}")
    print(f"Processed dataset: {processed_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("Please check your dataset and try again.")