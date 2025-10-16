#!/usr/bin/env python3
"""
Helper script to find your Kaggle dataset file.
"""

import os
import glob
from pathlib import Path

def find_datasets():
    """Find potential dataset files."""
    print("🔍 SEARCHING FOR YOUR KAGGLE DATASET")
    print("=" * 50)
    
    # Search patterns for common dataset files
    search_patterns = [
        "*.csv",
        "*.json", 
        "*.jsonl",
        "*medical*.csv",
        "*medical*.json",
        "*chatbot*.csv",
        "*chatbot*.json",
        "*ai-medical*.csv",
        "*ai-medical*.json"
    ]
    
    found_files = []
    
    # Search in current directory and subdirectories
    search_dirs = [
        ".",
        "./data",
        "../",
        "C:/Users/hp/Documents",
        "C:/Users/hp/Downloads"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"\n📂 Searching in: {search_dir}")
            
            for pattern in search_patterns:
                try:
                    files = glob.glob(os.path.join(search_dir, pattern))
                    for file in files:
                        if os.path.isfile(file):
                            size_mb = os.path.getsize(file) / (1024 * 1024)
                            found_files.append({
                                'path': os.path.abspath(file),
                                'name': os.path.basename(file),
                                'size_mb': size_mb,
                                'dir': search_dir
                            })
                except:
                    continue
    
    # Remove duplicates
    unique_files = []
    seen_paths = set()
    for file_info in found_files:
        if file_info['path'] not in seen_paths:
            unique_files.append(file_info)
            seen_paths.add(file_info['path'])
    
    return unique_files

def main():
    """Main function to find and suggest dataset files."""
    found_files = find_datasets()
    
    if found_files:
        print(f"\n✅ FOUND {len(found_files)} POTENTIAL DATASET FILE(S):")
        print("-" * 60)
        
        for i, file_info in enumerate(found_files, 1):
            print(f"{i:2d}. {file_info['name']}")
            print(f"    📁 Path: {file_info['path']}")
            print(f"    📊 Size: {file_info['size_mb']:.1f} MB")
            print(f"    📂 Directory: {file_info['dir']}")
            
            # Check if it looks like a medical dataset
            name_lower = file_info['name'].lower()
            if any(keyword in name_lower for keyword in ['medical', 'health', 'chatbot', 'qa']):
                print(f"    🏥 Likely medical dataset!")
            
            print()
        
        print("🎯 SUGGESTED COMMANDS:")
        print("-" * 25)
        
        # Show commands for the most likely candidates
        likely_candidates = [f for f in found_files if any(keyword in f['name'].lower() 
                           for keyword in ['medical', 'health', 'chatbot', 'ai-medical'])]
        
        if not likely_candidates:
            likely_candidates = found_files[:3]  # Show first 3 if no obvious candidates
        
        for i, file_info in enumerate(likely_candidates[:3], 1):
            print(f"\n{i}. Using {file_info['name']}:")
            
            # Use the fixed script
            cmd = f'python train_kaggle_chatbot_fixed.py --kaggle_dataset_path "{file_info["path"]}" --epochs 1 --max_samples 100 --model_key distilgpt2 --skip_hyperparameter_search'
            print(f"   {cmd}")
    
    else:
        print("\n❌ NO DATASET FILES FOUND")
        print("-" * 30)
        print("🔧 SOLUTIONS:")
        print("1. Download your ai-medical-chatbot dataset from Kaggle")
        print("2. Place it in this directory (C:/Users/hp/Documents/chatbot)")
        print("3. Make sure the file has .csv or .json extension")
        print("4. Check your Downloads folder")
        
        print("\n📤 COMMON LOCATIONS TO CHECK:")
        print("• C:/Users/hp/Downloads/")
        print("• C:/Users/hp/Documents/")
        print("• Desktop")
        print("• Where you saved the Kaggle download")
        
        print("\n💡 ALTERNATIVE: Use the sample dataset")
        print("python train_kaggle_chatbot_fixed.py --kaggle_dataset_path ./data/healthcare_qa_dataset.json --epochs 1 --max_samples 20 --model_key distilgpt2 --skip_hyperparameter_search")

if __name__ == "__main__":
    main()