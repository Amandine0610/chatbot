#!/usr/bin/env python3
"""
Simple CSV to JSON converter for medical datasets
"""

import pandas as pd
import json
import sys
import os

def convert_csv_to_json(csv_path, output_path=None):
    """Convert CSV to JSON format expected by the chatbot."""
    print(f"📊 Converting {csv_path} to JSON...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    
    # Try to identify question and answer columns
    question_col = None
    answer_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['question', 'query', 'input', 'prompt']):
            question_col = col
        elif any(word in col_lower for word in ['answer', 'response', 'output', 'reply']):
            answer_col = col
    
    if not question_col or not answer_col:
        print("🔧 Could not auto-detect columns. Available columns:")
        for i, col in enumerate(df.columns):
            print(f"   {i}: {col}")
        
        # Use first two columns as fallback
        question_col = df.columns[0]
        answer_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        print(f"🎯 Using: {question_col} -> {answer_col}")
    else:
        print(f"🎯 Auto-detected: {question_col} -> {answer_col}")
    
    # Convert to JSON format
    json_data = []
    for _, row in df.iterrows():
        if pd.notna(row[question_col]) and pd.notna(row[answer_col]):
            json_data.append({
                'question': str(row[question_col]).strip(),
                'answer': str(row[answer_col]).strip()
            })
    
    print(f"✅ Converted {len(json_data)} valid Q&A pairs")
    
    # Save JSON
    if output_path is None:
        output_path = csv_path.replace('.csv', '_converted.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to: {output_path}")
    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_csv_to_json.py <csv_file> [output_file]")
        print("\nExample:")
        print("  python convert_csv_to_json.py ai-medical-chatbot.csv")
        return
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    converted_path = convert_csv_to_json(csv_path, output_path)
    
    print(f"\n🎉 Conversion complete!")
    print(f"📊 Original: {csv_path}")
    print(f"📄 Converted: {converted_path}")
    print(f"\n🚀 Now you can train with:")
    print(f"python train_working.py --dataset_path {converted_path}")

if __name__ == "__main__":
    main()