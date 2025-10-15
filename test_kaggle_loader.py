#!/usr/bin/env python3
"""
Test script for Kaggle dataset loader.
Creates sample data and tests the loading pipeline.
"""

import pandas as pd
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.kaggle_data_loader import KaggleMedicalDataLoader


def create_sample_csv():
    """Create a sample CSV file in Kaggle format."""
    sample_data = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, extreme fatigue, blurred vision, slow-healing cuts and bruises, and tingling or numbness in hands or feet.",
            "category": "endocrine"
        },
        {
            "question": "How is hypertension treated?",
            "answer": "Hypertension treatment typically includes lifestyle changes like regular exercise, healthy diet, reduced sodium intake, and weight management. Medications may include ACE inhibitors, diuretics, or beta-blockers as prescribed by a doctor.",
            "category": "cardiovascular"
        },
        {
            "question": "What causes chest pain?",
            "answer": "Chest pain can be caused by various conditions including heart problems (angina, heart attack), lung issues (pneumonia, pleurisy), digestive problems (acid reflux), or muscle strain. Seek immediate medical attention for severe chest pain.",
            "category": "cardiovascular"
        },
        {
            "question": "How to prevent food poisoning?",
            "answer": "Prevent food poisoning by washing hands frequently, cooking foods to safe temperatures, refrigerating perishables promptly, avoiding cross-contamination, washing fruits and vegetables, and checking expiration dates.",
            "category": "preventive"
        },
        {
            "question": "What are the side effects of aspirin?",
            "answer": "Common side effects of aspirin include stomach irritation, increased bleeding risk, nausea, and heartburn. Serious side effects can include gastrointestinal bleeding, allergic reactions, and kidney problems. Always consult your doctor about medication side effects.",
            "category": "medication"
        }
    ]
    
    df = pd.DataFrame(sample_data)
    csv_path = "./test_kaggle_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Created sample CSV: {csv_path}")
    return csv_path


def create_sample_json():
    """Create a sample JSON file in Kaggle format."""
    sample_data = [
        {
            "query": "What is pneumonia?",
            "response": "Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing. It can be caused by bacteria, viruses, or fungi.",
            "medical_category": "respiratory"
        },
        {
            "patient_question": "How long does it take to recover from surgery?",
            "doctor_response": "Recovery time varies depending on the type of surgery, your overall health, and age. Minor surgeries may require days to weeks, while major surgeries can take months. Follow your surgeon's post-operative instructions for optimal recovery.",
            "specialty": "surgery"
        },
        {
            "input": "What foods should diabetics avoid?",
            "output": "Diabetics should limit refined sugars, white bread, sugary drinks, processed foods, and foods high in saturated fats. Focus on whole grains, lean proteins, vegetables, and foods with low glycemic index. Always consult with a nutritionist for personalized advice.",
            "topic": "nutrition"
        }
    ]
    
    json_path = "./test_kaggle_dataset.json"
    with open(json_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"✅ Created sample JSON: {json_path}")
    return json_path


def test_loader(file_path):
    """Test the Kaggle dataset loader."""
    print(f"\n🧪 Testing loader with: {file_path}")
    print("-" * 50)
    
    try:
        # Initialize loader
        loader = KaggleMedicalDataLoader(file_path)
        
        # Detect format
        format_type = loader.detect_format()
        print(f"📁 Detected format: {format_type}")
        
        # Load data
        raw_data = loader.load_data()
        print(f"📊 Loaded data type: {type(raw_data)}")
        
        if isinstance(raw_data, pd.DataFrame):
            print(f"📋 DataFrame shape: {raw_data.shape}")
            print(f"📋 Columns: {list(raw_data.columns)}")
        elif isinstance(raw_data, list):
            print(f"📋 List length: {len(raw_data)}")
            if raw_data:
                print(f"📋 Sample keys: {list(raw_data[0].keys())}")
        
        # Process data
        processed_data = loader.process_data()
        print(f"✅ Processed {len(processed_data)} Q&A pairs")
        
        # Get statistics
        stats = loader.get_data_statistics()
        print("\n📊 Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Preview data
        print("\n👀 Preview:")
        preview = loader.preview_data(2)
        for i, item in enumerate(preview, 1):
            print(f"\nSample {i}:")
            print(f"  Q: {item['question']}")
            print(f"  A: {item['answer'][:80]}...")
            if 'category' in item:
                print(f"  Category: {item['category']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing loader: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 KAGGLE DATASET LOADER TEST")
    print("=" * 50)
    
    # Create sample datasets
    csv_path = create_sample_csv()
    json_path = create_sample_json()
    
    # Test CSV loader
    print("\n" + "=" * 50)
    print("TESTING CSV LOADER")
    print("=" * 50)
    csv_success = test_loader(csv_path)
    
    # Test JSON loader
    print("\n" + "=" * 50)
    print("TESTING JSON LOADER")
    print("=" * 50)
    json_success = test_loader(json_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"CSV Loader: {'✅ PASSED' if csv_success else '❌ FAILED'}")
    print(f"JSON Loader: {'✅ PASSED' if json_success else '❌ FAILED'}")
    
    if csv_success and json_success:
        print("\n🎉 All tests passed! The Kaggle dataset loader is ready.")
        print("\nNext steps:")
        print("1. Upload your ai-medical-chatbot dataset")
        print("2. Run: python setup_kaggle_dataset.py")
        print("3. Or directly: python train_kaggle_chatbot.py --kaggle_dataset_path your-file.csv")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    # Cleanup
    try:
        os.remove(csv_path)
        os.remove(json_path)
        print(f"\n🧹 Cleaned up test files")
    except:
        pass


if __name__ == "__main__":
    main()