"""
Kaggle dataset loader for ai-medical-chatbot dataset.
Handles various formats and preprocessing for medical chatbot training.
"""

import pandas as pd
import json
import os
import logging
from typing import List, Dict, Any, Optional, Union
import re
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleMedicalDataLoader:
    """
    Loader for Kaggle ai-medical-chatbot dataset.
    Supports CSV, JSON, and other common formats.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the Kaggle dataset loader.
        
        Args:
            dataset_path: Path to the Kaggle dataset file or directory
        """
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.processed_data = None
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    def detect_format(self) -> str:
        """
        Detect the format of the dataset.
        
        Returns:
            Format string: 'csv', 'json', 'jsonl', 'txt', or 'directory'
        """
        if self.dataset_path.is_dir():
            return 'directory'
        
        suffix = self.dataset_path.suffix.lower()
        if suffix == '.csv':
            return 'csv'
        elif suffix == '.json':
            return 'json'
        elif suffix == '.jsonl':
            return 'jsonl'
        elif suffix == '.txt':
            return 'txt'
        else:
            return 'unknown'
    
    def load_csv_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with the loaded data
        """
        logger.info(f"Loading CSV data from {self.dataset_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(self.dataset_path, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any common encoding")
            
            logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_json_data(self) -> Union[List[Dict], Dict]:
        """
        Load data from JSON file.
        
        Returns:
            Loaded JSON data
        """
        logger.info(f"Loading JSON data from {self.dataset_path}")
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON data with {len(data) if isinstance(data, list) else 'nested'} entries")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def load_jsonl_data(self) -> List[Dict]:
        """
        Load data from JSONL file (JSON Lines).
        
        Returns:
            List of JSON objects
        """
        logger.info(f"Loading JSONL data from {self.dataset_path}")
        
        try:
            data = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            logger.info(f"Loaded {len(data)} JSONL entries")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSONL: {e}")
            raise
    
    def load_directory_data(self) -> List[Dict]:
        """
        Load data from directory containing multiple files.
        
        Returns:
            Combined data from all files
        """
        logger.info(f"Loading data from directory {self.dataset_path}")
        
        all_data = []
        
        for file_path in self.dataset_path.iterdir():
            if file_path.is_file():
                try:
                    loader = KaggleMedicalDataLoader(str(file_path))
                    file_data = loader.load_data()
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                    else:
                        all_data.append(file_data)
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_data)} total entries from directory")
        return all_data
    
    def load_data(self) -> Union[pd.DataFrame, List[Dict], Dict]:
        """
        Load data based on detected format.
        
        Returns:
            Loaded data in appropriate format
        """
        format_type = self.detect_format()
        logger.info(f"Detected format: {format_type}")
        
        if format_type == 'csv':
            self.data = self.load_csv_data()
        elif format_type == 'json':
            self.data = self.load_json_data()
        elif format_type == 'jsonl':
            self.data = self.load_jsonl_data()
        elif format_type == 'directory':
            self.data = self.load_directory_data()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return self.data
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to common formats.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        # Common column name mappings
        column_mappings = {
            # Question variations
            'question': 'question',
            'query': 'question', 
            'input': 'question',
            'user_input': 'question',
            'patient_question': 'question',
            'medical_question': 'question',
            'q': 'question',
            
            # Answer variations
            'answer': 'answer',
            'response': 'answer',
            'reply': 'answer',
            'output': 'answer',
            'bot_response': 'answer',
            'medical_answer': 'answer',
            'doctor_response': 'answer',
            'a': 'answer',
            
            # Category variations
            'category': 'category',
            'topic': 'category',
            'medical_category': 'category',
            'specialty': 'category',
            'department': 'category'
        }
        
        # Normalize column names (lowercase, remove spaces/underscores)
        normalized_columns = {}
        for col in df.columns:
            normalized = col.lower().replace(' ', '_').replace('-', '_')
            if normalized in column_mappings:
                normalized_columns[col] = column_mappings[normalized]
            else:
                normalized_columns[col] = col
        
        df = df.rename(columns=normalized_columns)
        
        logger.info(f"Standardized columns: {list(df.columns)}")
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'/°%]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"['']", "'", text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process_dataframe(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Process DataFrame into standard Q&A format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of question-answer dictionaries
        """
        logger.info("Processing DataFrame to Q&A format")
        
        # Standardize columns
        df = self.standardize_columns(df)
        
        # Check for required columns
        if 'question' not in df.columns and 'answer' not in df.columns:
            # Try to infer from available columns
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if len(text_columns) >= 2:
                logger.warning(f"No 'question' and 'answer' columns found. Using first two text columns: {text_columns[:2]}")
                df = df.rename(columns={
                    text_columns[0]: 'question',
                    text_columns[1]: 'answer'
                })
            else:
                raise ValueError("Could not identify question and answer columns")
        
        # Process the data
        processed_data = []
        
        for idx, row in df.iterrows():
            question = self.clean_text(str(row.get('question', '')))
            answer = self.clean_text(str(row.get('answer', '')))
            
            # Skip empty entries
            if not question or not answer:
                continue
            
            # Skip very short entries
            if len(question) < 5 or len(answer) < 10:
                continue
            
            entry = {
                'question': question,
                'answer': answer
            }
            
            # Add category if available
            if 'category' in row and pd.notna(row['category']):
                entry['category'] = str(row['category'])
            
            processed_data.append(entry)
        
        logger.info(f"Processed {len(processed_data)} valid Q&A pairs from {len(df)} rows")
        return processed_data
    
    def process_json_list(self, data: List[Dict]) -> List[Dict[str, str]]:
        """
        Process list of JSON objects into standard Q&A format.
        
        Args:
            data: List of dictionaries
            
        Returns:
            List of question-answer dictionaries
        """
        logger.info("Processing JSON list to Q&A format")
        
        processed_data = []
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            # Try different key combinations
            question_keys = ['question', 'query', 'input', 'user_input', 'q']
            answer_keys = ['answer', 'response', 'reply', 'output', 'a']
            
            question = None
            answer = None
            
            for key in question_keys:
                if key in item:
                    question = self.clean_text(str(item[key]))
                    break
            
            for key in answer_keys:
                if key in item:
                    answer = self.clean_text(str(item[key]))
                    break
            
            if question and answer and len(question) >= 5 and len(answer) >= 10:
                entry = {
                    'question': question,
                    'answer': answer
                }
                
                # Add category if available
                category_keys = ['category', 'topic', 'specialty']
                for key in category_keys:
                    if key in item:
                        entry['category'] = str(item[key])
                        break
                
                processed_data.append(entry)
        
        logger.info(f"Processed {len(processed_data)} valid Q&A pairs from {len(data)} JSON objects")
        return processed_data
    
    def process_data(self) -> List[Dict[str, str]]:
        """
        Process loaded data into standard format.
        
        Returns:
            List of question-answer dictionaries
        """
        if self.data is None:
            self.load_data()
        
        if isinstance(self.data, pd.DataFrame):
            self.processed_data = self.process_dataframe(self.data)
        elif isinstance(self.data, list):
            self.processed_data = self.process_json_list(self.data)
        else:
            raise ValueError("Unsupported data format for processing")
        
        return self.processed_data
    
    def save_processed_data(self, output_path: str):
        """
        Save processed data to JSON file.
        
        Args:
            output_path: Path to save the processed data
        """
        if self.processed_data is None:
            self.process_data()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.processed_data)} processed entries to {output_path}")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed data.
        
        Returns:
            Dictionary with data statistics
        """
        if self.processed_data is None:
            self.process_data()
        
        questions = [item['question'] for item in self.processed_data]
        answers = [item['answer'] for item in self.processed_data]
        
        stats = {
            'total_samples': len(self.processed_data),
            'avg_question_length': sum(len(q.split()) for q in questions) / len(questions),
            'avg_answer_length': sum(len(a.split()) for a in answers) / len(answers),
            'max_question_length': max(len(q.split()) for q in questions),
            'max_answer_length': max(len(a.split()) for a in answers),
            'min_question_length': min(len(q.split()) for q in questions),
            'min_answer_length': min(len(a.split()) for a in answers),
        }
        
        # Category statistics if available
        categories = [item.get('category') for item in self.processed_data if item.get('category')]
        if categories:
            from collections import Counter
            category_counts = Counter(categories)
            stats['categories'] = dict(category_counts.most_common(10))
            stats['num_categories'] = len(category_counts)
        
        return stats
    
    def preview_data(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Preview first n samples of processed data.
        
        Args:
            n: Number of samples to preview
            
        Returns:
            List of sample entries
        """
        if self.processed_data is None:
            self.process_data()
        
        return self.processed_data[:n]


def main():
    """
    Main function to demonstrate Kaggle data loading.
    """
    print("Kaggle Medical Dataset Loader")
    print("=" * 40)
    
    # Example usage
    dataset_path = input("Enter path to your Kaggle ai-medical-chatbot dataset: ").strip()
    
    if not dataset_path:
        print("No path provided. Please upload your dataset and run again.")
        return
    
    try:
        # Load and process data
        loader = KaggleMedicalDataLoader(dataset_path)
        processed_data = loader.process_data()
        
        # Show statistics
        stats = loader.get_data_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in list(value.items())[:5]:  # Show top 5
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Preview data
        print("\nData Preview:")
        preview = loader.preview_data(3)
        for i, item in enumerate(preview, 1):
            print(f"\nSample {i}:")
            print(f"  Q: {item['question'][:100]}...")
            print(f"  A: {item['answer'][:100]}...")
            if 'category' in item:
                print(f"  Category: {item['category']}")
        
        # Save processed data
        output_path = "./data/kaggle_medical_dataset.json"
        loader.save_processed_data(output_path)
        print(f"\nProcessed data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your dataset path and format.")


if __name__ == "__main__":
    main()