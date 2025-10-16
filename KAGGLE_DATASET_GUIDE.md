# 🏥 Kaggle AI-Medical-Chatbot Dataset Guide

This guide helps you use your Kaggle "ai-medical-chatbot" dataset with the healthcare chatbot training pipeline.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
python setup_kaggle_dataset.py
```
This interactive script will:
- Help you locate/upload your dataset
- Validate the data format
- Process and convert to training format
- Show you the next steps

### Option 2: Manual Setup
```bash
# 1. Place your dataset in the project directory
# 2. Run the training script directly
python train_kaggle_chatbot.py --kaggle_dataset_path your-dataset-file.csv
```

## 📁 Supported Dataset Formats

The system automatically detects and handles:

### CSV Files
```csv
question,answer,category
"What are diabetes symptoms?","Common symptoms include...",endocrine
"How to treat hypertension?","Treatment options include...",cardiovascular
```

### JSON Files
```json
[
  {
    "question": "What are diabetes symptoms?",
    "answer": "Common symptoms include...",
    "category": "endocrine"
  }
]
```

### JSONL Files (JSON Lines)
```jsonl
{"question": "What are diabetes symptoms?", "answer": "Common symptoms include...", "category": "endocrine"}
{"question": "How to treat hypertension?", "answer": "Treatment options include...", "category": "cardiovascular"}
```

## 🔧 Column Name Mapping

The system automatically maps various column names:

### Question Columns
- `question`, `query`, `input`, `user_input`
- `patient_question`, `medical_question`, `q`

### Answer Columns  
- `answer`, `response`, `reply`, `output`
- `bot_response`, `medical_answer`, `doctor_response`, `a`

### Category Columns (Optional)
- `category`, `topic`, `medical_category`
- `specialty`, `department`

## 🎓 Training Options

### Quick Test Training (1-2 minutes)
```bash
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --epochs 1 \
  --max_samples 100 \
  --model_key distilgpt2 \
  --skip_hyperparameter_search
```

### Full Training (10-30 minutes)
```bash
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --epochs 3 \
  --model_key dialogpt-medium
```

### Large Dataset Training
```bash
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --epochs 2 \
  --batch_size 2 \
  --max_samples 5000 \
  --model_key dialogpt-medium
```

## ⚙️ Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kaggle_dataset_path` | Required | Path to your Kaggle dataset |
| `--model_key` | dialogpt-medium | Model to use (see options below) |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 5e-5 | Learning rate |
| `--max_samples` | None | Limit dataset size (for testing) |
| `--train_split` | 0.8 | Fraction for training vs validation |

### Available Models
- `distilgpt2` - Fastest, smallest (recommended for testing)
- `gpt2` - Good balance of speed and quality
- `gpt2-medium` - Better quality, slower training
- `dialogpt-small` - Optimized for conversation
- `dialogpt-medium` - Best quality (recommended for production)
- `t5-small` - Encoder-decoder architecture

## 📊 Dataset Requirements

### Minimum Requirements
- At least 50 question-answer pairs
- Questions: minimum 5 characters
- Answers: minimum 10 characters
- Valid text format (no excessive special characters)

### Recommended
- 1000+ question-answer pairs for good performance
- Diverse medical topics and question types
- Clear, well-formatted answers
- Consistent formatting

## 🔍 Data Validation

The system automatically:
- ✅ Detects file format
- ✅ Maps column names
- ✅ Cleans text (removes HTML, normalizes spacing)
- ✅ Filters out invalid entries
- ✅ Provides statistics and preview

### Common Issues and Solutions

#### Issue: "No question/answer columns found"
**Solution**: Ensure your CSV has columns that can be mapped to questions and answers. Common names work automatically.

#### Issue: "Too few valid samples"
**Solution**: Check that your questions and answers meet minimum length requirements.

#### Issue: "Unicode decode error"
**Solution**: The system tries multiple encodings automatically. If it fails, save your CSV as UTF-8.

## 📈 Training Process

1. **Data Loading**: Load and detect format
2. **Data Processing**: Clean and standardize
3. **Data Splitting**: Train/validation split
4. **Model Loading**: Load pre-trained model
5. **Fine-tuning**: Train on your data
6. **Evaluation**: Test performance
7. **Model Saving**: Save trained model

## 📊 Expected Performance

### Small Dataset (100-500 samples)
- Training time: 5-15 minutes
- BLEU score: 0.15-0.25
- Good for basic medical Q&A

### Medium Dataset (500-2000 samples)  
- Training time: 15-45 minutes
- BLEU score: 0.20-0.35
- Good conversational quality

### Large Dataset (2000+ samples)
- Training time: 30-90 minutes  
- BLEU score: 0.25-0.40
- High-quality responses

## 🚀 After Training

### Test Your Chatbot
```bash
# Command line interface
python -m src.chatbot ./models/kaggle_healthcare_chatbot/final_model

# Web interface
python -m src.web_interface --model_path ./models/kaggle_healthcare_chatbot/final_model
```

### Evaluation Only
```bash
python train_kaggle_chatbot.py \
  --evaluate_only \
  --model_path_for_eval ./models/kaggle_healthcare_chatbot/final_model
```

## 🔧 Troubleshooting

### Memory Issues
```bash
# Reduce batch size
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --batch_size 1 \
  --model_key distilgpt2
```

### Speed Up Training
```bash
# Use smaller model and fewer samples
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --model_key distilgpt2 \
  --max_samples 1000 \
  --epochs 2
```

### Improve Quality
```bash
# Use larger model and more epochs
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --model_key dialogpt-medium \
  --epochs 4 \
  --learning_rate 3e-5
```

## 📁 Output Files

After training, you'll find:
```
models/kaggle_healthcare_chatbot/
├── final_model/                    # Trained model files
├── kaggle_training_summary.json    # Training summary
├── final_evaluation_results.json   # Evaluation metrics
└── qualitative_evaluation.json     # Sample Q&A results
```

## 🎯 Example Workflows

### Research/Testing Workflow
```bash
# 1. Quick setup and validation
python setup_kaggle_dataset.py

# 2. Fast training for testing
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --epochs 1 \
  --max_samples 200 \
  --model_key distilgpt2

# 3. Test the chatbot
python -m src.chatbot ./models/kaggle_healthcare_chatbot/final_model
```

### Production Workflow
```bash
# 1. Setup and validate full dataset
python setup_kaggle_dataset.py

# 2. Full training with hyperparameter search
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --model_key dialogpt-medium \
  --epochs 3

# 3. Evaluate thoroughly
python train_kaggle_chatbot.py \
  --evaluate_only \
  --eval_samples 500

# 4. Deploy web interface
python -m src.web_interface \
  --model_path ./models/kaggle_healthcare_chatbot/final_model \
  --share
```

## 📚 Additional Resources

- **Main Documentation**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **Project Summary**: `PROJECT_SUMMARY.md`
- **Training Logs**: `kaggle_training.log`

## 🆘 Getting Help

1. **Check the logs**: `kaggle_training.log`
2. **Validate your dataset**: `python setup_kaggle_dataset.py`
3. **Try with smaller dataset**: Use `--max_samples 100`
4. **Use lighter model**: Try `--model_key distilgpt2`

---

**Ready to train your healthcare chatbot with Kaggle data! 🏥🤖**