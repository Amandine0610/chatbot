# 🎯 Instructions for Your Kaggle AI-Medical-Chatbot Dataset

## 📋 What You Need to Do

### Step 1: Upload Your Dataset
Place your Kaggle "ai-medical-chatbot" dataset file in this project directory. Common formats:
- `ai-medical-chatbot.csv`
- `medical_data.csv` 
- `healthcare_qa.json`
- Any CSV/JSON file with medical Q&A data

### Step 2: Choose Your Approach

#### Option A: Interactive Setup (Recommended for First-Time Users)
```bash
python setup_kaggle_dataset.py
```
This will:
- Help you locate your dataset
- Validate the format and content
- Show you statistics and preview
- Process the data for training
- Guide you through next steps

#### Option B: Direct Training (For Experienced Users)
```bash
python train_kaggle_chatbot.py --kaggle_dataset_path your-dataset-file.csv
```

## 🚀 Quick Training Examples

### Test Training (2-3 minutes)
```bash
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --epochs 1 \
  --max_samples 100 \
  --model_key distilgpt2 \
  --skip_hyperparameter_search
```

### Full Training (15-30 minutes)
```bash
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --model_key dialogpt-medium \
  --epochs 3
```

### Large Dataset Training
```bash
python train_kaggle_chatbot.py \
  --kaggle_dataset_path your-dataset.csv \
  --model_key dialogpt-medium \
  --epochs 2 \
  --batch_size 2 \
  --max_samples 5000
```

## 📊 What the System Expects

### CSV Format Example
```csv
question,answer,category
"What are diabetes symptoms?","Symptoms include frequent urination...",endocrine
"How to treat hypertension?","Treatment includes lifestyle changes...",cardiovascular
```

### JSON Format Example
```json
[
  {
    "question": "What are diabetes symptoms?",
    "answer": "Symptoms include frequent urination...",
    "category": "endocrine"
  }
]
```

### Automatic Column Detection
The system automatically recognizes these column names:
- **Questions**: question, query, input, user_input, patient_question, q
- **Answers**: answer, response, reply, output, bot_response, a
- **Categories**: category, topic, specialty, department (optional)

## 🎯 After Training

### Test Your Chatbot
```bash
# Command line
python -m src.chatbot ./models/kaggle_healthcare_chatbot/final_model

# Web interface
python -m src.web_interface --model_path ./models/kaggle_healthcare_chatbot/final_model
```

### Share Your Chatbot
```bash
python -m src.web_interface \
  --model_path ./models/kaggle_healthcare_chatbot/final_model \
  --share
```

## 🔧 Common Issues and Solutions

### Issue: "Dataset not found"
**Solution**: Make sure your file is in the project directory and the path is correct.

### Issue: "No question/answer columns found"
**Solution**: Ensure your CSV has recognizable column names (see list above).

### Issue: "Too few valid samples"
**Solution**: Check that questions are at least 5 characters and answers at least 10 characters.

### Issue: "Out of memory"
**Solution**: Use smaller batch size (`--batch_size 1`) or smaller model (`--model_key distilgpt2`).

### Issue: "Training too slow"
**Solution**: Limit samples (`--max_samples 1000`) or use fewer epochs (`--epochs 2`).

## 📁 Expected Output Files

After training, you'll have:
```
models/kaggle_healthcare_chatbot/
├── final_model/                    # Your trained chatbot
├── kaggle_training_summary.json    # Training details
├── final_evaluation_results.json   # Performance metrics
└── qualitative_evaluation.json     # Sample conversations
```

## 🎉 Ready to Start?

1. **Upload your Kaggle dataset** to this directory
2. **Run the setup script**: `python setup_kaggle_dataset.py`
3. **Follow the interactive prompts**
4. **Start training** when ready!

The system will handle all the technical details automatically. Your Kaggle dataset will be transformed into a working healthcare chatbot!

---

**Need help?** Check `KAGGLE_DATASET_GUIDE.md` for detailed documentation.

**Ready to begin!** 🏥🤖