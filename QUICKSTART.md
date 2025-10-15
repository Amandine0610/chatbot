# 🚀 Quick Start Guide

Get your Healthcare Chatbot up and running in minutes!

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Quick Version)
```bash
# Quick training with minimal epochs for testing
python train_chatbot.py --epochs 1 --batch_size 2 --skip_hyperparameter_search
```

### 3. Test the Chatbot
```bash
# Command line interface
python -m src.chatbot ./models/healthcare_chatbot/final_model

# Or web interface
python -m src.web_interface --model_path ./models/healthcare_chatbot/final_model
```

## 🎯 Common Use Cases

### Training a Custom Model
```bash
# Full training with hyperparameter search
python train_chatbot.py --model_key dialogpt-medium --epochs 3

# Quick training for testing
python train_chatbot.py --model_key distilgpt2 --epochs 1 --skip_hyperparameter_search
```

### Running Different Interfaces

#### Command Line Chat
```bash
python -m src.chatbot ./models/healthcare_chatbot/final_model
```

#### Web Interface
```bash
python -m src.web_interface --model_path ./models/healthcare_chatbot/final_model
```

#### Demo Mode (No Model Required)
```bash
python -m src.web_interface --demo
```

### Evaluation Only
```bash
python train_chatbot.py --evaluate_only --model_path_for_eval ./models/healthcare_chatbot/final_model
```

## 💡 Example Conversations

### Healthcare Questions
```
User: What are the symptoms of diabetes?
Bot: Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, extreme fatigue, blurred vision, slow-healing cuts and bruises, and tingling or numbness in hands or feet. If you experience these symptoms, consult a healthcare professional for proper diagnosis.

User: How can I prevent heart disease?
Bot: To prevent heart disease: maintain a healthy diet low in saturated fats and high in fruits and vegetables, exercise regularly (at least 150 minutes per week), avoid smoking, limit alcohol consumption, manage stress, maintain a healthy weight, control blood pressure and cholesterol levels, and get regular check-ups.
```

### Out-of-Domain Handling
```
User: What's the weather like today?
Bot: I'm a healthcare assistant and can only help with health-related questions. Could you please ask me something about health, medical conditions, or wellness?

User: Can you help me with my math homework?
Bot: I specialize in healthcare topics. Please feel free to ask me about symptoms, treatments, medications, or general health advice.
```

## 🛠️ Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Use smaller batch size
python train_chatbot.py --batch_size 1

# Use smaller model
python train_chatbot.py --model_key distilgpt2
```

#### Model Not Found
```bash
# Check if model exists
ls -la ./models/healthcare_chatbot/final_model/

# Train a new model
python train_chatbot.py
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
export PYTHONPATH="${PYTHONPATH}:/workspace"
```

## 📊 Quick Performance Check

### Test Model Quality
```python
from src.evaluation import HealthcareEvaluator

evaluator = HealthcareEvaluator("./models/healthcare_chatbot/final_model")
test_questions = [
    "What are the symptoms of fever?",
    "How to prevent heart disease?",
    "What should I eat for better health?"
]

results = evaluator.qualitative_evaluation(test_questions)
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['response']}\n")
```

### Check Training Progress
```bash
# View training logs
tail -f training.log

# Check model files
ls -la ./models/healthcare_chatbot/final_model/
```

## 🎨 Customization

### Add Your Own Data
1. Edit `data/healthcare_qa_dataset.json`
2. Add your question-answer pairs
3. Retrain the model:
```bash
python train_chatbot.py
```

### Change Model
```bash
# Use different base model
python train_chatbot.py --model_key gpt2-medium

# Available models: gpt2, gpt2-medium, distilgpt2, dialogpt-small, dialogpt-medium, t5-small
```

### Adjust Training
```bash
# More epochs for better quality
python train_chatbot.py --epochs 5

# Different learning rate
python train_chatbot.py --learning_rate 3e-5

# Larger batch size (if you have GPU memory)
python train_chatbot.py --batch_size 8
```

## 🌐 Web Interface Features

### Basic Usage
1. Open web interface: `python -m src.web_interface --model_path ./models/healthcare_chatbot/final_model`
2. Navigate to `http://localhost:7860`
3. Start chatting!

### Advanced Features
- **Export Conversations**: Click "Export Chat" to save as JSON
- **Health Tips**: Click "Health Tip" for wellness advice
- **Quick Topics**: Use preset health questions
- **Session Stats**: View conversation statistics

### Sharing
```bash
# Create public link
python -m src.web_interface --model_path ./models/healthcare_chatbot/final_model --share

# Add authentication
python -m src.web_interface --auth username password
```

## 📱 Integration Examples

### Python API
```python
from src.chatbot import HealthcareChatbot

# Initialize
chatbot = HealthcareChatbot("./models/healthcare_chatbot/final_model")

# Single question
response = chatbot.chat("What are flu symptoms?")
print(response["response"])

# Conversation
chatbot.chat("Hello")
response = chatbot.chat("I have a headache, what should I do?")
print(response["response"])
```

### Batch Processing
```python
questions = [
    "What are diabetes symptoms?",
    "How to prevent heart disease?",
    "What foods boost immunity?"
]

for question in questions:
    response = chatbot.chat(question)
    print(f"Q: {question}")
    print(f"A: {response['response']}\n")
```

## 🎯 Next Steps

1. **Improve Dataset**: Add more healthcare Q&A pairs
2. **Fine-tune Parameters**: Experiment with different hyperparameters
3. **Evaluate Performance**: Use the evaluation metrics to assess quality
4. **Deploy**: Set up the web interface for others to use
5. **Customize**: Adapt the chatbot for specific healthcare domains

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Code Examples**: Check the `src/` directory
- **Training Logs**: Review `training.log`
- **Model Info**: Inspect `./models/healthcare_chatbot/training_summary.json`

Happy chatbotting! 🏥🤖