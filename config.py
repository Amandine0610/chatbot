"""
Configuration file for Healthcare Chatbot.
Contains default settings and hyperparameters.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")

# Dataset configuration
DATASET_CONFIG = {
    "default_dataset_path": os.path.join(DATA_DIR, "healthcare_qa_dataset.json"),
    "train_split": 0.8,
    "max_sequence_length": 512,
    "min_answer_length": 10,
    "max_answer_length": 500
}

# Model configuration
MODEL_CONFIG = {
    "default_model_key": "dialogpt-medium",
    "supported_models": {
        "gpt2": {
            "model_name": "gpt2",
            "type": "causal_lm",
            "recommended_batch_size": 4,
            "recommended_lr": 5e-5
        },
        "gpt2-medium": {
            "model_name": "gpt2-medium",
            "type": "causal_lm",
            "recommended_batch_size": 2,
            "recommended_lr": 3e-5
        },
        "distilgpt2": {
            "model_name": "distilgpt2",
            "type": "causal_lm",
            "recommended_batch_size": 8,
            "recommended_lr": 5e-5
        },
        "dialogpt-small": {
            "model_name": "microsoft/DialoGPT-small",
            "type": "causal_lm",
            "recommended_batch_size": 8,
            "recommended_lr": 5e-5
        },
        "dialogpt-medium": {
            "model_name": "microsoft/DialoGPT-medium",
            "type": "causal_lm",
            "recommended_batch_size": 4,
            "recommended_lr": 5e-5
        },
        "t5-small": {
            "model_name": "t5-small",
            "type": "seq2seq",
            "recommended_batch_size": 4,
            "recommended_lr": 3e-4
        }
    },
    "default_output_dir": os.path.join(MODELS_DIR, "healthcare_chatbot")
}

# Training configuration
TRAINING_CONFIG = {
    "default_hyperparameters": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "num_train_epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 2
    },
    "hyperparameter_search_grid": {
        "learning_rate": [3e-5, 5e-5, 1e-4],
        "per_device_train_batch_size": [2, 4],
        "num_train_epochs": [2, 3, 4],
        "warmup_steps": [50, 100, 200]
    },
    "early_stopping_patience": 3
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["bleu", "rouge", "semantic_similarity", "perplexity"],
    "test_questions": [
        "What are the symptoms of diabetes?",
        "How can I prevent heart disease?",
        "What should I do if I have a fever?",
        "How much water should I drink daily?",
        "What are healthy snack options?",
        "Can you help me with my homework?",  # Out of domain
        "What's the weather like today?"  # Out of domain
    ],
    "max_eval_samples": 100,
    "generation_params": {
        "max_length": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }
}

# Chatbot configuration
CHATBOT_CONFIG = {
    "max_conversation_history": 5,
    "max_response_length": 500,
    "healthcare_keywords": {
        'symptoms', 'disease', 'medication', 'treatment', 'doctor', 'hospital',
        'health', 'medical', 'pain', 'fever', 'infection', 'diagnosis',
        'therapy', 'surgery', 'prescription', 'vaccine', 'allergy', 'chronic',
        'acute', 'wellness', 'fitness', 'nutrition', 'diet', 'exercise',
        'mental health', 'anxiety', 'depression', 'stress', 'sleep',
        'blood pressure', 'diabetes', 'heart', 'lung', 'kidney', 'liver',
        'cancer', 'tumor', 'virus', 'bacteria', 'immune', 'pregnancy',
        'pediatric', 'elderly', 'emergency', 'first aid', 'injury'
    },
    "out_of_domain_responses": [
        "I'm a healthcare assistant and can only help with health-related questions. Could you please ask me something about health, medical conditions, or wellness?",
        "I specialize in healthcare topics. Please feel free to ask me about symptoms, treatments, medications, or general health advice.",
        "I'm designed to assist with healthcare questions only. How can I help you with your health concerns today?",
        "My expertise is in healthcare and medical topics. Please ask me something related to health or medical care."
    ],
    "greeting_responses": [
        "Hello! I'm your healthcare assistant. I can help you with questions about health, medical conditions, symptoms, treatments, and general wellness. How can I assist you today?",
        "Hi there! I'm here to help with your healthcare questions. Feel free to ask me about medical symptoms, treatments, medications, or health advice.",
        "Welcome! I'm a healthcare chatbot trained to assist with medical and health-related questions. What would you like to know about your health today?"
    ],
    "health_tips": [
        "Remember to stay hydrated by drinking at least 8 glasses of water daily.",
        "Regular exercise for 30 minutes a day can significantly improve your health.",
        "Getting 7-9 hours of sleep is crucial for your physical and mental well-being.",
        "Eating a balanced diet with fruits and vegetables boosts your immune system.",
        "Don't forget to wash your hands frequently to prevent infections.",
        "Regular health check-ups can help detect issues early.",
        "Managing stress through meditation or relaxation techniques is important for health.",
        "Limit processed foods and choose whole, natural foods when possible."
    ]
}

# Web interface configuration
WEB_CONFIG = {
    "default_port": 7860,
    "default_share": False,
    "theme": "soft",
    "title": "Healthcare Chatbot",
    "description": "Your AI-powered healthcare companion for medical questions and health advice",
    "disclaimer": "⚠️ Disclaimer: This chatbot provides general health information only. Always consult healthcare professionals for medical advice.",
    "max_chat_history": 50,
    "export_format": "json"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": os.path.join(LOGS_DIR, "healthcare_chatbot.log"),
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# System configuration
SYSTEM_CONFIG = {
    "device": "auto",  # "auto", "cuda", "cpu"
    "mixed_precision": True,
    "dataloader_num_workers": 0,
    "pin_memory": False,
    "seed": 42
}

# Create directories
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, MODELS_DIR, LOGS_DIR, EXPORTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories when module is imported
create_directories()