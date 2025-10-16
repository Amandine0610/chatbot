"""
Healthcare Chatbot inference pipeline.
Handles conversation management, response generation, and domain validation.
"""

import torch
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareChatbot:
    """
    Healthcare domain-specific chatbot with conversation management.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None, 
                 max_history: int = 5):
        """
        Initialize the healthcare chatbot.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on
            max_history: Maximum conversation history to maintain
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history = max_history
        
        # Conversation state
        self.conversation_history = []
        self.session_id = None
        
        # Healthcare domain keywords for validation
        self.healthcare_keywords = {
            'symptoms', 'disease', 'medication', 'treatment', 'doctor', 'hospital',
            'health', 'medical', 'pain', 'fever', 'infection', 'diagnosis',
            'therapy', 'surgery', 'prescription', 'vaccine', 'allergy', 'chronic',
            'acute', 'wellness', 'fitness', 'nutrition', 'diet', 'exercise',
            'mental health', 'anxiety', 'depression', 'stress', 'sleep',
            'blood pressure', 'diabetes', 'heart', 'lung', 'kidney', 'liver',
            'cancer', 'tumor', 'virus', 'bacteria', 'immune', 'pregnancy',
            'pediatric', 'elderly', 'emergency', 'first aid', 'injury'
        }
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.load_model()
        
        # Response templates
        self.out_of_domain_responses = [
            "I'm a healthcare assistant and can only help with health-related questions. Could you please ask me something about health, medical conditions, or wellness?",
            "I specialize in healthcare topics. Please feel free to ask me about symptoms, treatments, medications, or general health advice.",
            "I'm designed to assist with healthcare questions only. How can I help you with your health concerns today?",
            "My expertise is in healthcare and medical topics. Please ask me something related to health or medical care."
        ]
        
        self.greeting_responses = [
            "Hello! I'm your healthcare assistant. I can help you with questions about health, medical conditions, symptoms, treatments, and general wellness. How can I assist you today?",
            "Hi there! I'm here to help with your healthcare questions. Feel free to ask me about medical symptoms, treatments, medications, or health advice.",
            "Welcome! I'm a healthcare chatbot trained to assist with medical and health-related questions. What would you like to know about your health today?"
        ]
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading healthcare chatbot model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Healthcare chatbot model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def start_new_session(self) -> str:
        """
        Start a new conversation session.
        
        Returns:
            Session ID
        """
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        
        logger.info(f"Started new session: {self.session_id}")
        return self.session_id
    
    def is_healthcare_related(self, text: str) -> bool:
        """
        Check if the input text is healthcare-related.
        
        Args:
            text: Input text to check
            
        Returns:
            True if healthcare-related, False otherwise
        """
        text_lower = text.lower()
        
        # Check for healthcare keywords
        for keyword in self.healthcare_keywords:
            if keyword in text_lower:
                return True
        
        # Check for common health-related patterns
        health_patterns = [
            r'\b(hurt|pain|ache|sore|sick|ill|disease|condition)\b',
            r'\b(doctor|physician|nurse|hospital|clinic|medical)\b',
            r'\b(medicine|medication|drug|pill|tablet|treatment)\b',
            r'\b(symptom|diagnosis|cure|heal|recover|therapy)\b',
            r'\b(health|healthy|wellness|fitness|nutrition)\b'
        ]
        
        for pattern in health_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def is_greeting(self, text: str) -> bool:
        """
        Check if the input text is a greeting.
        
        Args:
            text: Input text to check
            
        Returns:
            True if greeting, False otherwise
        """
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                    'good evening', 'greetings', 'start', 'begin']
        
        text_lower = text.lower().strip()
        return any(greeting in text_lower for greeting in greetings)
    
    def preprocess_input(self, user_input: str) -> str:
        """
        Preprocess user input.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Preprocessed input
        """
        # Clean and normalize
        text = user_input.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper punctuation
        if text and not text.endswith(('.', '?', '!')):
            text += '?'
        
        return text
    
    def generate_response(self, user_input: str, max_length: int = 200,
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate response using the fine-tuned model.
        
        Args:
            user_input: User's input message
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response
        """
        # Create conversation context
        context_parts = []
        
        # Add recent conversation history
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current user input
        context_parts.append(f"User: {user_input}")
        context_parts.append("Assistant:")
        
        # Create full context
        context = "<|startoftext|>" + "<|endoftext|>".join(context_parts)
        
        # Tokenize
        inputs = self.tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=400)
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.1
            )
        
        # Decode and extract response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response.strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and post-process the generated response.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response
        """
        # Remove any remaining special tokens
        response = re.sub(r'<\|.*?\|>', '', response)
        
        # Remove repetitive patterns
        sentences = response.split('.')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        response = '. '.join(unique_sentences)
        
        # Ensure proper ending
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Limit response length
        if len(response) > 500:
            sentences = response.split('.')
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Main chat interface.
        
        Args:
            user_input: User's input message
            
        Returns:
            Dictionary with response and metadata
        """
        # Start session if not exists
        if self.session_id is None:
            self.start_new_session()
        
        # Preprocess input
        processed_input = self.preprocess_input(user_input)
        
        # Check for greetings
        if self.is_greeting(processed_input):
            response = np.random.choice(self.greeting_responses)
            response_type = "greeting"
        
        # Check if healthcare-related
        elif self.is_healthcare_related(processed_input):
            response = self.generate_response(processed_input)
            response_type = "healthcare"
        
        # Out of domain
        else:
            response = np.random.choice(self.out_of_domain_responses)
            response_type = "out_of_domain"
        
        # Add to conversation history
        if response_type in ["healthcare", "greeting"]:
            self.conversation_history.append({
                "user": processed_input,
                "assistant": response,
                "timestamp": datetime.now().isoformat(),
                "type": response_type
            })
            
            # Maintain history limit
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Prepare response
        chat_response = {
            "response": response,
            "type": response_type,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "is_healthcare_related": response_type == "healthcare"
        }
        
        logger.info(f"User: {processed_input}")
        logger.info(f"Assistant ({response_type}): {response}")
        
        return chat_response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get current conversation history.
        
        Returns:
            List of conversation exchanges
        """
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def save_conversation(self, filepath: str):
        """
        Save conversation history to file.
        
        Args:
            filepath: Path to save conversation
        """
        conversation_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "history": self.conversation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        logger.info(f"Conversation saved to {filepath}")
    
    def get_health_tips(self) -> str:
        """
        Get random health tips.
        
        Returns:
            Health tip message
        """
        tips = [
            "Remember to stay hydrated by drinking at least 8 glasses of water daily.",
            "Regular exercise for 30 minutes a day can significantly improve your health.",
            "Getting 7-9 hours of sleep is crucial for your physical and mental well-being.",
            "Eating a balanced diet with fruits and vegetables boosts your immune system.",
            "Don't forget to wash your hands frequently to prevent infections.",
            "Regular health check-ups can help detect issues early.",
            "Managing stress through meditation or relaxation techniques is important for health.",
            "Limit processed foods and choose whole, natural foods when possible."
        ]
        
        return f"💡 Health Tip: {np.random.choice(tips)}"


class ChatbotCLI:
    """
    Command-line interface for the healthcare chatbot.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize CLI interface.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        self.chatbot = HealthcareChatbot(model_path)
        
    def run(self):
        """Run the CLI interface."""
        print("=" * 60)
        print("🏥 HEALTHCARE CHATBOT")
        print("=" * 60)
        print("Welcome to your personal healthcare assistant!")
        print("I can help you with health questions, symptoms, and medical advice.")
        print("Type 'quit' to exit, 'clear' to clear history, 'tip' for health tips.")
        print("=" * 60)
        
        # Start session
        session_id = self.chatbot.start_new_session()
        print(f"Session started: {session_id}")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() == 'quit':
                    print("Thank you for using Healthcare Chatbot. Stay healthy! 👋")
                    break
                
                elif user_input.lower() == 'clear':
                    self.chatbot.clear_history()
                    print("Conversation history cleared.")
                    continue
                
                elif user_input.lower() == 'tip':
                    print(f"Assistant: {self.chatbot.get_health_tips()}")
                    continue
                
                # Get chatbot response
                response = self.chatbot.chat(user_input)
                
                # Display response with appropriate emoji
                emoji = "🏥" if response["is_healthcare_related"] else "ℹ️"
                print(f"Assistant {emoji}: {response['response']}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Take care of your health! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"CLI error: {e}")


def main():
    """
    Main function to run the chatbot CLI.
    """
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./models/healthcare_chatbot/final_model"
    
    try:
        cli = ChatbotCLI(model_path)
        cli.run()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("Please ensure you have a trained model at the specified path.")


if __name__ == "__main__":
    main()