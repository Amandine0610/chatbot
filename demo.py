#!/usr/bin/env python3
"""
Demo script for Healthcare Chatbot.
Demonstrates the complete pipeline from training to inference.
"""

import os
import sys
import time
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """Print demo banner."""
    print("=" * 70)
    print("🏥 HEALTHCARE CHATBOT DEMO")
    print("=" * 70)
    print("This demo showcases the complete healthcare chatbot pipeline:")
    print("1. Data preprocessing")
    print("2. Model training (quick version)")
    print("3. Evaluation")
    print("4. Interactive chat")
    print("=" * 70)
    print()

def demo_data_preprocessing():
    """Demonstrate data preprocessing."""
    print("📊 STEP 1: Data Preprocessing")
    print("-" * 30)
    
    try:
        from src.data_preprocessing import HealthcareDataPreprocessor
        
        # Initialize preprocessor
        preprocessor = HealthcareDataPreprocessor()
        
        # Load dataset
        data = preprocessor.load_dataset("./data/healthcare_qa_dataset.json")
        print(f"✅ Loaded {len(data)} healthcare Q&A pairs")
        
        # Get statistics
        stats = preprocessor.get_data_statistics(data)
        print("📈 Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # Show sample
        sample = data[0]
        print(f"\n📝 Sample Q&A:")
        print(f"   Q: {sample['question']}")
        print(f"   A: {sample['answer'][:100]}...")
        
        print("✅ Data preprocessing completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return False

def demo_quick_training():
    """Demonstrate quick model training."""
    print("🎓 STEP 2: Quick Model Training")
    print("-" * 30)
    
    try:
        from src.fine_tuning import HealthcareFinetuner
        
        # Initialize fine-tuner with smaller model for demo
        print("🔄 Initializing model (this may take a moment)...")
        finetuner = HealthcareFinetuner("distilgpt2", "./models/demo_chatbot")
        
        # Prepare data
        print("📊 Preparing training data...")
        dataset = finetuner.prepare_data("./data/healthcare_qa_dataset.json", train_split=0.8)
        
        # Quick training with minimal parameters
        print("🚀 Starting quick training (1 epoch for demo)...")
        hyperparameters = {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 1,  # Small batch for demo
            "num_train_epochs": 1,  # Just 1 epoch for demo
            "warmup_steps": 10
        }
        
        results = finetuner.fine_tune(dataset, hyperparameters, use_early_stopping=False)
        
        print(f"✅ Training completed!")
        print(f"   Final validation loss: {results['eval_loss']:.4f}")
        print(f"   Model saved to: {results['model_path']}")
        print()
        
        return results['model_path']
        
    except Exception as e:
        print(f"❌ Error in training: {e}")
        print("💡 You can skip training and use the demo mode instead.")
        return None

def demo_evaluation(model_path):
    """Demonstrate model evaluation."""
    print("📊 STEP 3: Model Evaluation")
    print("-" * 30)
    
    try:
        from src.evaluation import HealthcareEvaluator
        
        # Initialize evaluator
        print("🔄 Loading model for evaluation...")
        evaluator = HealthcareEvaluator(model_path)
        
        # Test questions
        test_questions = [
            "What are the symptoms of diabetes?",
            "How can I prevent heart disease?",
            "What should I do if I have a fever?",
            "Can you help me with my homework?"  # Out of domain
        ]
        
        print("🧪 Running qualitative evaluation...")
        results = evaluator.qualitative_evaluation(test_questions)
        
        print("📋 Evaluation Results:")
        for i, result in enumerate(results, 1):
            print(f"\n   Test {i}:")
            print(f"   Q: {result['question']}")
            print(f"   A: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
        
        print("\n✅ Evaluation completed!")
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        return False

def demo_interactive_chat(model_path):
    """Demonstrate interactive chat."""
    print("💬 STEP 4: Interactive Chat Demo")
    print("-" * 30)
    
    try:
        from src.chatbot import HealthcareChatbot
        
        # Initialize chatbot
        print("🔄 Loading chatbot...")
        chatbot = HealthcareChatbot(model_path)
        
        # Demo questions
        demo_questions = [
            "Hello!",
            "What are the symptoms of diabetes?",
            "How much water should I drink daily?",
            "What's the weather like?",  # Out of domain
            "Thank you!"
        ]
        
        print("🤖 Starting demo conversation:")
        print("=" * 50)
        
        for question in demo_questions:
            print(f"👤 User: {question}")
            
            response = chatbot.chat(question)
            print(f"🏥 Bot: {response['response']}")
            
            # Show response type
            if response['type'] == 'out_of_domain':
                print("   ℹ️  [Out of domain response]")
            elif response['type'] == 'healthcare':
                print("   ✅ [Healthcare response]")
            
            print("-" * 50)
            time.sleep(1)  # Pause for readability
        
        print("✅ Interactive chat demo completed!")
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error in chat demo: {e}")
        return False

def demo_web_interface():
    """Demonstrate web interface launch."""
    print("🌐 STEP 5: Web Interface")
    print("-" * 30)
    
    print("🚀 The web interface can be launched with:")
    print("   python -m src.web_interface --demo")
    print()
    print("Or with a trained model:")
    print("   python -m src.web_interface --model_path ./models/demo_chatbot/final_model")
    print()
    
    response = input("Would you like to launch the demo web interface? (y/n): ")
    if response.lower() == 'y':
        try:
            print("🔄 Launching demo web interface...")
            os.system("python -m src.web_interface --demo")
        except KeyboardInterrupt:
            print("\n👋 Web interface closed.")
        except Exception as e:
            print(f"❌ Error launching web interface: {e}")

def main():
    """Main demo function."""
    print_banner()
    
    # Check if we should run full demo or quick demo
    response = input("Run full demo with training? (y/n, default=n): ").strip().lower()
    run_training = response == 'y'
    
    print()
    
    # Step 1: Data preprocessing
    if not demo_data_preprocessing():
        print("❌ Demo failed at data preprocessing step.")
        return
    
    model_path = None
    
    if run_training:
        # Step 2: Quick training
        model_path = demo_quick_training()
        
        if model_path:
            # Step 3: Evaluation
            demo_evaluation(model_path)
            
            # Step 4: Interactive chat
            demo_interactive_chat(model_path)
    else:
        print("⏩ Skipping training (use --demo mode for web interface)")
        print()
    
    # Step 5: Web interface
    demo_web_interface()
    
    # Summary
    print("🎉 DEMO COMPLETED!")
    print("=" * 70)
    print("What you've seen:")
    print("✅ Healthcare dataset preprocessing")
    if run_training:
        print("✅ Model training pipeline")
        print("✅ Automated evaluation")
        print("✅ Interactive chatbot")
    print("✅ Web interface options")
    print()
    print("Next steps:")
    print("1. Train a full model: python train_chatbot.py")
    print("2. Launch web interface: python -m src.web_interface --demo")
    print("3. Read documentation: README.md")
    print()
    print("Thank you for trying the Healthcare Chatbot! 🏥🤖")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Please check the installation and try again.")