"""
Web interface for healthcare chatbot using Gradio.
Provides an intuitive web-based interface for users to interact with the chatbot.
"""

import gradio as gr
import json
import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
import pandas as pd

from .chatbot import HealthcareChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareChatbotWebInterface:
    """
    Web interface for the healthcare chatbot using Gradio.
    """
    
    def __init__(self, model_path: str, share: bool = False, 
                 server_port: int = 7860, auth: Tuple[str, str] = None):
        """
        Initialize the web interface.
        
        Args:
            model_path: Path to the fine-tuned model
            share: Whether to create a public link
            server_port: Port to run the server on
            auth: Optional (username, password) tuple for authentication
        """
        self.model_path = model_path
        self.share = share
        self.server_port = server_port
        self.auth = auth
        
        # Initialize chatbot
        try:
            self.chatbot = HealthcareChatbot(model_path)
            logger.info("Healthcare chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise
        
        # Session management
        self.sessions = {}
        
        # Create interface
        self.interface = self.create_interface()
    
    def chat_with_history(self, message: str, history: List[List[str]], 
                         session_state: Dict) -> Tuple[str, List[List[str]], Dict]:
        """
        Handle chat with conversation history.
        
        Args:
            message: User message
            history: Chat history
            session_state: Session state dictionary
            
        Returns:
            Tuple of (empty_string, updated_history, updated_session_state)
        """
        if not message.strip():
            return "", history, session_state
        
        try:
            # Get session ID or create new one
            session_id = session_state.get("session_id")
            if not session_id:
                session_id = self.chatbot.start_new_session()
                session_state["session_id"] = session_id
                logger.info(f"Started new web session: {session_id}")
            
            # Get chatbot response
            response_data = self.chatbot.chat(message)
            response = response_data["response"]
            
            # Add to history
            history.append([message, response])
            
            # Update session statistics
            if "message_count" not in session_state:
                session_state["message_count"] = 0
            session_state["message_count"] += 1
            
            # Add response type indicator
            response_type = response_data.get("type", "unknown")
            if response_type == "out_of_domain":
                response += "\n\n⚠️ *Note: I specialize in healthcare topics. Please ask health-related questions for best assistance.*"
            elif response_type == "healthcare":
                response += "\n\n🏥 *Healthcare Assistant Response*"
            
            return "", history, session_state
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_response = "I apologize, but I encountered an error. Please try again or rephrase your question."
            history.append([message, error_response])
            return "", history, session_state
    
    def clear_chat(self) -> Tuple[List, Dict]:
        """
        Clear chat history and start new session.
        
        Returns:
            Tuple of (empty_history, new_session_state)
        """
        self.chatbot.clear_history()
        new_session_id = self.chatbot.start_new_session()
        
        return [], {"session_id": new_session_id, "message_count": 0}
    
    def get_health_tip(self, session_state: Dict) -> Tuple[str, Dict]:
        """
        Get a random health tip.
        
        Args:
            session_state: Current session state
            
        Returns:
            Tuple of (health_tip, session_state)
        """
        tip = self.chatbot.get_health_tips()
        return tip, session_state
    
    def export_conversation(self, history: List[List[str]], 
                           session_state: Dict) -> str:
        """
        Export conversation history to a file.
        
        Args:
            history: Chat history
            session_state: Session state
            
        Returns:
            Path to exported file
        """
        try:
            # Create exports directory
            os.makedirs("exports", exist_ok=True)
            
            # Prepare conversation data
            conversation_data = {
                "session_id": session_state.get("session_id", "unknown"),
                "export_timestamp": datetime.now().isoformat(),
                "message_count": len(history),
                "conversation": []
            }
            
            for user_msg, bot_msg in history:
                conversation_data["conversation"].append({
                    "user": user_msg,
                    "assistant": bot_msg,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Save to file
            filename = f"conversation_{session_state.get('session_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("exports", filename)
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            return f"Error exporting: {str(e)}"
    
    def get_session_stats(self, session_state: Dict) -> str:
        """
        Get session statistics.
        
        Args:
            session_state: Current session state
            
        Returns:
            Formatted session statistics
        """
        session_id = session_state.get("session_id", "No active session")
        message_count = session_state.get("message_count", 0)
        
        stats = f"""
        **Session Statistics**
        - Session ID: {session_id}
        - Messages in this session: {message_count}
        - Healthcare Assistant Status: ✅ Active
        """
        
        return stats
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f1f8e9;
            margin-right: 20%;
        }
        """
        
        with gr.Blocks(
            title="Healthcare Chatbot",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # Session state
            session_state = gr.State({"session_id": None, "message_count": 0})
            
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>🏥 Healthcare Assistant Chatbot</h1>
                <p style="font-size: 18px; color: #666;">
                    Your AI-powered healthcare companion for medical questions and health advice
                </p>
                <p style="font-size: 14px; color: #888;">
                    ⚠️ <strong>Disclaimer:</strong> This chatbot provides general health information only. 
                    Always consult healthcare professionals for medical advice.
                </p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Main chat interface
                    chatbot_interface = gr.Chatbot(
                        label="Healthcare Assistant",
                        height=500,
                        show_label=True,
                        avatar_images=("👤", "🏥")
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me about symptoms, treatments, medications, or general health advice...",
                            label="Your Message",
                            lines=2,
                            max_lines=5
                        )
                        send_btn = gr.Button("Send 📤", variant="primary")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                        tip_btn = gr.Button("Health Tip 💡", variant="secondary")
                        export_btn = gr.Button("Export Chat 📥", variant="secondary")
                
                with gr.Column(scale=1):
                    # Sidebar with additional features
                    gr.HTML("<h3>🔧 Chat Controls</h3>")
                    
                    # Session statistics
                    stats_display = gr.Markdown("**Session Statistics**\n- No active session")
                    refresh_stats_btn = gr.Button("Refresh Stats 📊")
                    
                    # Health tip display
                    tip_display = gr.Textbox(
                        label="💡 Health Tip",
                        interactive=False,
                        lines=3
                    )
                    
                    # Export status
                    export_status = gr.Textbox(
                        label="📥 Export Status",
                        interactive=False,
                        lines=2
                    )
                    
                    # Quick health topics
                    gr.HTML("""
                    <h4>🩺 Quick Health Topics</h4>
                    <p style="font-size: 12px;">Click to ask about:</p>
                    """)
                    
                    with gr.Column():
                        topic_btns = []
                        topics = [
                            "What are symptoms of flu?",
                            "How to prevent heart disease?",
                            "Managing diabetes naturally",
                            "Healthy diet recommendations",
                            "Exercise for beginners"
                        ]
                        
                        for topic in topics:
                            btn = gr.Button(topic, size="sm")
                            topic_btns.append((btn, topic))
            
            # Event handlers
            
            # Send message
            send_btn.click(
                fn=self.chat_with_history,
                inputs=[msg_input, chatbot_interface, session_state],
                outputs=[msg_input, chatbot_interface, session_state]
            )
            
            msg_input.submit(
                fn=self.chat_with_history,
                inputs=[msg_input, chatbot_interface, session_state],
                outputs=[msg_input, chatbot_interface, session_state]
            )
            
            # Clear chat
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot_interface, session_state]
            )
            
            # Health tip
            tip_btn.click(
                fn=self.get_health_tip,
                inputs=[session_state],
                outputs=[tip_display, session_state]
            )
            
            # Export conversation
            export_btn.click(
                fn=self.export_conversation,
                inputs=[chatbot_interface, session_state],
                outputs=[export_status]
            )
            
            # Refresh statistics
            refresh_stats_btn.click(
                fn=self.get_session_stats,
                inputs=[session_state],
                outputs=[stats_display]
            )
            
            # Quick topic buttons
            for btn, topic in topic_btns:
                btn.click(
                    fn=lambda t=topic: (t, [], {}),
                    outputs=[msg_input, chatbot_interface, session_state]
                ).then(
                    fn=self.chat_with_history,
                    inputs=[msg_input, chatbot_interface, session_state],
                    outputs=[msg_input, chatbot_interface, session_state]
                )
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666; font-size: 12px;">
                <p>Healthcare Chatbot v1.0 | Built with Transformers & Gradio</p>
                <p>🔒 Your conversations are private and not stored permanently</p>
            </div>
            """)
        
        return interface
    
    def launch(self):
        """Launch the web interface."""
        logger.info("Launching Healthcare Chatbot web interface...")
        
        try:
            self.interface.launch(
                share=self.share,
                server_port=self.server_port,
                auth=self.auth,
                show_error=True,
                quiet=False
            )
        except Exception as e:
            logger.error(f"Error launching interface: {e}")
            raise


def create_demo_interface(model_path: str = None) -> gr.Blocks:
    """
    Create a demo interface without loading the actual model.
    Useful for testing the interface design.
    
    Args:
        model_path: Path to model (not used in demo)
        
    Returns:
        Gradio interface
    """
    def demo_chat(message, history):
        """Demo chat function with predefined responses."""
        if not message.strip():
            return "", history
        
        # Demo responses
        demo_responses = {
            "hello": "Hello! I'm a healthcare assistant demo. In the full version, I can help with health questions!",
            "symptoms": "In the full version, I can help identify symptoms and provide health advice.",
            "fever": "For fever, rest and stay hydrated. Consult a doctor if it persists. (This is a demo response)",
            "default": "This is a demo version. The full chatbot would provide detailed healthcare assistance!"
        }
        
        # Simple keyword matching for demo
        message_lower = message.lower()
        if "hello" in message_lower or "hi" in message_lower:
            response = demo_responses["hello"]
        elif "symptom" in message_lower:
            response = demo_responses["symptoms"]
        elif "fever" in message_lower:
            response = demo_responses["fever"]
        else:
            response = demo_responses["default"]
        
        history.append([message, response])
        return "", history
    
    with gr.Blocks(title="Healthcare Chatbot Demo") as demo:
        gr.HTML("<h1>🏥 Healthcare Chatbot Demo</h1>")
        gr.HTML("<p><strong>Note:</strong> This is a demo interface. Load a trained model to use the full functionality.</p>")
        
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Try: 'Hello' or 'What are fever symptoms?'")
        
        msg.submit(demo_chat, [msg, chatbot], [msg, chatbot])
        
        gr.Button("Clear").click(lambda: ([], ""), outputs=[chatbot, msg])
    
    return demo


def main():
    """
    Main function to launch the web interface.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Healthcare Chatbot Web Interface")
    parser.add_argument("--model_path", type=str, 
                       default="./models/healthcare_chatbot/final_model",
                       help="Path to the fine-tuned model")
    parser.add_argument("--share", action="store_true",
                       help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the server on")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo interface without loading model")
    parser.add_argument("--auth", type=str, nargs=2, metavar=('username', 'password'),
                       help="Add authentication (username password)")
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            # Launch demo interface
            print("Launching demo interface...")
            demo = create_demo_interface()
            demo.launch(share=args.share, server_port=args.port)
        else:
            # Launch full interface
            auth = tuple(args.auth) if args.auth else None
            web_interface = HealthcareChatbotWebInterface(
                model_path=args.model_path,
                share=args.share,
                server_port=args.port,
                auth=auth
            )
            web_interface.launch()
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo run the demo interface without a trained model, use:")
        print("python -m src.web_interface --demo")


if __name__ == "__main__":
    main()