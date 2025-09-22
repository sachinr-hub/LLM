import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import re
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="📧 Email Response Generator",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .email-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .response-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the T5 model for text generation"""
    try:
        # Using FLAN-T5 small for better performance and free usage
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_email_response(original_email, tone, context, tokenizer, model):
    """Generate email response using T5 model"""
    
    # Create a detailed prompt based on tone and context
    tone_instructions = {
        "Professional": "Write a professional and formal email response",
        "Casual": "Write a casual and friendly email response",
        "Apologetic": "Write an apologetic and understanding email response",
        "Grateful": "Write a grateful and appreciative email response",
        "Urgent": "Write an urgent but polite email response"
    }
    
    # Construct the prompt
    prompt = f"""
{tone_instructions[tone]}. 

Context: {context}

Original email: {original_email}

Response:"""
    
    try:
        # Tokenize and generate
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response
        response = response.replace(prompt, "").strip()
        
        # If response is too short or doesn't make sense, provide a template
        if len(response) < 20:
            response = generate_template_response(original_email, tone, context)
            
        return response
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return generate_template_response(original_email, tone, context)

def generate_template_response(original_email, tone, context):
    """Generate template-based response as fallback"""
    
    templates = {
        "Professional": f"""Dear [Name],

Thank you for your email regarding {context}.

I have reviewed your message and understand your requirements. I will look into this matter and provide you with a detailed response shortly.

If you have any urgent questions in the meantime, please don't hesitate to reach out.

Best regards,
[Your Name]""",
        
        "Casual": f"""Hi there!

Thanks for reaching out about {context}. 

I got your message and I'll get back to you soon with more details. Let me know if you need anything else in the meantime!

Cheers,
[Your Name]""",
        
        "Apologetic": f"""Dear [Name],

I sincerely apologize for any inconvenience regarding {context}.

I understand your concerns and take full responsibility for this situation. I am working to resolve this matter immediately and will keep you updated on the progress.

Thank you for your patience and understanding.

Best regards,
[Your Name]""",
        
        "Grateful": f"""Dear [Name],

Thank you so much for your email about {context}.

I truly appreciate you taking the time to reach out. Your input is valuable to me, and I'm grateful for the opportunity to assist you.

I'll review this carefully and get back to you with a comprehensive response.

With appreciation,
[Your Name]""",
        
        "Urgent": f"""Dear [Name],

Thank you for your urgent message regarding {context}.

I understand the time-sensitive nature of this matter and am prioritizing it accordingly. I will provide you with an update within the next few hours.

Please let me know if you need immediate assistance in the meantime.

Best regards,
[Your Name]"""
    }
    
    return templates.get(tone, templates["Professional"])

def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Email Response Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate professional email responses in seconds using AI</p>', unsafe_allow_html=True)
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load the AI model. Please check your internet connection and try again.")
        return
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Tone selection
        tone = st.selectbox(
            "Select Response Tone:",
            ["Professional", "Casual", "Apologetic", "Grateful", "Urgent"],
            help="Choose the tone that best fits your response needs"
        )
        
        # Context input
        context = st.text_input(
            "Email Context/Subject:",
            placeholder="e.g., meeting request, project update, complaint",
            help="Brief description of what the email is about"
        )
        
        # Additional options
        st.subheader("📋 Additional Options")
        include_greeting = st.checkbox("Include greeting", value=True)
        include_signature = st.checkbox("Include signature placeholder", value=True)
        
        # Quick templates
        st.subheader("🚀 Quick Templates")
        if st.button("Meeting Request"):
            st.session_state.template_context = "meeting request"
        if st.button("Project Update"):
            st.session_state.template_context = "project update"
        if st.button("Thank You"):
            st.session_state.template_context = "thank you message"
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📨 Original Email")
        original_email = st.text_area(
            "Paste the email you want to respond to:",
            height=300,
            placeholder="""Example:
Hi John,

I hope this email finds you well. I wanted to follow up on our discussion about the marketing campaign. Could we schedule a meeting next week to go over the details?

Looking forward to hearing from you.

Best,
Sarah""",
            help="Paste the entire email you received"
        )
        
        # Use template context if selected
        if hasattr(st.session_state, 'template_context'):
            context = st.session_state.template_context
    
    with col2:
        st.subheader("✨ Generated Response")
        
        if st.button("🚀 Generate Response", type="primary"):
            if original_email.strip() and context.strip():
                with st.spinner("Generating your email response..."):
                    response = generate_email_response(original_email, tone, context, tokenizer, model)
                    
                    # Store in session state
                    st.session_state.generated_response = response
                    st.session_state.generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.warning("Please provide both the original email and context.")
        
        # Display generated response
        if hasattr(st.session_state, 'generated_response'):
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.text_area(
                "Your Generated Response:",
                value=st.session_state.generated_response,
                height=300,
                help="You can edit this response before using it"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="📥 Download Response",
                data=st.session_state.generated_response,
                file_name=f"email_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            # Copy to clipboard (JavaScript)
            st.markdown("""
            <script>
            function copyToClipboard() {
                navigator.clipboard.writeText(document.querySelector('textarea[aria-label="Your Generated Response:"]').value);
                alert('Response copied to clipboard!');
            }
            </script>
            """, unsafe_allow_html=True)
    
    # Footer with tips
    st.markdown("---")
    st.subheader("💡 Tips for Better Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📝 Original Email**
        - Include the complete email
        - Keep sender's name and context
        - Don't remove important details
        """)
    
    with col2:
        st.markdown("""
        **🎯 Context**
        - Be specific about the topic
        - Mention urgency if needed
        - Include key keywords
        """)
    
    with col3:
        st.markdown("""
        **✨ Tone Selection**
        - Professional: Business emails
        - Casual: Friends/colleagues
        - Apologetic: Mistakes/delays
        """)
    
    # Usage statistics
    if hasattr(st.session_state, 'generated_response'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Session Stats")
        st.sidebar.info(f"Last generated: {st.session_state.generation_time}")
        st.sidebar.success("✅ Model loaded successfully")

if __name__ == "__main__":
    main()
