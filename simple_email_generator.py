import streamlit as st
import re
from datetime import datetime

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

def generate_template_response(original_email, tone, context):
    """Generate template-based response"""
    
    # Extract sender name if possible
    sender_name = "there"
    lines = original_email.split('\n')
    for line in lines:
        if any(word in line.lower() for word in ['best', 'regards', 'sincerely', 'thanks']):
            # Look for name in the line above or after
            words = line.split()
            if len(words) > 1:
                sender_name = words[-1].replace(',', '')
                break
    
    templates = {
        "Professional": f"""Dear {sender_name},

Thank you for your email regarding {context}.

I have reviewed your message and understand your requirements. I will look into this matter and provide you with a detailed response shortly.

If you have any urgent questions in the meantime, please don't hesitate to reach out.

Best regards,
[Your Name]""",
        
        "Casual": f"""Hi {sender_name}!

Thanks for reaching out about {context}. 

I got your message and I'll get back to you soon with more details. Let me know if you need anything else in the meantime!

Cheers,
[Your Name]""",
        
        "Apologetic": f"""Dear {sender_name},

I sincerely apologize for any inconvenience regarding {context}.

I understand your concerns and take full responsibility for this situation. I am working to resolve this matter immediately and will keep you updated on the progress.

Thank you for your patience and understanding.

Best regards,
[Your Name]""",
        
        "Grateful": f"""Dear {sender_name},

Thank you so much for your email about {context}.

I truly appreciate you taking the time to reach out. Your input is valuable to me, and I'm grateful for the opportunity to assist you.

I'll review this carefully and get back to you with a comprehensive response.

With appreciation,
[Your Name]""",
        
        "Urgent": f"""Dear {sender_name},

Thank you for your urgent message regarding {context}.

I understand the time-sensitive nature of this matter and am prioritizing it accordingly. I will provide you with an update within the next few hours.

Please let me know if you need immediate assistance in the meantime.

Best regards,
[Your Name]""",
        
        "Follow-up": f"""Dear {sender_name},

I hope this email finds you well. I wanted to follow up on {context}.

Could you please provide an update on the current status? I'm happy to discuss this further at your convenience.

Looking forward to hearing from you.

Best regards,
[Your Name]""",
        
        "Meeting Request": f"""Dear {sender_name},

Thank you for your email regarding {context}.

I would be happy to schedule a meeting to discuss this further. Please let me know your availability for next week, and I'll send you a calendar invitation.

Some potential topics we could cover:
- Current status and requirements
- Next steps and timeline
- Any questions or concerns

Looking forward to our discussion.

Best regards,
[Your Name]"""
    }
    
    return templates.get(tone, templates["Professional"])

def enhance_response_with_context(response, original_email, context):
    """Enhance the template response with specific context"""
    
    # Extract key information from original email
    email_lower = original_email.lower()
    
    # Common email patterns and responses
    if "meeting" in email_lower or "schedule" in email_lower:
        response = response.replace("I will look into this matter", "I would be happy to schedule a meeting")
    
    if "urgent" in email_lower or "asap" in email_lower:
        response = response.replace("shortly", "as soon as possible")
        response = response.replace("within the next few hours", "immediately")
    
    if "thank" in email_lower:
        response = response.replace("Thank you for your email", "Thank you for your kind message")
    
    if "question" in email_lower or "help" in email_lower:
        response = response.replace("I will look into this matter", "I'll be happy to help answer your questions")
    
    return response

def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Email Response Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate professional email responses using smart templates</p>', unsafe_allow_html=True)
    
    # Info about AI version
    st.info("🚀 **Template Version**: This version uses smart templates. Install torch and transformers for AI-powered responses!")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Tone selection
        tone = st.selectbox(
            "Select Response Tone:",
            ["Professional", "Casual", "Apologetic", "Grateful", "Urgent", "Follow-up", "Meeting Request"],
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
            st.session_state.template_tone = "Meeting Request"
        if st.button("Project Update"):
            st.session_state.template_context = "project update"
            st.session_state.template_tone = "Professional"
        if st.button("Thank You"):
            st.session_state.template_context = "thank you message"
            st.session_state.template_tone = "Grateful"
        if st.button("Apology"):
            st.session_state.template_context = "apology"
            st.session_state.template_tone = "Apologetic"
    
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
        if hasattr(st.session_state, 'template_tone'):
            tone = st.session_state.template_tone
    
    with col2:
        st.subheader("✨ Generated Response")
        
        if st.button("🚀 Generate Response", type="primary"):
            if original_email.strip() and context.strip():
                with st.spinner("Generating your email response..."):
                    response = generate_template_response(original_email, tone, context)
                    enhanced_response = enhance_response_with_context(response, original_email, context)
                    
                    # Store in session state
                    st.session_state.generated_response = enhanced_response
                    st.session_state.generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.warning("Please provide both the original email and context.")
        
        # Display generated response
        if hasattr(st.session_state, 'generated_response'):
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            response_text = st.text_area(
                "Your Generated Response:",
                value=st.session_state.generated_response,
                height=300,
                help="You can edit this response before using it"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="📥 Download Response",
                data=response_text,
                file_name=f"email_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            # Copy button simulation
            st.code(response_text, language=None)
    
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
    
    # Installation instructions
    st.markdown("---")
    st.subheader("🔧 Upgrade to AI Version")
    
    with st.expander("Click here for AI installation instructions"):
        st.markdown("""
        **To get AI-powered responses, install these packages:**
        
        ```bash
        pip install torch transformers
        ```
        
        **Then run the full AI version:**
        ```bash
        streamlit run email_response_generator.py
        ```
        
        **What you'll get with AI:**
        - More natural, contextual responses
        - Better understanding of email content
        - Adaptive writing style
        - Smarter tone matching
        """)
    
    # Usage statistics
    if hasattr(st.session_state, 'generated_response'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Session Stats")
        st.sidebar.info(f"Last generated: {st.session_state.generation_time}")
        st.sidebar.success("✅ Template engine ready")

if __name__ == "__main__":
    main()
