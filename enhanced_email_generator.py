import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="📧 Enhanced Email Response Generator",
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
    .copy-btn {
        background-color: #28a745 !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.8rem !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the FLAN-T5-small model for text generation"""
    try:
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# -----------------------------
# Utilities
# -----------------------------

def extract_sender_name(original_email: str) -> str:
    """Extract the sender's name from an email text.
    This adapts and encapsulates the heuristic used in simple_email_generator.py.
    """
    if not original_email:
        return "there"

    # Try common greeting patterns: "Hi Name," or "Hello Name,"
    greeting_match = re.search(r"\b(?:hi|hello|dear)\s+([A-Z][a-zA-Z'\-]+)(?:\s+[A-Z][a-zA-Z'\-]+)?\s*,", original_email, flags=re.IGNORECASE)
    if greeting_match:
        name = greeting_match.group(1)
        return name

    # Look for sign-off lines (best, regards, sincerely, thanks)
    sender_name = "there"
    lines = original_email.split('\n')
    for idx, line in enumerate(lines):
        lower = line.strip().lower()
        if any(word in lower for word in ['best', 'regards', 'sincerely', 'thanks', 'thank you']):
            # Look at the next non-empty line for a name
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                candidate = lines[j].strip()
                # Simple name pattern: words with letters/hyphens/apostrophes (avoid emails/phones)
                if re.match(r"^[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2}$", candidate):
                    return candidate.split()[0]
            # Otherwise attempt last word in current line
            words = re.findall(r"[A-Za-z'\-]+", line)
            if words:
                sender_name = words[-1]
                break

    return sender_name or "there"


def generate_template_response(original_email: str, tone: str, context: str, sender_name: str) -> str:
    """Generate template-based response (richer fallback)."""
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


def enhance_response_with_context(response: str, original_email: str, context: str) -> str:
    """Enhance the template response with specific context (from simple app)."""
    email_lower = original_email.lower()

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


def post_generation_cleanup(text: str, sender_name: str, include_greeting: bool, include_signature: bool) -> str:
    """Replace placeholders and optionally ensure greeting/signature are present."""
    user_name_placeholder = "Your Name"

    # Replace generic placeholders
    replacements = {
        r"\[Name\]": sender_name or "there",
        r"\[Sender Name\]": sender_name or "there",
        r"\[Your Name\]": user_name_placeholder,
        r"<name>": sender_name or "there",
        r"<your name>": user_name_placeholder,
    }
    cleaned = text
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # Ensure greeting/signature if toggled
    if include_greeting:
        if not re.match(r"^(hi|hello|dear)\b", cleaned.strip(), flags=re.IGNORECASE):
            cleaned = f"Dear {sender_name or 'there'},\n\n" + cleaned
    if include_signature:
        if not re.search(r"\b(best regards|cheers|sincerely|thanks)\b", cleaned.lower()):
            cleaned = cleaned.rstrip() + f"\n\nBest regards,\n{user_name_placeholder}"

    return cleaned

# -----------------------------
# AI Generation
# -----------------------------

def generate_email_response(original_email: str, tone: str, context: str, tokenizer, model, sender_name: str) -> (str, str):
    """Generate email response using FLAN-T5. Returns (response, source)."""
    prompt = f"""
Tone: {tone}
Sender Name: {sender_name}
Context/Subject: {context}

Instruction: Generate a response email using the specified Tone and Subject. Address the sender by name.

Original email: {original_email}

Response:"""

    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove echo of prompt if any
        response = response.replace(prompt, "").strip()

        # Basic sanity check; fall back if too short or empty
        if len(response) < 40:
            raise ValueError("AI response too short; using template")

        return response, "AI Model"
    except Exception:
        # Fallback to template
        response = generate_template_response(original_email, tone, context, sender_name)
        response = enhance_response_with_context(response, original_email, context)
        return response, "Template Fallback"

# -----------------------------
# App
# -----------------------------

def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Enhanced Email Response Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered responses with smart templates and automatic name extraction</p>', unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Tone selection (7 options)
        tone = st.selectbox(
            "Select Response Tone:",
            ["Professional", "Casual", "Apologetic", "Grateful", "Urgent", "Follow-up", "Meeting Request"],
            key="tone_select",
            help="Choose the tone that best fits your response needs"
        )

        # Context input
        context = st.text_input(
            "Email Context/Subject:",
            placeholder="e.g., meeting request, project update, complaint",
            key="context_input",
            help="Brief description of what the email is about"
        )

        # Additional options
        st.subheader("📋 Additional Options")
        include_greeting = st.checkbox("Include greeting", value=True, key="include_greeting")
        include_signature = st.checkbox("Include signature placeholder", value=True, key="include_signature")

        # Quick templates (set both context and tone)
        st.subheader("🚀 Quick Templates")
        if st.button("Meeting Request"):
            st.session_state.context_input = "meeting request"
            st.session_state.tone_select = "Meeting Request"
        if st.button("Project Update"):
            st.session_state.context_input = "project update"
            st.session_state.tone_select = "Professional"
        if st.button("Thank You"):
            st.session_state.context_input = "thank you message"
            st.session_state.tone_select = "Grateful"
        if st.button("Apology"):
            st.session_state.context_input = "apology"
            st.session_state.tone_select = "Apologetic"

    # Load model once sidebar is ready
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.error("Failed to load the AI model. Please check your internet connection and try again.")
        st.stop()

    # Main area
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
            help="Paste the entire email you received",
            key="original_email_text"
        )

        # Keep context in sync if quick template clicked (already handled by keys)
        tone = st.session_state.get("tone_select", tone)
        context = st.session_state.get("context_input", context)

    with col2:
        st.subheader("✨ Generated Response")
        if st.button("🚀 Generate Response", type="primary"):
            if original_email.strip() and context.strip():
                with st.spinner("Generating your email response..."):
                    # Extract sender name from original email
                    sender_name = extract_sender_name(original_email)

                    # Generate with AI, fallback to template if needed
                    response_text, source = generate_email_response(original_email, tone, context, tokenizer, model, sender_name)

                    # Cleanup placeholders and ensure greeting/signature
                    final_text = post_generation_cleanup(response_text, sender_name, st.session_state.get("include_greeting", True), st.session_state.get("include_signature", True))

                    # Store in session state
                    st.session_state.generated_response = final_text
                    st.session_state.generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.response_source = source
                    st.session_state.sender_name = sender_name
            else:
                st.warning("Please provide both the original email and context.")

        # Display generated response
        if 'generated_response' in st.session_state:
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            response_text = st.text_area(
                "Your Generated Response:",
                value=st.session_state.generated_response,
                height=300,
                help="You can edit this response before using it",
                key="response_area"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Status transparency
            if st.session_state.get("response_source") == "AI Model":
                st.success("✅ Generated by AI Model (FLAN-T5)")
            else:
                st.info("ℹ️ Generated by Template Fallback")

            # Update stored value in case user edits it
            st.session_state.generated_response = response_text

            # Download button
            st.download_button(
                label="📥 Download Response",
                data=response_text,
                file_name=f"email_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

            # Copy to Clipboard (robust JS with explicit key)
            st.markdown(
                """
                <script>
                function copyResponseToClipboard(){
                    const textareas = parent.document.querySelectorAll('textarea');
                    let target = null;
                    for (const ta of textareas){
                        if (ta.getAttribute('aria-label') === 'Your Generated Response:') { target = ta; break; }
                    }
                    if (target){
                        navigator.clipboard.writeText(target.value).then(() => {
                            alert('Response copied to clipboard!');
                        });
                    } else {
                        alert('Could not find response area to copy.');
                    }
                }
                </script>
                <button class="copy-btn" onclick="copyResponseToClipboard()">📋 Copy to Clipboard</button>
                """,
                unsafe_allow_html=True
            )

    # Footer with tips and stats
    st.markdown("---")
    st.subheader("💡 Tips for Better Results")

    tips1, tips2, tips3 = st.columns(3)
    with tips1:
        st.markdown(
            """
            **📝 Original Email**
            - Include the complete email
            - Keep sender's name and context
            - Don't remove important details
            """
        )
    with tips2:
        st.markdown(
            """
            **🎯 Context**
            - Be specific about the topic
            - Mention urgency if needed
            - Include key keywords
            """
        )
    with tips3:
        st.markdown(
            """
            **✨ Tone Selection**
            - Professional: Business emails
            - Casual: Friends/colleagues
            - Apologetic: Mistakes/delays
            - Urgent/Follow-up/Meeting as needed
            """
        )

    if 'generated_response' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Session Stats")
        st.sidebar.info(f"Last generated: {st.session_state.generation_time}")
        st.sidebar.success("✅ Model loaded successfully")
        st.sidebar.write(f"Detected sender: {st.session_state.get('sender_name', 'there')}")


if __name__ == "__main__":
    main()
