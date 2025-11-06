import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datetime import datetime
import re
import os
import json

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
def load_model(force_template: bool = False, finetuned_path: str | None = None):
    """Load the FLAN-T5-small (or fine-tuned) model for text generation.
    If loading fails or force_template is True, return (None, None).
    """
    try:
        if force_template:
            st.info("Force Template Mode is ON. Skipping model load.")
            return None, None

        model_name = finetuned_path if (finetuned_path and os.path.exists(finetuned_path)) else "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Device selection
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        model.to(device)
        return tokenizer, model
    except Exception as e:
        # Non-blocking: app can run in template-only mode
        st.warning(f"AI model not available, running in template-only mode. Details: {str(e)}")
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


def post_generation_cleanup(text: str, sender_name: str, include_greeting: bool, include_signature: bool, user_name: str = "Your Name") -> str:
    """Replace placeholders and optionally ensure greeting/signature are present."""
    user_name_placeholder = user_name or "Your Name"

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
# Callbacks / Helpers
# -----------------------------

def set_quick_template(context_val: str, tone_val: str):
    """Set quick template values before widgets render and rerun."""
    st.session_state["context_input"] = context_val
    st.session_state["tone_select"] = tone_val
    st.rerun()

# -----------------------------
# Settings persistence
# -----------------------------

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "app_settings.json")

def load_app_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"user_name": "Your Name", "finetuned_model_path": "", "force_template_mode": False}

def save_app_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        st.warning(f"Failed to save settings: {e}")

# -----------------------------
# Sentiment detection (rule-based stub)
# -----------------------------

def detect_sentiment(email_text: str) -> str:
    """Very lightweight rule-based sentiment detector. Can be replaced by a classifier later."""
    text = (email_text or "").lower()
    urgency = any(w in text for w in ["urgent", "asap", "immediately", "priority", "right away"])
    gratitude = any(w in text for w in ["thank you", "thanks", "appreciate", "grateful"])    
    apology = any(w in text for w in ["sorry", "apologize", "apologies"]) 
    frustration = any(w in text for w in ["disappointed", "frustrated", "angry", "upset"]) 

    if urgency:
        return "Urgent/Demanding"
    if frustration:
        return "Frustrated/Concerned"
    if apology:
        return "Apologetic"
    if gratitude:
        return "Positive/Grateful"
    return "Neutral/Informational"

# -----------------------------
# AI Generation
# -----------------------------

def generate_email_response(original_email: str, tone: str, context: str, tokenizer, model, sender_name: str, detected_sentiment: str | None = None) -> (str, str):
    """Generate email response using FLAN-T5. Returns (response, source)."""
    # If model unavailable, fallback immediately
    if tokenizer is None or model is None:
        response = generate_template_response(original_email, tone, context, sender_name)
        response = enhance_response_with_context(response, original_email, context)
        return response, "Template Fallback"
    sentiment_line = f"Detected Sentiment: {detected_sentiment}\n" if detected_sentiment else ""
    prompt = f"""
{sentiment_line}Tone: {tone}
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


def generate_alternative_responses(original_email: str, tone: str, context: str, tokenizer, model, sender_name: str, detected_sentiment: str | None = None, k: int = 3) -> list[str]:
    """Generate multiple alternative responses using sampling. Falls back to single template variant if model unavailable."""
    if tokenizer is None or model is None:
        base = generate_template_response(original_email, tone, context, sender_name)
        enhanced = enhance_response_with_context(base, original_email, context)
        return [enhanced]

    sentiment_line = f"Detected Sentiment: {detected_sentiment}\n" if detected_sentiment else ""
    prompt = f"""
{sentiment_line}Tone: {tone}
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
                num_return_sequences=k,
                temperature=0.9,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id
            )
        variants = []
        for i in range(len(outputs)):
            text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            text = text.replace(prompt, "").strip()
            if len(text) >= 40:
                variants.append(text)
        if not variants:
            # fallback
            base = generate_template_response(original_email, tone, context, sender_name)
            variants = [enhance_response_with_context(base, original_email, context)]
        return variants
    except Exception:
        base = generate_template_response(original_email, tone, context, sender_name)
        return [enhance_response_with_context(base, original_email, context)]

# -----------------------------
# App
# -----------------------------

def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Enhanced Email Response Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered responses with smart templates, name extraction, and a new UI</p>', unsafe_allow_html=True)

    # Load settings
    settings = load_app_settings()
    st.session_state.setdefault('user_name', settings.get('user_name', 'Your Name'))
    st.session_state.setdefault('finetuned_model_path', settings.get('finetuned_model_path', ''))
    st.session_state.setdefault('force_template_mode', settings.get('force_template_mode', False))

    # Load/Check model
    tokenizer, model = load_model(
        force_template=st.session_state.get('force_template_mode', False),
        finetuned_path=st.session_state.get('finetuned_model_path') or None,
    )

    # Status bar
    status_col1, status_col2, status_col3 = st.columns([1, 1, 1])
    with status_col1:
        if tokenizer is not None and model is not None:
            st.success("Model: FLAN-T5-small ✅")
        else:
            st.warning("Model: Unavailable (Template Mode)")
    with status_col2:
        st.info("Name Extraction: Enabled")
    with status_col3:
        if 'generation_time' in st.session_state:
            st.caption(f"Last generated: {st.session_state.generation_time}")

    # Initialize session_state containers
    st.session_state.setdefault('history', [])

    # Tabs for new UI
    tab_compose, tab_response, tab_history, tab_settings = st.tabs(["📝 Compose", "✨ Response", "📚 History", "⚙️ Settings"])

    with tab_compose:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Original Email")
            original_email = st.text_area(
                "Paste the email you want to respond to:",
                height=350,
                placeholder="""Example:\nHi John,\n\nI hope this email finds you well. I wanted to follow up on our discussion about the marketing campaign. Could we schedule a meeting next week to go over the details?\n\nLooking forward to hearing from you.\n\nBest,\nSarah""",
                key="original_email_text"
            )
        with right:
            st.subheader("Response Settings")

            # Quick Templates FIRST: use callbacks to set state and rerun before widgets are built
            st.markdown("**Quick Templates**")
            qt1, qt2, qt3, qt4 = st.columns(4)
            with qt1:
                st.button("Meeting", key="qt_meeting", on_click=set_quick_template, kwargs={"context_val": "meeting request", "tone_val": "Meeting Request"})
            with qt2:
                st.button("Update", key="qt_update", on_click=set_quick_template, kwargs={"context_val": "project update", "tone_val": "Professional"})
            with qt3:
                st.button("Thank You", key="qt_thanks", on_click=set_quick_template, kwargs={"context_val": "thank you message", "tone_val": "Grateful"})
            with qt4:
                st.button("Apology", key="qt_apology", on_click=set_quick_template, kwargs={"context_val": "apology", "tone_val": "Apologetic"})

            # Now render widgets bound to those keys
            tone = st.selectbox(
                "Select Response Tone:",
                ["Professional", "Casual", "Apologetic", "Grateful", "Urgent", "Follow-up", "Meeting Request"],
                key="tone_select"
            )
            context = st.text_input(
                "Email Context/Subject:",
                placeholder="e.g., meeting request, project update, complaint",
                key="context_input"
            )
            include_greeting = st.checkbox("Include greeting", value=True, key="include_greeting")
            include_signature = st.checkbox("Include signature placeholder", value=True, key="include_signature")

            st.divider()
            generate_clicked = st.button("🚀 Generate Response", type="primary", use_container_width=True)

        if generate_clicked:
            if st.session_state.get('original_email_text', '').strip() and st.session_state.get('context_input', '').strip():
                with st.spinner("Generating your email response..."):
                    original_email_val = st.session_state['original_email_text']
                    tone_val = st.session_state['tone_select']
                    context_val = st.session_state['context_input']
                    sender_name = extract_sender_name(original_email_val)
                    detected_sentiment = detect_sentiment(original_email_val)

                    # Generate
                    response_text, source = generate_email_response(original_email_val, tone_val, context_val, tokenizer, model, sender_name, detected_sentiment)
                    final_text = post_generation_cleanup(
                        response_text,
                        sender_name,
                        st.session_state.get("include_greeting", True),
                        st.session_state.get("include_signature", True),
                        st.session_state.get('user_name', 'Your Name')
                    )

                    # Store session
                    st.session_state.generated_response = final_text
                    st.session_state.generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.response_source = source
                    st.session_state.sender_name = sender_name
                    st.session_state.detected_sentiment = detected_sentiment

                    # Append to history
                    st.session_state.history.append({
                        "time": st.session_state.generation_time,
                        "source": source,
                        "tone": tone_val,
                        "context": context_val,
                        "sender": sender_name,
                        "original": original_email_val,
                        "response": final_text,
                        "sentiment": detected_sentiment
                    })
                st.success("Response generated! Check the 'Response' tab.")
            else:
                st.warning("Please provide both the original email and context.")

    with tab_response:
        st.subheader("Generated Response")
        if 'generated_response' in st.session_state:
            # Source and sentiment badges
            if st.session_state.get("response_source") == "AI Model":
                st.success("✅ Generated by AI Model (FLAN-T5)")
            else:
                st.info("ℹ️ Generated by Template Fallback")
            if st.session_state.get("detected_sentiment"):
                st.caption(f"Detected Sentiment: {st.session_state['detected_sentiment']}")

            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            response_text = st.text_area(
                "Your Generated Response:",
                value=st.session_state.generated_response,
                height=320,
                key="response_area"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Update stored value in case user edits it
            st.session_state.generated_response = response_text

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.download_button(
                    label="📥 Download Response",
                    data=response_text,
                    file_name=f"email_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with c2:
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
            with c3:
                if st.button("✨ Generate Alternatives", use_container_width=True):
                    variants = generate_alternative_responses(
                        st.session_state.get('original_email_text', ''),
                        st.session_state.get('tone_select', 'Professional'),
                        st.session_state.get('context_input', ''),
                        tokenizer, model,
                        st.session_state.get('sender_name', 'there'),
                        st.session_state.get('detected_sentiment')
                    )
                    st.session_state['alt_variants'] = [
                        post_generation_cleanup(v, st.session_state.get('sender_name','there'),
                                                st.session_state.get('include_greeting', True),
                                                st.session_state.get('include_signature', True),
                                                st.session_state.get('user_name','Your Name'))
                        for v in variants
                    ]

            if 'alt_variants' in st.session_state:
                st.markdown("---")
                st.subheader("Alternative Responses")
                for i, v in enumerate(st.session_state['alt_variants'], start=1):
                    with st.expander(f"Variant {i}"):
                        st.code(v)
        else:
            st.info("No response yet. Generate one from the Compose tab.")

    with tab_history:
        st.subheader("Generation History")
        if st.session_state['history']:
            # List simple table-like view
            items = [f"{i+1}. {h['time']} • {h['source']} • {h['tone']} • {h['context']} • {h.get('sentiment','')}" for i, h in enumerate(st.session_state['history'])]
            selected = st.selectbox("Select an entry to view:", options=list(range(len(items))), format_func=lambda i: items[i])

            sel = st.session_state['history'][selected]
            st.markdown(f"**Time:** {sel['time']}  ")
            st.markdown(f"**Source:** {sel['source']}  ")
            st.markdown(f"**Tone:** {sel['tone']}  ")
            st.markdown(f"**Context:** {sel['context']}  ")
            st.markdown(f"**Sender:** {sel['sender']}  ")
            if sel.get('sentiment'):
                st.markdown(f"**Sentiment:** {sel['sentiment']}  ")

            hc1, hc2, hc3 = st.columns([1,1,1])
            with hc1:
                if st.button("Load to Compose"):
                    st.session_state.original_email_text = sel['original']
                    st.session_state.tone_select = sel['tone']
                    st.session_state.context_input = sel['context']
                    st.success("Loaded into Compose tab.")
            with hc2:
                if st.button("Load to Response"):
                    st.session_state.generated_response = sel['response']
                    st.session_state.response_source = sel['source']
                    st.session_state.sender_name = sel['sender']
                    st.session_state.generation_time = sel['time']
                    st.success("Loaded into Response tab.")
            with hc3:
                if st.button("Delete Entry"):
                    st.session_state['history'].pop(selected)
                    st.warning("History entry deleted. Reload the tab to refresh view.")
        else:
            st.info("No history yet. Generate a response to start building history.")

    with tab_settings:
        st.subheader("Application Settings")
        user_name = st.text_input("Your Name (used in signature)", value=st.session_state.get('user_name','Your Name'))
        finetuned_model_path = st.text_input("Fine-tuned model path (optional)", value=st.session_state.get('finetuned_model_path',''))
        force_template_mode = st.checkbox("Force Template Mode (skip AI model)", value=st.session_state.get('force_template_mode', False))

        s1, s2 = st.columns([1,1])
        with s1:
            if st.button("💾 Save Settings", use_container_width=True):
                new_settings = {
                    "user_name": user_name or "Your Name",
                    "finetuned_model_path": finetuned_model_path or "",
                    "force_template_mode": bool(force_template_mode)
                }
                save_app_settings(new_settings)
                # mirror to session
                st.session_state['user_name'] = new_settings['user_name']
                st.session_state['finetuned_model_path'] = new_settings['finetuned_model_path']
                st.session_state['force_template_mode'] = new_settings['force_template_mode']
                st.success("Settings saved. Restart the app or click 'Reload Model' to apply model changes.")
        with s2:
            if st.button("🔁 Reload Model", use_container_width=True):
                # Clear cache and reload
                load_model.clear()
                _tok, _mod = load_model(
                    force_template=st.session_state.get('force_template_mode', False),
                    finetuned_path=st.session_state.get('finetuned_model_path') or None,
                )
                st.success("Model reload requested. Navigate to Compose to generate.")


if __name__ == "__main__":
    main()
