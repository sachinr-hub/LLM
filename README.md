# 📧 Email Response Generator

A powerful AI-powered email response generator built with Streamlit and Hugging Face Transformers. Generate professional email responses in seconds with different tones and styles.

![Email Response Generator](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=for-the-badge)

## ✨ Features

- 🤖 **AI-Powered Responses**: Uses Google's FLAN-T5 model for intelligent email generation
- 🎯 **Multiple Tones**: Professional, Casual, Apologetic, Grateful, and Urgent
- 📝 **Context-Aware**: Generates responses based on email content and context
- 💾 **Download Responses**: Save generated emails as text files
- 🎨 **Beautiful UI**: Modern, responsive Streamlit interface
- 🆓 **100% Free**: No API keys or subscriptions required
- ⚡ **Fast Generation**: Quick response times with local processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this project**
   ```bash
   git clone <your-repo-url>
   cd email-response-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run email_response_generator.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in your terminal

## 📱 How to Use

### Step 1: Input Original Email
- Paste the email you want to respond to in the left panel
- Include the complete email with sender information

### Step 2: Set Context and Tone
- **Context**: Briefly describe what the email is about (e.g., "meeting request", "project update")
- **Tone**: Choose from:
  - **Professional**: Formal business communication
  - **Casual**: Friendly, informal responses
  - **Apologetic**: For mistakes or delays
  - **Grateful**: Appreciative and thankful
  - **Urgent**: Time-sensitive matters

### Step 3: Generate Response
- Click "🚀 Generate Response"
- Wait a few seconds for AI processing
- Review and edit the generated response

### Step 4: Use Your Response
- Copy the response to your email client
- Download as a text file for later use
- Edit as needed before sending

## 💡 Tips for Best Results

### Original Email Input
- ✅ Include the complete email
- ✅ Keep sender's name and context
- ✅ Don't remove important details
- ❌ Don't paste just fragments

### Context Description
- ✅ Be specific about the topic
- ✅ Mention urgency if needed
- ✅ Include key keywords
- ❌ Don't be too vague

### Tone Selection
- **Professional**: Client emails, formal requests, business proposals
- **Casual**: Team members, friends, informal check-ins
- **Apologetic**: Delays, mistakes, service issues
- **Grateful**: Thank you messages, appreciation emails
- **Urgent**: Time-sensitive requests, emergency communications

## 🛠️ Technical Details

### AI Model
- **Model**: Google FLAN-T5 Small
- **Size**: ~250MB download
- **Performance**: Optimized for speed and quality
- **Offline**: Works without internet after initial setup

### System Requirements
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB for model and dependencies
- **CPU**: Any modern processor (GPU not required)

### Dependencies
- `streamlit`: Web application framework
- `transformers`: Hugging Face model library
- `torch`: PyTorch for model inference
- `tokenizers`: Text tokenization
- `sentencepiece`: Text processing

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file: `email_response_generator.py`
   - Click "Deploy"

3. **Your app will be live** at `https://your-app-name.streamlit.app`

### Local Network Access

To access from other devices on your network:
```bash
streamlit run email_response_generator.py --server.address 0.0.0.0
```

## 🔧 Customization

### Adding New Tones
Edit the `tone_instructions` dictionary in `generate_email_response()`:
```python
tone_instructions = {
    "Professional": "Write a professional and formal email response",
    "Your_New_Tone": "Your custom instruction here"
}
```

### Changing the AI Model
Replace the model name in `load_model()`:
```python
model_name = "google/flan-t5-base"  # Larger model for better quality
```

### Custom Templates
Add new templates in `generate_template_response()` function.

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error**
- Check internet connection for first-time download
- Ensure sufficient disk space (1GB+)
- Try restarting the application

**Slow Performance**
- Use FLAN-T5 Small instead of Base/Large
- Close other applications to free RAM
- Consider using GPU if available

**Import Errors**
- Reinstall requirements: `pip install -r requirements.txt --upgrade`
- Check Python version (3.8+ required)
- Use virtual environment to avoid conflicts

### Getting Help

1. Check the terminal for error messages
2. Ensure all dependencies are installed
3. Try restarting the Streamlit server
4. Check GitHub issues for similar problems

## 📈 Future Enhancements

- [ ] Email thread analysis
- [ ] Multiple language support
- [ ] Custom signature integration
- [ ] Email scheduling suggestions
- [ ] Sentiment analysis
- [ ] Template library expansion
- [ ] Integration with email clients

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the amazing Transformers library
- [Streamlit](https://streamlit.io/) for the fantastic web app framework
- [Google](https://ai.google/) for the FLAN-T5 model

---

**Made with ❤️ and AI**

*Generate professional email responses in seconds!*
