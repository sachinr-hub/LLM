# 📧 Email Response Generator

A powerful AI-powered email response generator built with Streamlit and Hugging Face Transformers. Generate professional email responses in seconds with different tones and styles.

![Email Response Generator](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=for-the-badge)

## ✨ Features

- 🤖 **AI-Powered Responses**: Uses Google's FLAN-T5 model locally (with optional fine-tuned checkpoint)
- 🎯 **Seven Tones**: Professional, Casual, Apologetic, Grateful, Urgent, Follow-up, Meeting Request
- 🧠 **ML Sentiment Detection**: DistilBERT-based sentiment with rule-based fallback; injected into prompts
- 📚 **Optional RAG**: TF-IDF over `kb/*.txt` to inject relevant context into the prompt
- 📝 **Context-Aware**: Name extraction, tone, context, and sentiment-aware prompting
- 🧩 **Alternative Variants**: Generate multiple response variants
- 🕓 **History**: View, reload, and manage previously generated responses
- ⚙️ **Settings**: Persist "Your Name", fine-tuned model path, template-only mode, FP16, and RAG settings
- 💾 **Download & Copy**: Download responses and copy to clipboard
- 🎨 **Modern UI**: Tabbed interface (Compose, Response, History, Settings)

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
   streamlit run enhanced_email_generator.py
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
  - **Professional**, **Casual**, **Apologetic**, **Grateful**, **Urgent**, **Follow-up**, **Meeting Request**
  - Use Quick Templates to prefill context + tone

### Step 3: Generate Response
- Click "🚀 Generate Response"
- Wait a few seconds for AI processing
- Review and edit the generated response

### Step 4: Use Your Response
- Copy the response to your email client
- Download as a text file for later use
- Edit as needed before sending

### Settings (⚙️ tab)
- Set your preferred signature name in "Your Name"
- Optionally provide a local fine-tuned model path (e.g., `models/flan-t5-finetuned`)
- Toggle Template Mode (skip AI), FP16 on GPU, and RAG with Top-K control
- Click "Save Settings" to persist to `app_settings.json` and "Reload Model" to apply model changes

### RAG (Optional)
- Place domain text files under `kb/` as `.txt` files
- Enable RAG in Settings and set Top-K
- The app will retrieve relevant snippets and inject them in the prompt under "Relevant Context"

### Sentiment
- The app uses a DistilBERT pipeline when available, otherwise falls back to rule-based detection
- Detected sentiment is shown in the Response tab and included in the prompt

### Alternative Variants
- Use "✨ Generate Alternatives" to produce multiple responses with different sampling

### History
- The History tab lists prior generations with time, tone, sentiment, and context
- Load items back into Compose or Response, or delete entries

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
- **Model**: Google FLAN-T5 Small (or optional fine-tuned checkpoint)
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
- `scikit-learn`: TF-IDF and cosine similarity for RAG and evaluation
- `datasets` (for fine-tuning script)
- Optional: `bert-score` for evaluation

## 🧪 Training and Evaluation

### Fine-tuning FLAN-T5
- Script: `scripts/train_t5_finetune.py`
- CSV schema: `original_email, sentiment, tone, context, target_response`
- Example:
```bash
python scripts/train_t5_finetune.py \
  --train_csv data/train.csv \
  --output_dir models/flan-t5-finetuned \
  --epochs 3 --batch_size 4 --fp16
```
- Point the app Settings → Fine-tuned model path to `models/flan-t5-finetuned`

### Evaluation
- Script: `scripts/evaluate_generation.py`
- CSV schema: `pred, ref`
- TF-IDF cosine similarity:
```bash
python scripts/evaluate_generation.py --pairs_csv data/pairs.csv
```
- BERTScore (install optional dependency first):
```bash
pip install bert-score
python scripts/evaluate_generation.py --pairs_csv data/pairs.csv --use_bertscore
```

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
   - Set main file: `enhanced_email_generator.py`
   - Click "Deploy"

3. **Your app will be live** at `https://your-app-name.streamlit.app`

### Local Network Access

To access from other devices on your network:
```bash
streamlit run enhanced_email_generator.py --server.address 0.0.0.0
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
