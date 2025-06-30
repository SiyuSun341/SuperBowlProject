## Model Selection Decision and Comparison

The following table outlines the model selection process, including candidate comparisons and the rationale behind the final choice.

### Text Sentiment Classification Models

| Model | Type | Platform / Execution | Justification |
|-------|------|---------------------|---------------|
| **TextBlob** | Lexicon-based | Local Python / CPU | Easy to use, intuitive polarity scoring, suitable for baseline analysis |
| **VADER** | Lexicon-based | Local Python / CPU | Designed specifically for short, informal social media text |
| **RoBERTa** | Transformer model | HuggingFace on local GPU | High performance on short-form sentiment tasks in social contexts |
| **FinBERT** | Domain-adapted | HuggingFace on local GPU | Fine-tuned for financial and business contexts, better interpretation of ads |
| **BART** | Generative model | HuggingFace on local GPU | Captures complex structures and rhetorical tones, complements classification gaps |

### Multimodal Video Content Models

| Model | Type | Platform / Execution | Justification |
|-------|------|---------------------|---------------|
| **Whisper (medium)** | Speech-to-text (audio-to-subtitle) | Local inference (OpenAI Whisper) | Balanced accuracy and speed; medium model is efficient for large-scale transcription |
| **FFmpeg** | Video preprocessing | Local execution via script | Efficient extraction of audio, frames, and consistent file formatting |
| **Gemini API** | Multimodal scene analysis | Google Gemini Pro + Python script | Capable of processing keyframes, subtitles, and audio to extract themes, brand tone, and emotional signals |

### Emotion Prediction Machine Learning Models

| Model | Type | Platform / Execution | Justification |
|-------|------|---------------------|---------------|
| **Logistic Regression** | Linear model | scikit-learn / local | Interpretable baseline model |
| **Random Forest** | Tree ensemble | scikit-learn / local | Handles non-linear feature combinations well |
| **SVM** | Kernel-based model | scikit-learn / local | Effective in high-dimensional text spaces |
| **Naive Bayes** | Probabilistic model | scikit-learn / local | Fast, scalable for discrete text features |
| **KNN** | Distance-based | scikit-learn / local | Good for exploratory and small-sample scenarios |
| **MLP (Neural Network)** | Deep learning model | PyTorch / local GPU | Captures high-order non-linear interactions in sentiment features |
| **CatBoost** | Gradient boosting | CatBoost / local | Strong performance on categorical and imbalanced data, fast training speed |
