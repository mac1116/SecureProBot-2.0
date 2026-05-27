# SecureProBot — AI Twitter/X Bot Detection Platform

AI-powered bot detection using hybrid LSTM + Ensemble models to detect fake Twitter/X accounts with 94.42% accuracy.

## Features

- 🤖 **Hybrid AI Model**: LSTM feature extraction + Ensemble classification
- 📊 **55+ Behavioral Signals**: Account metadata, posting patterns, follower analysis
- 🎯 **94.42% Accuracy**: 0.9929 ± 0.0028 cross-validation AUC-ROC
- 🔍 **Real-time Analysis**: Web UI and REST API for instant bot detection
- 🔐 **Privacy-First**: No data storage, local analysis
- 📱 **Responsive UI**: Dark/light theme with real-time insights

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation & Run

```bash
# Clone repository
git clone https://github.com/mac1116/SecureProBot-2.0.git
cd SecureProBot-2.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python SecureProBotApp.py
```

Visit **http://127.0.0.1:5000** in your browser

### Optional: Twitter API Integration

```bash
cp .env.example .env
# Edit .env and add your TWITTER_BEARER_TOKEN from GetXAPI
```

## API Endpoints

### POST /analyze
Analyze account with manual data

```json
{
  "screen_name": "example_user",
  "followers_count": 1000,
  "friends_count": 500,
  "statuses_count": 5000,
  "verified": false
}
```

**Response:**
```json
{
  "label": "Human",
  "confidence": 87.5,
  "bot_probability": 12.5,
  "authenticity_score": 88,
  "risk_level": "Low Risk"
}
```

### POST /analyze-twitter-api
Fetch & analyze from Twitter/GetXAPI

```json
{
  "url": "https://twitter.com/username",
  "debug": false
}
```

### POST /analyze-link
Parse Twitter URL & analyze

```json
{
  "url": "@username"
}
```

### GET /health
Service health check


## Project Structure

```
SecureProBot-2.0/
├── SecureProBotApp.py              # Main Flask app
├── SecureProBot_Complete.ipynb     # Model training notebook
├── requirements.txt                # Dependencies
├── Dockerfile                      # Docker config
├── models/                         # Trained models
│   ├── lstm_model.keras
│   ├── lstm_feature_extractor.keras
│   ├── ensemble_clf.joblib
│   └── model_config.json
└── README.md
```

## Model Performance

- **Validation Accuracy**: 94.42%
- **AUC-ROC**: 0.9882
- **Cross-Validation**: 0.9929 ± 0.0028

## Technologies

- **Backend**: Flask, Scikit-learn, TensorFlow
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: LSTM RNN, Random Forest, Gradient Boosting

## Team

- **Joevan B. Capote** — Backend & ML
- **Mheil Andrei N. Cenita** — FullStack & UI/UX

University of Mindanao

## License

Research and educational use only.


