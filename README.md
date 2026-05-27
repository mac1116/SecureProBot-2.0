# SecureProBot — AI Twitter/X Bot Detection Platform v2.1

AI-powered bot detection using hybrid LSTM + Ensemble models to analyze Twitter/X accounts with 94.42% accuracy.

## Features

- 🤖 **Hybrid AI Model**: Combines LSTM feature extraction with Ensemble classification
- 📊 **55+ Behavioral Signals**: Analyzes metadata, account age, posting patterns, follower ratios
- 🎯 **High Accuracy**: 94.42% validation accuracy, 0.9882 AUC-ROC
- 🔍 **Real-time Analysis**: Instant bot detection via web UI or REST API
- 🔐 **Privacy-First**: No data storage, analysis runs locally
- 📱 **Responsive UI**: Modern dark/light theme with real-time insights

## Quick Start

### Local Development

1. **Clone repository**
   ```bash
   git clone https://github.com/mac1116/SecureProBot-2.0.git
   cd SecureProBot-2.0
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional - for Twitter API integration)
   ```bash
   cp .env.example .env
   # Edit .env and add your TWITTER_BEARER_TOKEN
   ```

5. **Run the app**
   ```bash
   python SecureProBotApp.py
   ```

   Visit http://127.0.0.1:5000 in your browser

## Deployment to Render

### Prerequisites
- GitHub account with SecureProBot repository pushed
- Render account (free at https://render.com)

### Step-by-Step Deployment

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Click "Create +" → "Web Service"

2. **Connect GitHub Repository**
   - Select "GitHub"
   - Authorize Render to access your GitHub
   - Select `SecureProBot-2.0` repository
   - Click "Connect"

3. **Configure Build & Deploy**
   - **Name**: `secureprobot` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 4 -b 0.0.0.0:$PORT SecureProBotApp:app`
   - **Plan**: Free tier available (with some limitations)

4. **Environment Variables** (if using Twitter API)
   - Click "Advanced"
   - Add environment variable:
     - **Key**: `TWITTER_BEARER_TOKEN`
     - **Value**: Your GetXAPI bearer token
     - **Key**: `FLASK_SECRET_KEY`
     - **Value**: Any random string

5. **Deploy**
   - Click "Create Web Service"
   - Render will start building and deploying
   - Wait for "Live" status (2-3 minutes)
   - Your app URL: `https://secureprobot.onrender.com`

### Important Notes for Render

- **Free tier**: 0.5GB RAM, auto-spins down after 15 min inactivity
- **Cold starts**: First request takes 10-30 seconds (normal)
- **Model loading**: TensorFlow + scikit-learn add ~15s to first startup
- **Models not included**: Git-ignored `.joblib` and `.keras` files need to be:
  - Downloaded/generated during build, OR
  - Stored in a separate location and fetched at runtime

### Alternative: AWS/Azure/Google Cloud

For production with better performance:
- AWS: EC2, Lambda, or Elastic Beanstalk
- Azure: App Service or Container Instances
- Google Cloud: Cloud Run or App Engine

## API Documentation

### Endpoints

#### 1. **POST /analyze** — Analyze account with manual data
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "screen_name": "example_user",
    "name": "Example User",
    "followers_count": 1000,
    "friends_count": 500,
    "statuses_count": 5000,
    "verified": false
  }'
```

**Response**:
```json
{
  "label": "Human",
  "confidence": 87.5,
  "bot_probability": 12.5,
  "human_probability": 87.5,
  "authenticity_score": 88,
  "risk_level": "Low Risk",
  "insights": [...],
  "behaviors": [...]
}
```

#### 2. **POST /analyze-twitter-api** — Fetch & analyze from Twitter/GetXAPI
```bash
curl -X POST http://localhost:5000/analyze-twitter-api \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://twitter.com/elonmusk",
    "debug": false
  }'
```

#### 3. **POST /analyze-link** — Parse Twitter URL & analyze
```bash
curl -X POST http://localhost:5000/analyze-link \
  -H "Content-Type: application/json" \
  -d '{"url": "@elonmusk"}'
```

#### 4. **GET /health** — Health check
```bash
curl http://localhost:5000/health
```

## Project Structure

```
SecureProBot-2.0/
├── SecureProBotApp.py          # Main Flask application
├── SecureProBot_Complete.ipynb # Training notebook
├── requirements.txt            # Python dependencies
├── Procfile                    # Render deployment config
├── .gitignore                  # Git ignore patterns
├── models/                     # Model files (git-ignored)
│   ├── lstm_model.keras
│   ├── lstm_feature_extractor.keras
│   ├── ensemble_clf.joblib
│   ├── scaler.joblib
│   ├── tokenizer.joblib
│   ├── model_config.json
│   └── scaler_stats.json
└── secureprobot_prototype/     # Backup models
```

## Model Architecture

### Feature Pipeline
1. **LSTM Feature Extractor** (32-dimensional output)
   - Processes tweet/bio text sequences
   - Captures linguistic patterns and temporal dynamics

2. **Metadata Features** (23 features)
   - Account age, follower/friend ratios, verification status
   - Posted frequency, name entropy, description completeness
   - Growth rates, listed count, favorites ratio

3. **Ensemble Classifier** (Hybrid)
   - Random Forest + Gradient Boosting
   - Combines LSTM + metadata for final prediction

### Performance Metrics
- **Validation Accuracy**: 94.42%
- **Cross-Validation AUC**: 0.9929 ± 0.0028
- **ROC-AUC**: 0.9882
- **Precision/Recall**: Optimized for low false positives

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TWITTER_BEARER_TOKEN` | GetXAPI bearer token | Optional (for API features) |
| `FLASK_SECRET_KEY` | Flask session encryption | Optional (auto-generated if missing) |
| `FLASK_ENV` | `development` or `production` | Optional (default: production) |
| `PORT` | Server port (set by Render) | Auto (default: 5000) |

## Troubleshooting

### Models Not Loading
```
⚠ No model found — using heuristic engine
```
**Solution**: Ensure model files are in `models/` directory. Download from the notebook or previous exports.

### Slow on Render Free Tier
**Cause**: 0.5GB RAM, TensorFlow startup overhead
**Solution**: 
- Upgrade to paid plan
- Optimize model size
- Use model quantization

### API Returns 401 Unauthorized
**Cause**: Invalid or missing `TWITTER_BEARER_TOKEN`
**Solution**: Add valid GetXAPI token to `.env` or Render environment variables

### Port Already in Use
```
Address already in use
```
**Solution**: 
```bash
kill -9 $(lsof -t -i:5000)  # macOS/Linux
netstat -ano | findstr :5000  # Windows (then taskkill /PID <PID> /F)
```

## Technologies Used

- **Backend**: Flask, Scikit-learn, TensorFlow, joblib
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **APIs**: Twitter/X (via GetXAPI), REST
- **Deployment**: Render, GitHub
- **ML**: LSTM, Random Forest, Gradient Boosting

## Team

- **Joevan B. Capote** — Backend Development, ML Pipeline
- **Mheil Andrei N. Cenita** — Full Stack Development, UI/UX Design

University of Mindanao

## License

Research and educational use only. See LICENSE file.

## Support & Contact

- 📧 **Issues**: GitHub Issues
- 🐦 **Twitter**: [@SecureProBot](https://twitter.com/secureprobot)
- 💬 **Email**: secureprobot@example.com

---

**Version 2.1** — Built with ❤️ for research and platform safety
