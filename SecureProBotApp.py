"""
SecureProBot — AI-Powered Twitter/X Bot Detection Platform  v2.1
Flask single-file application with embedded HTML/CSS/JS

Changes from v2.0:
  • Loads the new Hybrid LSTM + Ensemble model exported from the notebook
  • Computes the correct 23-dim METADATA_COLS feature vector
  • Adds GetXAPI integration (/analyze-twitter-api route)

Run:  python SecureProBotApp.py
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import math
import random
import re
import os
import time
import json
from pathlib import Path
from collections import Counter

# ── Load .env file ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
# ─────────────────────────────────────────────────────────────────────────────

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import requests as http_req
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================================
# Paths
# ============================================================================

WORKSPACE_DIR = Path(__file__).parent
MODELS_DIR    = WORKSPACE_DIR / "models"
BUNDLE_PATH   = MODELS_DIR / "secureprobot_multi.pkl"
EXTRACTOR_PATH= MODELS_DIR / "lstm_feature_extractor.keras"
CONFIG_PATH   = MODELS_DIR / "model_config.json"

# ============================================================================
# METADATA_COLS  — must match exactly what the notebook used during training
# ============================================================================
METADATA_COLS = [
    'statuses_count', 'followers_count', 'friends_count', 'favourites_count',
    'listed_count', 'verified', 'description_length', 'description_entropy',
    'screen_name_entropy', 'tweet_freq', 'followers_friends_ratio', 'has_description',
    'user_age', 'followers_growth_rate', 'friends_growth_rate',
    'favourites_growth_rate', 'listed_growth_rate',
    'name_entropy', 'num_digits_in_name', 'num_digits_in_screen_name',
    'screen_name_freq', 'name_sim', 'default_profile',
]

MAX_LEN = 30   # overridden by model_config.json if present

# ============================================================================
# Load model config (populated at startup)
# ============================================================================

_config: dict = {}

def load_config():
    global _config, MAX_LEN
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                _config = json.load(f)
            MAX_LEN = _config.get('max_len', 30)
        except Exception:
            pass

load_config()

# ============================================================================
# Model + Keras extractor cache
# ============================================================================

_bundle        = None   # sklearn bundle
_extractor     = None   # tf.keras feature extractor
_model_type    = None   # 'hybrid' | 'legacy' | None

def load_model_artifacts():
    """Load joblib bundle + Keras feature extractor once."""
    global _bundle, _extractor, _model_type

    if _bundle is not None:
        return _bundle, _extractor

    if not JOBLIB_AVAILABLE or not BUNDLE_PATH.exists():
        return None, None

    try:
        _bundle = joblib.load(BUNDLE_PATH)
    except Exception as e:
        print(f"  ✗ Failed to load bundle: {e}")
        return None, None

    # Detect model type
    _model_type = _bundle.get('model_type', 'legacy')

    # Load Keras feature extractor for hybrid models
    if 'hybrid' in str(_model_type) and TF_AVAILABLE and EXTRACTOR_PATH.exists():
        try:
            _extractor = tf.keras.models.load_model(str(EXTRACTOR_PATH), compile=False)
            print(f"  ✓ Keras feature extractor loaded ({_extractor.output_shape})")
        except Exception as e:
            print(f"  ⚠ Keras extractor load failed: {e}")
            _extractor = None

    return _bundle, _extractor


# ============================================================================
# Feature Engineering  — produces METADATA_COLS (23-dim)
# ============================================================================

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    probs  = np.array(list(counts.values()), dtype=float) / len(text)
    ent    = -np.sum(probs * np.log2(probs + 1e-10))
    max_e  = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return float(ent / max_e) if max_e > 0 else 0.0

def _bigram_freq(s: str) -> float:
    s = str(s)
    if len(s) < 2:
        return 0.0
    bigrams = [s[i:i+2] for i in range(len(s) - 1)]
    counts  = Counter(bigrams)
    return sum(counts.values()) / len(counts) if counts else 0.0

def _name_sim(screen_name: str, display_name: str) -> float:
    sn = screen_name.lower()
    n  = display_name.lower().replace(' ', '')
    if not sn or not n:
        return 0.0
    matches = sum(c in n for c in sn)
    return 2 * matches / (len(sn) + len(n)) if (len(sn) + len(n)) > 0 else 0.0

def extract_metadata_features(account: dict) -> np.ndarray:
    """
    Build the 23-dim METADATA_COLS vector from an account dict.

    Expected keys (all optional, sensible defaults applied):
        screen_name, name, description, verified,
        followers_count, friends_count (= following), statuses_count,
        favourites_count, listed_count, user_age (days),
        default_profile
    """
    # Raw fields
    sc_name     = str(account.get('screen_name', '') or '')
    disp_name   = str(account.get('name', '')        or '')
    description = str(account.get('description', '') or '')
    verified    = 1 if account.get('verified', False) else 0
    followers   = max(0, int(account.get('followers_count', 0)  or 0))
    friends     = max(0, int(account.get('friends_count',   0)  or 0))   # "following"
    statuses    = max(0, int(account.get('statuses_count',  0)  or 0))
    favourites  = max(0, int(account.get('favourites_count',0)  or 0))
    listed      = max(0, int(account.get('listed_count',    0)  or 0))
    user_age    = max(1, int(account.get('user_age',        365) or 365))
    default_p   = 1 if account.get('default_profile', False) else 0

    # Derived
    tweet_freq             = statuses  / user_age
    followers_growth_rate  = followers / user_age
    friends_growth_rate    = friends   / user_age
    favourites_growth_rate = favourites / user_age
    listed_growth_rate     = listed    / user_age
    ff_ratio               = followers / friends if friends > 0 else 0.0
    desc_len               = len(description)
    desc_entropy           = _entropy(description)
    sc_entropy             = _entropy(sc_name)
    has_desc               = 1 if desc_len > 0 else 0
    name_entropy           = _entropy(disp_name)
    num_digits_name        = sum(c.isdigit() for c in disp_name)
    num_digits_sc          = sum(c.isdigit() for c in sc_name)
    sc_freq                = _bigram_freq(sc_name)
    n_sim                  = _name_sim(sc_name, disp_name)

    row = [
        statuses,                # statuses_count
        followers,               # followers_count
        friends,                 # friends_count
        favourites,              # favourites_count
        listed,                  # listed_count
        verified,                # verified
        desc_len,                # description_length
        desc_entropy,            # description_entropy
        sc_entropy,              # screen_name_entropy
        tweet_freq,              # tweet_freq
        ff_ratio,                # followers_friends_ratio
        has_desc,                # has_description
        user_age,                # user_age
        followers_growth_rate,   # followers_growth_rate
        friends_growth_rate,     # friends_growth_rate
        favourites_growth_rate,  # favourites_growth_rate
        listed_growth_rate,      # listed_growth_rate
        name_entropy,            # name_entropy
        num_digits_name,         # num_digits_in_name
        num_digits_sc,           # num_digits_in_screen_name
        sc_freq,                 # screen_name_freq
        n_sim,                   # name_sim
        default_p,               # default_profile
    ]
    arr = np.array(row, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.reshape(1, -1)   # (1, 23)


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_lstm_features(text: str, tokenizer, extractor) -> np.ndarray:
    """Return (1, 32) LSTM features from description text."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([_clean_text(text)])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return extractor.predict(padded, verbose=0)   # (1, 32)


# ============================================================================
# Legacy fallback: simple 16-dim + 50-dim GloVe (if old model loaded)
# ============================================================================

_glove = None

def _load_glove():
    global _glove
    if _glove is not None:
        return _glove
    for path in [WORKSPACE_DIR / "glove.twitter.27B.50d.txt"]:
        if path.exists():
            _glove = {}
            with open(path, encoding='utf8') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 50:
                        _glove[parts[0]] = np.array(parts[1:51], dtype='float32')
            return _glove
    return {}

def _bio_to_glove(text: str) -> np.ndarray:
    glove = _load_glove()
    if not glove or not text:
        return np.zeros(50)
    vecs = [glove[w] for w in text.lower().split() if w in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(50)

def _extract_legacy_features(account: dict) -> np.ndarray:
    """16-dim feature vector for old model format."""
    sc   = account.get('screen_name', '')
    f    = max(1, account.get('followers_count', 1))
    fw   = max(1, account.get('friends_count', account.get('following_count', 1)))
    st   = max(1, account.get('statuses_count', 1))
    fav  = max(1, account.get('favourites_count', 0))
    age  = max(1, account.get('user_age', account.get('account_age_days', 365)))
    desc = str(account.get('description', '') or '')
    feat = [
        1 if account.get('verified', False) else 0,
        np.log1p(f),
        np.log1p(st),
        np.log1p(fav),
        len(desc),
        _entropy(desc),
        _bigram_freq(sc),
        1 if 'http' in desc or 'www' in desc else 0,
        desc.count('http') + desc.count('www'),
        _entropy(sc),
        sum(c.isdigit() for c in sc) / max(len(sc), 1),
        sum(not c.isalnum() for c in sc) / max(len(sc), 1),
        np.log1p(fw),
        f / fw,
        st / age,
        np.log1p(age),
    ]
    return np.array(feat).reshape(1, -1)


# ============================================================================
# Rule-based fallback
# ============================================================================

def _rule_based(account: dict) -> tuple:
    sc    = account.get('screen_name', '')
    f     = account.get('followers_count', 0)
    fw    = account.get('friends_count', account.get('following_count', 1)) or 1
    st    = account.get('statuses_count', 0)
    desc  = str(account.get('description', '') or '')
    age   = max(account.get('user_age', account.get('account_age_days', 365)), 1)
    vrf   = account.get('verified', False)

    score = 0.0

    # ── Verification ──────────────────────────────────────────────────────────
    if vrf:
        score -= 35

    # ── Username digit ratio ──────────────────────────────────────────────────
    digit_r = sum(c.isdigit() for c in sc) / max(len(sc), 1)
    score  += digit_r * 40

    # ── Follower / Following ratio (FIX 1) ───────────────────────────────────
    # Real influencers have very high ratios; bots tend to follow many but gain few
    ff_r = f / max(fw, 1)
    if ff_r < 0.05:
        score += 30           # follows many, almost no one follows back -> strong bot signal
    elif ff_r < 0.5:
        score += 10           # low ratio, mildly suspicious
    elif ff_r > 10_000:
        score -= 50           # massive organic following -> very likely human
    elif ff_r > 1_000:
        score -= 40           # large organic following -> strong human signal
    elif ff_r > 100:
        score -= 25           # healthy ratio -> human signal
    elif ff_r > 10:
        score -= 10           # above average -> slight human lean

    # ── Tweet frequency ───────────────────────────────────────────────────────
    tpd = st / age
    if tpd > 72:
        score += 30           # 72+ tweets/day -> very likely automated
    elif tpd > 20:
        score += 10
    elif 0.1 <= tpd <= 15:
        score -= 5            # healthy posting rate -> human signal

    # ── Bio / description ─────────────────────────────────────────────────────
    if not desc or len(desc) < 15:
        score += 15
    else:
        ent = _entropy(desc)
        if ent < 2.0:
            score += 20       # very repetitive bio -> bot signal
        elif ent > 3.5:
            score -= 10       # rich, varied bio -> human signal

    # ── Account age (FIX 3) ──────────────────────────────────────────────────
    if age < 30:
        # Brand new — but if activity is also low, it's probably just a new real user
        if st < 50 and fw < 200:
            score += 8   # mild suspicion only
        else:
            score += 20  # new AND already posting/following a lot → suspicious
    elif age < 90:
        score += 5


    # ── High follower count absolute floor ───────────────────────────────────
    if f > 500_000:
        score -= 30
    elif f > 50_000:
        score -= 15

    bot_prob = float(min(max(score / 100.0, 0.02), 0.97))
    return bot_prob, 1.0 - bot_prob


# ============================================================================
# GetXAPI fetcher
# ============================================================================

GETXAPI_BASE = os.getenv("GETXAPI_BASE", "https://api.getxapi.com")

def fetch_twitter_account(username: str, bearer_token: str) -> dict:
    """
    Fetch account data via GetXAPI and normalise it
    into the dict format expected by extract_metadata_features().
    Raises ValueError with a human-readable message on failure.
    """
    if not REQUESTS_AVAILABLE:
        raise ValueError("'requests' library not installed — run: pip install requests")

    headers = {"Authorization": f"Bearer {bearer_token.strip()}"}
    url     = f"{GETXAPI_BASE}/twitter/user/info"
    params  = {"userName": username.lstrip('@')}

    try:
        resp = http_req.get(url, headers=headers, params=params, timeout=10)
    except http_req.exceptions.ConnectionError:
        raise ValueError("Network error — could not reach api.getxapi.com")
    except http_req.exceptions.Timeout:
        raise ValueError("GetXAPI timed out — try again")

    if resp.status_code == 401:
        raise ValueError("Invalid or expired API Key — check your .env file")
    if resp.status_code == 403:
        raise ValueError("Access denied — check your GetXAPI key")
    if resp.status_code == 429:
        raise ValueError("Rate limit exceeded — wait a moment and retry")
    if resp.status_code == 404:
        raise ValueError(f"User @{username} not found")
    if resp.status_code == 402:
        raise ValueError("Insufficient GetXAPI credits — top up at getxapi.com")
    if not resp.ok:
        raise ValueError(f"GetXAPI error {resp.status_code}")

    body = resp.json()
    if body.get('status') != 'success' or 'data' not in body:
        detail = body.get('error', body.get('msg', 'Unknown error from GetXAPI'))
        raise ValueError(detail)

    user = body['data']
    if not isinstance(user, dict) or not user:
        raise ValueError(f"User @{username} not found")

    user_name = user.get('userName') or user.get('username')
    user_id = user.get('id') or user.get('userId')
    if not user_name and not user_id:
        raise ValueError(f"User @{username} not found")

    # Account age in days
    created_at = user.get('createdAt', '')
    if created_at:
        from datetime import datetime, timezone
        try:
            created  = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            age_days = max(1, (datetime.now(timezone.utc) - created).days)
        except Exception:
            age_days = 365
    else:
        age_days = 365

    profile_url  = user.get('profilePicture', '')
    default_prof = 'default_profile_images' in profile_url or not profile_url

    return {
        "screen_name":      user.get('userName', username),
        "name":             user.get('name', ''),
        "description":      user.get('description', ''),
        "verified":         bool(user.get('isVerified', False) or user.get('isBlueVerified', False)),
        "followers_count":  int(user.get('followers', 0)       or 0),
        "friends_count":    int(user.get('following', 0)       or 0),
        "statuses_count":   int(user.get('statusesCount', 0)   or 0),
        "favourites_count": int(user.get('favouritesCount', 0) or 0),
        "listed_count":     int(user.get('listedCount', user.get('listed_count', 0)) or 0),
        "user_age":         age_days,
        "default_profile":  default_prof,
        "_created_at":      created_at,
        "_profile_image":   profile_url,
        "_raw_metrics":     user,
    }


# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)
_secret = os.getenv("FLASK_SECRET_KEY", "")
if not _secret:
    import warnings
    warnings.warn("FLASK_SECRET_KEY not set — using random key. Sessions will reset on restart.")
    _secret = os.urandom(24).hex()
app.secret_key = _secret


# ============================================================================
# Core Analysis Logic
# ============================================================================

def generate_insights(account: dict, bot_prob: float) -> list:
    insights  = []
    sc        = account.get('screen_name', '')
    followers = account.get('followers_count', 0)
    friends   = account.get('friends_count', account.get('following_count', 1)) or 1
    statuses  = account.get('statuses_count', 0)
    desc      = str(account.get('description', '') or '')
    verified  = account.get('verified', False)
    age       = max(account.get('user_age', account.get('account_age_days', 365)), 1)
    ff_r      = followers / max(friends, 1)
    listed    = account.get('listed_count', 0)
    fav       = account.get('favourites_count', 0)
    default_profile = account.get('default_profile', False)
    tpd       = statuses / age
    digit_r   = sum(c.isdigit() for c in sc) / max(len(sc), 1)

    # ── Verification ─────────────────────────────────────────────────────
    if verified:
        insights.append({"icon": "✓", "type": "positive",
                         "text": f"✓ Verified — 100% trust boost, significantly reduces bot likelihood"})
    else:
        insights.append({"icon": "⚠", "type": "warning",
                         "text": f"⚠ Not verified (0%) — common for new/bot accounts"})

    # ── Default Profile ──────────────────────────────────────────────────
    if default_profile:
        insights.append({"icon": "⚡", "type": "danger",
                         "text": "⚡ Default profile image — 80% correlated with bot accounts"})

    # ── Username digit ratio ──────────────────────────────────────────────────
    if digit_r > 0.3:
        insights.append({"icon": "⚡", "type": "danger",
                         "text": f"⚡ High digit ratio ({digit_r:.0%}) — 75% bot pattern indicator"})
    else:
        insights.append({"icon": "✓", "type": "positive",
                         "text": f"✓ Natural username ({digit_r:.0%} digits) — human-written pattern"})

    # ── Follower / Following ratio ───────────────────────────────────────
    if ff_r > 100:
        insights.append({"icon": "✓", "type": "positive",
                         "text": f"✓ Strong follower/following ({ff_r:.0f}:1) — authentic account"})
    elif ff_r < 0.05 and friends > 100:
        insights.append({"icon": "⚡", "type": "danger",
                         "text": f"⚡ Follows {friends:,} but {followers:,} followers ({ff_r:.2f}:1) — 70% bot pattern"})
    else:
        insights.append({"icon": "⚠", "type": "warning",
                         "text": f"⚠ Imbalanced ratio ({ff_r:.2f}:1) — potential concern"})

    # ── Follower count ────────────────────────────────────────────────────────
    if followers < 50:
        insights.append({"icon": "⚠", "type": "warning",
                         "text": f"⚠ Low followers ({followers}) — 30% bot correlation"})
    elif followers > 100_000:
        insights.append({"icon": "✓", "type": "positive",
                         "text": f"✓ High followers ({followers:,}) — authentic account pattern"})

    # ── Tweet frequency ───────────────────────────────────────────────────────
    if tpd > 30:
        insights.append({"icon": "⚡", "type": "danger",
                         "text": f"⚡ Extreme posting ({tpd:.0f}/day) — 85% automation indicator"})
    elif tpd < 0.1:
        insights.append({"icon": "⚠", "type": "warning",
                         "text": f"⚠ Low activity ({tpd:.2f}/day) — 40% bot correlation"})
    else:
        insights.append({"icon": "✓", "type": "positive",
                         "text": f"✓ Normal posting ({tpd:.1f}/day) — human activity pattern"})

    # ── Account age ──────────────────────────────────────────────────────────
    age_years = age // 365
    age_months = (age % 365) // 30
    if age > 1825:
        insights.append({"icon": "✓", "type": "positive",
                         "text": f"✓ Established ({age_years}+ years) — 5% bot likelihood"})
    elif age > 365:
        insights.append({"icon": "✓", "type": "positive",
                         "text": f"✓ Established ({age_years}y {age_months}m) — 20% bot correlation"})
    else:
        insights.append({"icon": "⚠", "type": "warning",
                         "text": f"⚠ New account ({age} days) — 35% bot correlation"})

    # ── Bio ────────────────────────────────────────────────────────────────────
    if len(desc) < 10:
        insights.append({"icon": "⚠", "type": "warning",
                         "text": f"⚠ Missing bio ({len(desc)} chars) — 50% bot correlation"})
    else:
        ent = _entropy(desc)
        if ent < 2.0:
            insights.append({"icon": "⚡", "type": "danger",
                             "text": f"⚡ Low bio entropy ({ent:.2f}/4) — templated text, 65% bot indicator"})
        else:
            insights.append({"icon": "✓", "type": "positive",
                             "text": f"✓ Natural bio (entropy {ent:.2f}/4) — human-written content"})

    # ── Favorites ──────────────────────────────────────────────────────────────
    if statuses > 100:
        fav_pct = (fav / statuses) * 100
        if fav_pct < 0.5:
            insights.append({"icon": "⚡", "type": "danger",
                             "text": f"⚡ No favorites ({fav_pct:.1f}%) — 60% bot correlation"})

    # ── Listed ──────────────────────────────────────────────────────────────────
    if listed == 0 and followers > 1000:
        insights.append({"icon": "⚡", "type": "danger",
                         "text": f"⚡ Not in lists (0) despite {followers:,} followers — 55% bot correlation"})

    return insights[:6]


def analyze_account(account: dict) -> dict:
    """
    Run the full hybrid LSTM + Ensemble pipeline (or fallback gracefully).
    account dict keys: screen_name, name, description, verified,
                       followers_count, friends_count, statuses_count,
                       favourites_count, listed_count, user_age, default_profile
    """
    bundle, extractor = load_model_artifacts()

    bot_prob = human_prob = None
    engine_label = "Heuristic Engine"

    if bundle is not None:
        model_type = bundle.get('model_type', 'legacy')

        # ── Hybrid LSTM + Ensemble path ────────────────────────────────
        if 'hybrid' in str(model_type) and extractor is not None:
            try:
                tokenizer = bundle['tokenizer']
                scaler    = bundle['scaler']
                ensemble  = bundle['ensemble']
                cfg       = bundle.get('config', {})
                global MAX_LEN
                MAX_LEN   = cfg.get('max_len', MAX_LEN)

                text        = account.get('description', '') or ''
                lstm_feat   = extract_lstm_features(text, tokenizer, extractor)   # (1,32)
                meta_raw    = extract_metadata_features(account)                   # (1,23)
                meta_scaled = scaler.transform(meta_raw)
                X           = np.hstack([lstm_feat, meta_scaled])                  # (1,55)

                proba     = ensemble.predict_proba(X)[0]
                bot_prob  = float(proba[1]) if len(proba) > 1 else 0.5
                human_prob = float(proba[0])
                engine_label = "Hybrid LSTM + Ensemble"
            except Exception as e:
                print(f"  ⚠ Hybrid inference failed: {e}")

        # ── Legacy ensemble (metadata + GloVe) path ───────────────────
        if bot_prob is None and 'ensemble' in bundle:
            try:
                meta_feat = _extract_legacy_features(account)
                glove_feat = _bio_to_glove(account.get('description', '')).reshape(1, -1)
                X         = np.hstack([meta_feat, glove_feat])
                proba     = bundle['ensemble'].predict_proba(X)[0]
                bot_prob  = float(proba[1]) if len(proba) > 1 else 0.5
                human_prob = float(proba[0])
                engine_label = "Ensemble (Legacy)"
            except Exception as e:
                print(f"  ⚠ Legacy inference failed: {e}")

    # ── Pure heuristic fallback ────────────────────────────────────────
    if bot_prob is None:
        bot_prob, human_prob = _rule_based(account)

    # ── Long-standing account soft correction ──────────────────────────
    user_age_days = max(account.get('user_age', account.get('account_age_days', 365)), 1)
    if user_age_days > 1825:
      if bot_prob < 0.90:
        discount = 0.15 if user_age_days > 3650 else 0.10
        bot_prob = max(0.02, bot_prob - discount)
        human_prob = 1.0 - bot_prob
      tpd = account.get('statuses_count', 0) / user_age_days
      if tpd < 0.5:
        bot_prob = max(0.02, bot_prob - 0.08)
        human_prob = 1.0 - bot_prob

    # ── Build result ───────────────────────────────────────────────────
    label      = "BOT" if bot_prob > 0.75 else "HUMAN"
    confidence = max(bot_prob, human_prob)
    auth_score = round(human_prob * 100, 1)

    if bot_prob > 0.75:
        risk_level, risk_color = "HIGH",   "#FF4B4B"
    elif bot_prob > 0.45:
        risk_level, risk_color = "MEDIUM", "#F59E0B"
    else:
        risk_level, risk_color = "LOW",    "#10B981"

    insights = generate_insights(account, bot_prob)

    sc      = account.get('screen_name', '')
    st      = account.get('statuses_count', 0)
    desc    = str(account.get('description', '') or '')
    age     = max(account.get('user_age', account.get('account_age_days', 365)), 1)
    friends = account.get('friends_count', account.get('following_count', 1)) or 1
    fol     = account.get('followers_count', 0)

    tpd = st / age
    ff_ratio = fol / friends if friends > 0 else 0
    ff_pct = (fol / (fol + friends) * 100) if (fol + friends) > 0 else 0
    entropy_pct = (_entropy(sc) / 4.0) * 100  # max entropy ~4
    bio_entropy = _entropy(desc)
    bio_pct = min(100, len(desc) / 200 * 100) if desc else 0
    fav_rate = (account.get('favourites_count', 0) / max(st, 1) * 100) if st > 0 else 0
    digit_r = sum(c.isdigit() for c in sc) / max(len(sc), 1)
    
    # Determine status for each metric
    posting_status = "⚠ High" if tpd > 30 else ("✓ Normal" if tpd >= 0.1 else "⚠ Low")
    digit_status = "⚠ High" if digit_r > 0.3 else "✓ Natural"
    bio_status = "⚠ Minimal" if len(desc) < 20 else ("✓ Complete" if len(desc) > 50 else "⚠ Short")
    ratio_status = "✓ Healthy" if (ff_ratio > 0.5 or ff_ratio > 100) else ("⚠ Low" if ff_ratio < 0.05 else "⚠ Imbalanced")
    age_status = "✓ Established" if age > 365 else ("⚠ New" if age < 30 else "⚠ Infant")
    fav_status = "✓ Engaged" if fav_rate > 2 else ("⚠ Low" if fav_rate < 0.5 and st > 100 else "⚠ None")
    
    behaviors = [
        {"label": "Posting Frequency",
         "value": f"{posting_status} — {tpd:.2f}/day ({st:,} tweets / {age} days)",
         "percentage": None,
         "status": posting_status,
         "suspicious": tpd > 30,
         "score": min(100, int(tpd * 2))},
        
        {"label": "Username Randomness",
         "value": f"{digit_status} — {digit_r:.0%} digits ({_entropy(sc):.2f} entropy bits)",
         "percentage": None,
         "status": digit_status,
         "suspicious": digit_r > 0.3,
         "score": min(100, int(_entropy(sc) * 25))},
        
        {"label": "Bio Completeness",
         "value": f"{bio_status} — {len(desc)}/150 chars (entropy: {bio_entropy:.2f})",
         "percentage": None,
         "status": bio_status,
         "suspicious": len(desc) < 20,
         "score": min(100, int(bio_pct))},
        
        {"label": "Follower / Following Ratio",
         "value": f"{ratio_status} — {ff_ratio:.2f}:1 ({fol:,} / {friends:,}, {ff_pct:.1f}%)",
         "percentage": None,
         "status": ratio_status,
         "suspicious": ff_ratio < 0.05 and friends > 100,
         "score": min(100, int(min(ff_ratio * 20, 100)))},
        
        {"label": "Account Age",
         "value": f"{age_status} — {age} days ({age // 365}y {(age % 365) // 30}m)",
         "percentage": None,
         "status": age_status,
         "suspicious": age < 30,
         "score": min(100, int((age / 1825) * 100))},
        
        {"label": "Tweet Favorites Ratio",
         "value": f"{fav_status} — {fav_rate:.1f}% favorites ({account.get('favourites_count', 0):,}/{st:,})",
         "percentage": None,
         "status": fav_status,
         "suspicious": fav_rate < 0.5 and st > 100,
         "score": int(fav_rate)},
    ]

    return {
        "label":              label,
        "confidence":         round(confidence * 100, 1),
        "bot_probability":    round(bot_prob    * 100, 1),
        "human_probability":  round(human_prob  * 100, 1),
        "authenticity_score": auth_score,
        "risk_level":         risk_level,
        "risk_color":         risk_color,
        "insights":           insights,
        "behaviors":          behaviors,
        "screen_name":        sc,
        "model_used":         engine_label,
    }


def _parse_twitter_url(url: str) -> str:
    url = url.strip()
    for pat in [r'(?:twitter\.com|x\.com)/([A-Za-z0-9_]+)', r'@([A-Za-z0-9_]+)']:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    clean = url.lstrip('@').strip()
    if re.match(r'^[A-Za-z0-9_]{1,50}$', clean):
        return clean
    return None


# ============================================================================
# HTML Template
# ============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SecureProBot — AI Twitter Bot Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap" rel="stylesheet"/>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#0A0E17;--bg2:#0F1520;--surface:#141B27;--surface2:#1A2235;
  --border:rgba(255,255,255,0.07);--border-hover:rgba(29,161,242,0.4);
  --text:#EEF2F7;--text2:#8B98AA;--text3:#5A6478;
  --accent:#1DA1F2;--accent2:#0D8BD9;--accent-glow:rgba(29,161,242,0.18);
  --green:#10B981;--red:#EF4444;--amber:#F59E0B;
  --radius:14px;--radius-sm:8px;--radius-lg:20px;
  --shadow:0 4px 32px rgba(0,0,0,0.45);--shadow-sm:0 2px 12px rgba(0,0,0,0.3);
  --font-display:'Syne',sans-serif;--font-body:'DM Sans',sans-serif;
  --transition:0.22s cubic-bezier(0.4,0,0.2,1);
}
[data-theme="light"] {
  --bg:#F0F4FA;--bg2:#FFF;--surface:#FFF;--surface2:#F5F8FF;
  --border:rgba(0,0,0,0.08);--text:#0D1117;--text2:#4A5568;--text3:#8899AA;
  --accent-glow:rgba(29,161,242,0.1);
  --shadow:0 4px 32px rgba(0,0,0,0.08);--shadow-sm:0 2px 12px rgba(0,0,0,0.06);
}
html{scroll-behavior:smooth;}
body{font-family:var(--font-body);background:var(--bg);color:var(--text);line-height:1.6;overflow-x:hidden;transition:background var(--transition),color var(--transition);}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:99px}
h1,h2,h3,h4{font-family:var(--font-display);font-weight:700;line-height:1.15;}
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");opacity:0.4;}
.grid-bg{position:fixed;inset:0;z-index:0;pointer-events:none;background-image:linear-gradient(var(--border) 1px,transparent 1px),linear-gradient(90deg,var(--border) 1px,transparent 1px);background-size:60px 60px;mask-image:radial-gradient(ellipse 80% 50% at 50% 0%,black,transparent);}
.nav{position:fixed;top:0;left:0;right:0;z-index:100;display:flex;align-items:center;justify-content:space-between;padding:0 clamp(1.5rem,5vw,4rem);height:64px;background:rgba(10,14,23,0.7);backdrop-filter:blur(20px) saturate(180%);border-bottom:1px solid var(--border);}
[data-theme="light"] .nav{background:rgba(240,244,250,0.75);}
.nav-logo{display:flex;align-items:center;gap:10px;font-family:var(--font-display);font-weight:800;font-size:1.15rem;color:var(--text);text-decoration:none;letter-spacing:-0.02em;}
.logo-icon{width:34px;height:34px;background:linear-gradient(135deg,var(--accent),#5B8EFF);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;box-shadow:0 0 18px var(--accent-glow);}
.nav-links{display:flex;align-items:center;gap:2rem;list-style:none;}
@media(max-width:768px){.nav-links{display:none;}}
.nav-link{color:var(--text2);text-decoration:none;font-size:0.875rem;font-weight:600;transition:color var(--transition);background:none;border:none;cursor:pointer;padding:0;}
.nav-link:hover{color:var(--text);}
.nav-link.active{color:var(--accent);}
.nav-actions{display:flex;align-items:center;gap:12px;}
.theme-toggle{width:40px;height:40px;border-radius:10px;border:1px solid var(--border);background:var(--surface);color:var(--text2);cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all var(--transition);}
.theme-toggle:hover{border-color:var(--border-hover);color:var(--accent);background:var(--surface2);}
.btn{display:inline-flex;align-items:center;gap:8px;padding:0.55rem 1.25rem;border-radius:var(--radius-sm);font-family:var(--font-body);font-size:0.875rem;font-weight:600;cursor:pointer;text-decoration:none;border:none;transition:all var(--transition);white-space:nowrap;}
.btn-primary{background:var(--accent);color:#fff;box-shadow:0 0 24px var(--accent-glow);}
.btn-primary:hover{background:var(--accent2);box-shadow:0 0 32px rgba(29,161,242,0.35);transform:translateY(-1px);}
.btn-ghost{background:transparent;color:var(--text2);border:1px solid var(--border);}
.btn-ghost:hover{border-color:var(--border-hover);color:var(--text);background:var(--surface);}
.btn-lg{padding:0.85rem 2rem;font-size:1rem;border-radius:var(--radius);}
.section{position:relative;z-index:1;padding:clamp(4rem,8vw,7rem) clamp(1.5rem,5vw,4rem);max-width:1280px;margin:0 auto;}
.page{display:none;}
.page.active{display:block;}
.spotlight{background:radial-gradient(80% 120% at 10% 0%,rgba(29,161,242,0.18),transparent 65%),rgba(20,27,39,0.9);border:1px solid rgba(29,161,242,0.2);border-radius:var(--radius-lg);padding:clamp(1.25rem,3vw,2rem);box-shadow:0 12px 40px rgba(0,0,0,0.35);}
.spotlight + .spotlight{margin-top:1.5rem;}
.results-explain{font-size:0.92rem;color:var(--text2);line-height:1.65;margin:1.25rem 0 1.75rem;}
.prob-explain{font-size:0.78rem;color:var(--text3);line-height:1.55;margin:8px 0 12px;}
.behavior-explain{font-size:0.8rem;color:var(--text3);line-height:1.55;margin-bottom:12px;}
.doc-kicker{font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--text3);margin-bottom:0.5rem;font-weight:700;}
.doc-flow{display:flex;flex-direction:column;gap:10px;font-size:0.9rem;color:var(--text2);}
.doc-flow span{display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border:1px solid var(--border);border-radius:10px;background:var(--surface2);}
.doc-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
@media(max-width:900px){.doc-grid{grid-template-columns:repeat(2,1fr);}}
@media(max-width:640px){.doc-grid{grid-template-columns:1fr;}}
.tag{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;border:1px solid rgba(29,161,242,0.3);background:rgba(29,161,242,0.08);color:var(--accent);font-size:0.72rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;}
.warn-box{border:1px solid rgba(245,158,11,0.35);background:rgba(245,158,11,0.08);color:var(--text2);border-radius:12px;padding:14px 16px;}
.doc-table{width:100%;border-collapse:collapse;font-size:0.85rem;color:var(--text2);}
.doc-table th,.doc-table td{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left;}
.doc-table th{font-size:0.72rem;letter-spacing:0.06em;text-transform:uppercase;color:var(--text3);}
.section-badge{display:inline-flex;align-items:center;gap:7px;padding:5px 14px;border-radius:99px;border:1px solid rgba(29,161,242,0.3);background:rgba(29,161,242,0.08);font-size:0.75rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent);margin-bottom:1.5rem;}
.section-badge::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse 2s ease infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}
.section-title{font-size:clamp(2rem,5vw,3.5rem);letter-spacing:-0.03em;color:var(--text);margin-bottom:1rem;}
.section-sub{font-size:1.1rem;color:var(--text2);max-width:560px;line-height:1.7;}
.hero{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:7rem clamp(1.5rem,5vw,4rem) 4rem;text-align:center;position:relative;z-index:1;overflow:hidden;}
.hero-glow{position:absolute;width:800px;height:500px;border-radius:50%;pointer-events:none;background:radial-gradient(ellipse,rgba(29,161,242,0.12) 0%,transparent 70%);top:10%;left:50%;transform:translateX(-50%);}
.hero-eyebrow{display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:99px;border:1px solid rgba(29,161,242,0.3);background:rgba(29,161,242,0.07);font-size:0.8rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--accent);margin-bottom:2rem;animation:fadeUp 0.6s ease both;}
.hero-eyebrow .dot{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pulse 2s infinite;}
.hero-headline{font-size:clamp(2.8rem,7vw,5.5rem);letter-spacing:-0.04em;line-height:1.05;color:var(--text);margin-bottom:1.5rem;animation:fadeUp 0.6s 0.1s ease both;}
.hero-headline .gradient-text{background:linear-gradient(135deg,var(--accent) 0%,#7EB8FF 50%,var(--accent) 100%);background-size:200% 200%;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:gradientShift 4s ease infinite;}
@keyframes gradientShift{0%,100%{background-position:0% 50%;}50%{background-position:100% 50%;}}
.hero-sub{font-size:1.2rem;color:var(--text2);max-width:580px;line-height:1.7;margin:0 auto 2.5rem;animation:fadeUp 0.6s 0.2s ease both;}
.hero-cta{display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;animation:fadeUp 0.6s 0.3s ease both;margin-bottom:4rem;}
.hero-search{display:flex;gap:14px;align-items:center;justify-content:center;width:100%;max-width:760px;margin:1.5rem auto 2.75rem;}
.hero-search .input-field{flex:2.4;min-width:280px;}
.hero-search .btn-analyze{flex:1;margin-top:0;white-space:nowrap;padding:0.95rem 1.6rem;min-width:240px;}
@media(max-width:720px){.hero-search{flex-direction:column;align-items:stretch;}}
.hero-dashboard{position:relative;max-width:860px;width:100%;animation:fadeUp 0.7s 0.4s ease both;}
.dashboard-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);overflow:hidden;box-shadow:var(--shadow),0 0 60px rgba(29,161,242,0.06);}
.dashboard-topbar{display:flex;align-items:center;gap:8px;padding:14px 20px;background:var(--surface2);border-bottom:1px solid var(--border);}
.dash-dot{width:10px;height:10px;border-radius:50%;}.dash-dot.r{background:#FF5F57;}.dash-dot.y{background:#FEBC2E;}.dash-dot.g{background:#28C840;}
.dash-title{flex:1;text-align:center;font-size:0.8rem;color:var(--text3);font-family:var(--font-display);}
.dashboard-body{padding:20px;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;gap:14px;}
@media(max-width:640px){.dashboard-body{grid-template-columns:1fr;}}
.dash-metric{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;text-align:left;position:relative;overflow:hidden;}
.dash-metric::before{content:'';position:absolute;top:-20px;right:-20px;width:70px;height:70px;border-radius:50%;background:var(--accent-glow);}
.dash-metric-label{font-size:0.7rem;color:var(--text3);letter-spacing:0.07em;text-transform:uppercase;font-weight:600;margin-bottom:8px;}
.dash-metric-value{font-family:var(--font-display);font-size:1.75rem;font-weight:800;color:var(--text);}
.auth-ring{grid-column:span 2;display:flex;align-items:center;justify-content:space-between;padding:16px 24px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);gap:20px;}
.ring-container{position:relative;width:96px;height:96px;flex-shrink:0;}
.ring-svg{transform:rotate(-90deg);}
.ring-text{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.ring-pct{font-family:var(--font-display);font-size:1.1rem;font-weight:800;color:var(--green);line-height:1;}
.ring-lbl{font-size:0.6rem;color:var(--text3);letter-spacing:0.06em;margin-top:2px;}
.ring-bars{flex:1;display:flex;flex-direction:column;gap:10px;}
.ring-bar-row{display:flex;flex-direction:column;gap:5px;}
.ring-bar-meta{display:flex;justify-content:space-between;}
.ring-bar-meta span{font-size:0.72rem;color:var(--text2);font-weight:500;}
.ring-bar-track{height:6px;background:var(--surface);border-radius:99px;overflow:hidden;}
.ring-bar-fill{height:100%;border-radius:99px;}
.auth-ring-info{display:flex;flex-direction:column;gap:4px;}
.auth-ring-title{font-size:0.7rem;color:var(--text3);letter-spacing:0.07em;text-transform:uppercase;font-weight:600;}
.auth-ring-label{font-family:var(--font-display);font-size:1.1rem;font-weight:800;color:var(--green);}
.status-indicator{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:99px;font-size:0.72rem;font-weight:600;}
.status-human{background:rgba(16,185,129,0.12);color:var(--green);border:1px solid rgba(16,185,129,0.25);}
.status-bot{background:rgba(239,68,68,0.12);color:var(--red);border:1px solid rgba(239,68,68,0.25);}
.status-medium{background:rgba(245,158,11,0.12);color:var(--amber);border:1px solid rgba(245,158,11,0.25);}
.hero-badges{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;padding:0 4px;}
.float-badge{border-radius:var(--radius);background:var(--surface);border:1px solid var(--border);padding:9px 14px;font-size:0.78rem;white-space:nowrap;box-shadow:var(--shadow-sm);animation:float 5s ease-in-out infinite;display:flex;align-items:center;gap:8px;}
.float-badge:last-child{animation-delay:2.5s;}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-10px);}}
#analyze{scroll-margin-top:80px;}
.analysis-wrapper{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);overflow:hidden;box-shadow:var(--shadow);}
.tabs-header{display:flex;border-bottom:1px solid var(--border);background:var(--surface2);}
.tab-btn{flex:1;padding:1.1rem 1.5rem;background:transparent;border:none;cursor:pointer;font-family:var(--font-display);font-size:0.9rem;font-weight:600;color:var(--text2);border-bottom:2px solid transparent;transition:all var(--transition);display:flex;align-items:center;justify-content:center;gap:8px;}
.tab-btn:hover{color:var(--text);background:var(--surface);}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent);background:var(--surface);}
.tab-content{display:none;padding:clamp(1.5rem,4vw,2.5rem);}
.tab-content.active{display:block;}
.url-input-wrap{display:flex;gap:12px;align-items:stretch;}
@media(max-width:580px){.url-input-wrap{flex-direction:column;}}
.inline-search{display:flex;gap:16px;align-items:center;}
.inline-search .input-field{flex:1;}
@media(max-width:720px){.inline-search{flex-direction:column;align-items:stretch;}}
.input-field{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-family:var(--font-body);font-size:0.95rem;padding:0.85rem 1.1rem;transition:all var(--transition);outline:none;}
.input-field:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow);}
.input-field::placeholder{color:var(--text3);}
.input-label{display:block;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;color:var(--text3);margin-bottom:6px;}
.form-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;}
@media(max-width:580px){.form-grid{grid-template-columns:1fr;}}
.form-grid-3{grid-template-columns:repeat(3,1fr);}
@media(max-width:720px){.form-grid-3{grid-template-columns:repeat(2,1fr);}}
.form-group{display:flex;flex-direction:column;}
.checkbox-row{display:flex;align-items:center;gap:10px;padding:0.85rem 1.1rem;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);cursor:pointer;transition:border-color var(--transition);}
.checkbox-row:hover{border-color:var(--border-hover);}
.checkbox-row input[type=checkbox]{accent-color:var(--accent);width:16px;height:16px;cursor:pointer;}
.checkbox-row span{font-size:0.9rem;color:var(--text2);user-select:none;}
.btn-analyze{width:100%;margin-top:1.5rem;padding:1rem;font-size:1rem;border-radius:var(--radius);background:linear-gradient(135deg,var(--accent),#5B8EFF);color:#fff;font-family:var(--font-display);font-weight:700;border:none;cursor:pointer;letter-spacing:0.01em;box-shadow:0 4px 24px rgba(29,161,242,0.3);transition:all var(--transition);position:relative;overflow:hidden;}
.btn-analyze:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(29,161,242,0.45);}
.btn-analyze:active{transform:translateY(0);}

/* ── Twitter API tab extras ── */
.api-info-box{background:rgba(29,161,242,0.06);border:1px solid rgba(29,161,242,0.2);border-radius:var(--radius-sm);padding:14px 16px;margin-bottom:1.25rem;font-size:0.83rem;color:var(--text2);line-height:1.6;}
.api-info-box strong{color:var(--accent);}
.api-info-box code{background:var(--surface);padding:2px 6px;border-radius:4px;font-size:0.8rem;color:var(--text);}
.fetched-fields{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px;}
@media(max-width:640px){.fetched-fields{grid-template-columns:repeat(2,1fr);}}
.fetched-field{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:7px 10px;font-size:0.75rem;color:var(--text3);display:flex;align-items:center;gap:6px;}
.fetched-field::before{content:"●";color:var(--green);font-size:8px;}

.scan-overlay{display:none;position:fixed;inset:0;background:rgba(10,14,23,0.85);z-index:500;align-items:center;justify-content:center;flex-direction:column;gap:1.5rem;backdrop-filter:blur(8px);}
.scan-overlay.active{display:flex;}
.scan-ring{width:80px;height:80px;border-radius:50%;border:3px solid var(--surface2);border-top-color:var(--accent);animation:spin 0.9s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.scan-text{font-family:var(--font-display);font-size:1.1rem;color:var(--text2);}
.scan-steps{font-size:0.82rem;color:var(--text3);text-align:center;}
#results{scroll-margin-top:80px;}
.results-grid{display:grid;grid-template-columns:340px 1fr;gap:20px;}
@media(max-width:900px){.results-grid{grid-template-columns:1fr;}}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:28px;box-shadow:var(--shadow-sm);}
.verdict-circle{width:140px;height:140px;margin:0 auto 20px;position:relative;}
.verdict-svg{transform:rotate(-90deg);}
.verdict-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;}
.verdict-pct{font-family:var(--font-display);font-size:1.9rem;font-weight:800;line-height:1;}
.verdict-tag{font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;opacity:0.7;}
.verdict-label{font-family:var(--font-display);font-size:1.5rem;font-weight:800;text-align:center;margin-bottom:6px;}
.verdict-conf{text-align:center;font-size:0.85rem;color:var(--text2);margin-bottom:18px;}
.risk-row{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-radius:var(--radius-sm);background:var(--surface2);border:1px solid var(--border);margin-bottom:10px;}
.risk-row-label{font-size:0.8rem;color:var(--text2);font-weight:500;}
.risk-badge{padding:3px 10px;border-radius:99px;font-size:0.75rem;font-weight:700;}
.prob-bars{display:flex;flex-direction:column;gap:10px;margin-top:18px;}
.prob-bar-row{display:flex;flex-direction:column;gap:5px;}
.prob-bar-meta{display:flex;justify-content:space-between;align-items:center;}
.prob-bar-label{font-size:0.78rem;font-weight:600;color:var(--text2);}
.prob-bar-val{font-size:0.78rem;font-weight:700;}
.prob-bar-track{height:7px;background:var(--surface2);border-radius:99px;overflow:hidden;}
.prob-bar-fill{height:100%;border-radius:99px;transition:width 1s cubic-bezier(0.4,0,0.2,1);}
.insights-list{display:flex;flex-direction:column;gap:10px;}
.insight-item{display:flex;align-items:flex-start;gap:12px;padding:14px 16px;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface2);}
.insight-icon{width:28px;height:28px;min-width:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:13px;}
.insight-icon.positive{background:rgba(16,185,129,0.15);}
.insight-icon.warning{background:rgba(245,158,11,0.15);}
.insight-icon.danger{background:rgba(239,68,68,0.15);}
.insight-text{font-size:0.85rem;color:var(--text2);line-height:1.5;}
.behavior-grid{display:flex;flex-direction:column;gap:14px;}
.behavior-item{display:flex;flex-direction:column;gap:7px;padding:14px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-sm);}
.behavior-top{display:flex;justify-content:space-between;align-items:center;}
.behavior-label{font-size:0.82rem;font-weight:600;color:var(--text);display:flex;align-items:center;gap:8px;}
.behavior-value{font-size:0.78rem;color:var(--text2);}
.metric-desc{font-size:0.75rem;color:var(--text3);line-height:1.5;}
.metric-why{font-size:0.74rem;color:var(--text3);line-height:1.5;}
.info-tip{width:16px;height:16px;border-radius:50%;border:1px solid var(--border);display:inline-flex;align-items:center;justify-content:center;font-size:10px;color:var(--text3);cursor:help;position:relative;}
.info-tip::after{content:attr(data-tip);position:absolute;left:50%;bottom:130%;transform:translateX(-50%);background:var(--surface);border:1px solid var(--border);padding:8px 10px;border-radius:8px;color:var(--text2);font-size:0.72rem;white-space:normal;line-height:1.4;min-width:200px;max-width:260px;opacity:0;pointer-events:none;transition:opacity var(--transition);box-shadow:var(--shadow-sm);}
.info-tip:hover::after{opacity:1;}
.summary-card{background:linear-gradient(180deg,rgba(29,161,242,0.08),rgba(20,27,39,0.9));border:1px solid rgba(29,161,242,0.25);}
.summary-text{font-size:0.92rem;color:var(--text2);line-height:1.7;}
.signal-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;}
.chip{padding:6px 10px;border-radius:999px;font-size:0.72rem;font-weight:600;letter-spacing:0.02em;border:1px solid transparent;}
.chip.good{background:rgba(16,185,129,0.12);color:var(--green);border-color:rgba(16,185,129,0.3);}
.chip.warn{background:rgba(245,158,11,0.12);color:var(--amber);border-color:rgba(245,158,11,0.3);}
.chip.bad{background:rgba(239,68,68,0.12);color:var(--red);border-color:rgba(239,68,68,0.3);}
.trust-list{display:grid;gap:8px;margin-top:10px;color:var(--text2);font-size:0.82rem;}
.trust-item{display:flex;gap:8px;align-items:flex-start;}
.confidence-pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:var(--surface);border:1px solid var(--border);font-size:0.74rem;color:var(--text2);}
.behavior-track{height:6px;background:var(--surface);border-radius:99px;overflow:hidden;}
.behavior-fill{height:100%;border-radius:99px;transition:width 1.2s cubic-bezier(0.4,0,0.2,1);}
.features-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:3rem;}
@media(max-width:900px){.features-grid{grid-template-columns:repeat(2,1fr);}}
@media(max-width:580px){.features-grid{grid-template-columns:1fr;}}
.feature-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;transition:all var(--transition);cursor:default;position:relative;overflow:hidden;}
.feature-card::before{content:'';position:absolute;width:100px;height:100px;border-radius:50%;background:var(--accent-glow);top:-30px;right:-30px;transition:all var(--transition);}
.feature-card:hover{border-color:var(--border-hover);transform:translateY(-3px);box-shadow:var(--shadow);}
.feature-card:hover::before{width:160px;height:160px;}
.feature-icon{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,rgba(29,161,242,0.15),rgba(91,142,255,0.15));border:1px solid rgba(29,161,242,0.2);display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:16px;}
.feature-title{font-family:var(--font-display);font-size:1rem;font-weight:700;color:var(--text);margin-bottom:8px;}
.feature-desc{font-size:0.855rem;color:var(--text2);line-height:1.6;}
.stats-section{background:var(--surface);border-top:1px solid var(--border);border-bottom:1px solid var(--border);padding:4rem clamp(1.5rem,5vw,4rem);}
.stats-inner{max-width:1280px;margin:0 auto;display:grid;grid-template-columns:repeat(3,1fr);gap:2rem;text-align:center;}
@media(max-width:640px){.stats-inner{grid-template-columns:1fr;}}
.stat-value{font-family:var(--font-display);font-size:clamp(2.2rem,5vw,3.5rem);font-weight:800;letter-spacing:-0.04em;background:linear-gradient(135deg,var(--text),var(--accent));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.stat-label{font-size:0.9rem;color:var(--text2);margin-top:4px;font-weight:500;}
.footer{position:relative;z-index:1;padding:3rem clamp(1.5rem,5vw,4rem);border-top:1px solid var(--border);max-width:1280px;margin:0 auto;}
.footer-inner{display:flex;justify-content:space-between;align-items:flex-start;gap:2rem;flex-wrap:wrap;}
.footer-brand{max-width:280px;}
.footer-brand-name{font-family:var(--font-display);font-size:1.05rem;font-weight:800;color:var(--text);margin-bottom:8px;display:flex;align-items:center;gap:8px;}
.footer-brand-desc{font-size:0.83rem;color:var(--text3);line-height:1.6;}
.footer-links-col h4{font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--text3);margin-bottom:14px;}
.footer-links-col ul{list-style:none;display:flex;flex-direction:column;gap:9px;}
.footer-links-col a{font-size:0.85rem;color:var(--text2);text-decoration:none;transition:color var(--transition);}
.footer-links-col a:hover{color:var(--accent);}
.footer-bottom{margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;font-size:0.8rem;color:var(--text3);}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
.fade-in-up{opacity:0;animation:fadeUp 0.6s ease forwards;}
.toast{position:fixed;bottom:2rem;right:2rem;z-index:999;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:14px 20px;font-size:0.875rem;color:var(--text);box-shadow:var(--shadow);transform:translateY(20px);opacity:0;transition:all 0.3s ease;pointer-events:none;}
.toast.show{transform:translateY(0);opacity:1;}
.about-hero{background:linear-gradient(135deg,rgba(29,161,242,0.12) 0%,rgba(20,27,39,0.95) 50%,rgba(59,130,246,0.08) 100%);border:1px solid rgba(29,161,242,0.2);border-radius:var(--radius-lg);padding:3rem 2.5rem;position:relative;overflow:hidden;margin-bottom:3rem;}
.about-hero::before{content:'';position:absolute;inset:0;background:radial-gradient(circle 800px at 20% 50%,rgba(29,161,242,0.15),transparent);pointer-events:none;}
.about-hero-content{position:relative;z-index:1;}
.about-hero h1{font-size:clamp(1.8rem,5vw,2.8rem);color:var(--text);margin-bottom:0.5rem;letter-spacing:-0.02em;}
.about-hero p{font-size:clamp(0.95rem,2vw,1.1rem);color:var(--text2);line-height:1.7;max-width:720px;}
.about-grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px;margin-bottom:3rem;}
.about-grid-3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-bottom:3rem;}
.about-grid-4{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:3rem;}
.card-glass{background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));backdrop-filter:blur(10px) saturate(120%);border:1px solid rgba(255,255,255,0.1);border-radius:var(--radius);padding:24px;transition:all var(--transition);position:relative;}
.card-glass:hover{border-color:rgba(29,161,242,0.4);background:linear-gradient(135deg,rgba(29,161,242,0.08),rgba(59,130,246,0.05));transform:translateY(-4px);box-shadow:0 12px 40px rgba(29,161,242,0.12);}
.card-glass::before{content:'';position:absolute;inset:-1px;border-radius:var(--radius);background:linear-gradient(135deg,rgba(29,161,242,0.1),transparent);opacity:0;transition:opacity var(--transition);pointer-events:none;z-index:-1;}
.card-glass:hover::before{opacity:1;}
.section-label{font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--accent);font-weight:700;margin-bottom:1rem;}
.card-title{font-size:1.05rem;font-weight:700;color:var(--text);margin-bottom:0.75rem;display:flex;align-items:center;gap:8px;}
.card-desc{font-size:0.9rem;color:var(--text2);line-height:1.6;}
.tech-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:999px;background:rgba(29,161,242,0.1);border:1px solid rgba(29,161,242,0.3);color:var(--accent);font-size:0.78rem;font-weight:600;transition:all var(--transition);}
.tech-badge:hover{background:rgba(29,161,242,0.2);border-color:rgba(29,161,242,0.5);transform:translateY(-2px);}
.badge-group{display:flex;flex-wrap:wrap;gap:10px;}
.highlight-stat{font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,var(--accent),#5B8EFF);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.highlight-label{font-size:0.85rem;color:var(--text2);margin-top:6px;}
.roadmap-item{display:flex;gap:16px;padding:16px;border-left:2px solid rgba(29,161,242,0.3);transition:all var(--transition);}
.roadmap-item:hover{border-left-color:var(--accent);background:rgba(29,161,242,0.05);margin-left:4px;}
.roadmap-dot{width:12px;height:12px;border-radius:50%;background:var(--accent);position:relative;top:6px;}
.roadmap-content h4{font-size:0.9rem;color:var(--text);margin-bottom:4px;}
.roadmap-content p{font-size:0.82rem;color:var(--text3);}
.team-card{background:linear-gradient(135deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02));backdrop-filter:blur(10px);border:1px solid rgba(29,161,242,0.2);border-radius:var(--radius);padding:20px;text-align:center;transition:all var(--transition);}
.team-card:hover{border-color:rgba(29,161,242,0.4);transform:translateY(-6px);box-shadow:0 16px 48px rgba(29,161,242,0.15);}
.avatar{width:60px;height:60px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#5B8EFF);display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:700;color:#fff;margin:0 auto 16px;}
.team-name{font-size:1rem;font-weight:700;color:var(--text);margin-bottom:4px;}
.team-role{font-size:0.75rem;letter-spacing:0.05em;text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:10px;}
.team-details{font-size:0.8rem;color:var(--text3);line-height:1.5;margin-bottom:10px;}
.team-email{font-size:0.75rem;color:var(--text2);word-break:break-all;}
</style>
</head>
<body>
<div class="grid-bg"></div>
<div class="scan-overlay" id="scanOverlay">
  <div class="scan-ring"></div>
  <p class="scan-text">Analyzing Account...</p>
  <p class="scan-steps" id="scanStep">Initializing behavioral analysis</p>
</div>
<div class="toast" id="toast"></div>

<nav class="nav">
  <a href="#" class="nav-logo"><div class="logo-icon">🛡</div>SecureProBot</a>
  <ul class="nav-links">
    <li><button class="nav-link active" data-page="home">Analyze</button></li>
    <li><button class="nav-link" data-page="results">Results</button></li>
        <li><button class="nav-link" data-page="docs">Documentation</button></li>
        <li><button class="nav-link" data-page="about">About</button></li>
  </ul>
  <div class="nav-actions">
    <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn" title="Toggle theme">🌙</button>
    <button class="btn btn-primary" onclick="setPage('home')">Analyze Account</button>
  </div>
</nav>

<div class="page active" id="page-home">
<section class="hero">
  <div class="hero-glow"></div>
  <div class="hero-eyebrow"><span class="dot"></span>AI-Powered Bot Detection · v2.1</div>
  <h1 class="hero-headline">Detect Fake Twitter<br/>Bots <span class="gradient-text">Instantly</span></h1>
  <p class="hero-sub">AI-powered Twitter/X account analysis using behavioral intelligence, semantic patterns, and authenticity scoring.</p>
    <div class="hero-search">
        <input type="text" id="apiUsername" class="input-field" placeholder="Paste Twitter/X profile link"/>
        <button class="btn-analyze" onclick="analyzeViaAPI()">
            🔎 Analyze
        </button>
    </div>
    <div class="hero-dashboard">
        <div class="dashboard-card">
      <div class="dashboard-topbar">
        <div class="dash-dot r"></div><div class="dash-dot y"></div><div class="dash-dot g"></div>
        <div class="dash-title">SecureProBot Analysis Dashboard</div>
      </div>
      <div class="dashboard-body" style="grid-template-columns:repeat(3,1fr);gap:14px;">
        <div class="dash-metric">
          <div class="dash-metric-label">Validation AUC</div>
          <div class="dash-metric-value" style="color:var(--green)">0.9882</div>
          <div style="margin-top:6px;font-size:0.72rem;color:var(--text3);">Held-out validation split</div>
        </div>

        <div class="dash-metric">
          <div class="dash-metric-label">Validation Accuracy</div>
          <div class="dash-metric-value" style="color:var(--accent)">94.42%</div>
          <div style="margin-top:6px;font-size:0.72rem;color:var(--text3);">Final held-out accuracy</div>
        </div>

        <div class="dash-metric">
          <div class="dash-metric-label">5-Fold CV AUC</div>
          <div class="dash-metric-value" style="color:var(--amber)">0.9929 ± 0.0028</div>
          <div style="margin-top:6px;font-size:0.72rem;color:var(--text3);">Cross-validated performance</div>
        </div>

        <div style="grid-column:span 3;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;">
          <div style="font-size:0.7rem;color:var(--text3);letter-spacing:0.07em;text-transform:uppercase;font-weight:600;margin-bottom:10px;">Cross-Domain Results</div>
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:10px;"><span style="font-size:0.8rem;color:var(--text2);">midterm_18</span><span style="font-size:0.8rem;font-weight:700;color:var(--green);">AUC 0.9333</span></div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:10px;"><span style="font-size:0.8rem;color:var(--text2);">cresci_rtbust</span><span style="font-size:0.8rem;font-weight:700;color:var(--amber);">AUC 0.6656</span></div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:10px;"><span style="font-size:0.8rem;color:var(--text2);">gilani_17</span><span style="font-size:0.8rem;font-weight:700;color:var(--amber);">AUC 0.5338</span></div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:10px;"><span style="font-size:0.8rem;color:var(--text2);">botwiki_verified</span><span style="font-size:0.8rem;font-weight:700;color:var(--green);">AUC 0.9900</span></div>
          </div>
        </div>

      </div>
    </div>
  </div>
</section>

<!-- ── Analysis Section ── -->
</div>

<!-- ── Results ── -->
<section id="results" class="page">
  <div class="section" style="padding-top:2rem;">
    <div class="section-badge">Analysis Complete</div>
    <h2 class="section-title" id="resultsTitle">Account Report</h2>
    <p class="results-explain">This report now highlights the training evidence behind SecureProBot, including validation AUC, 5-fold cross-validation, selected features, and cross-domain performance. The live verdict still summarizes the account outcome, but the UI no longer shows human and bot probability percentages.</p>
    <div class="results-grid" style="margin-top:2rem;" id="resultsGrid"></div>
  </div>
</section>

<section id="page-docs" class="page">
  <div class="section" style="padding-top:2rem;">
    <!-- Hero Section -->
    <div class="about-hero">
      <div class="about-hero-content">
        <h1>Complete Documentation</h1>
        <p>Understand how SecureProBot detects bots, what signals matter, and how to interpret your results with confidence.</p>
      </div>
    </div>

    <!-- Introduction Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Getting Started</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">What Is SecureProBot?</h2>
      
      <div class="card-glass" style="padding:32px;">
        <p class="card-desc" style="font-size:1rem;line-height:1.8;">SecureProBot is an AI-powered Twitter/X bot detection platform designed to identify suspicious automated accounts using behavioral analysis, metadata signals, and machine learning. It detects likely automated behavior, helps researchers and students evaluate accounts, and supports safer online discussion by surfacing bot-like patterns early.</p>
      </div>
    </div>

    <!-- Detection Pipeline Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">How It Works</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Detection Pipeline</h2>
      
      <div class="card-glass">
        <div style="display:flex;flex-direction:column;gap:16px;">
          <div style="display:flex;align-items:center;gap:16px;">
            <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#5B8EFF);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;flex-shrink:0;">1</div>
            <div>
              <div class="card-title" style="margin:0;">Twitter/X URL Input</div>
              <p class="card-desc" style="margin:0;">Enter a username to analyze</p>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:16px;">
            <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#5B8EFF);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;flex-shrink:0;">2</div>
            <div>
              <div class="card-title" style="margin:0;">Metadata Collection</div>
              <p class="card-desc" style="margin:0;">Fetch public profile data via Twitter API</p>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:16px;">
            <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#5B8EFF);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;flex-shrink:0;">3</div>
            <div>
              <div class="card-title" style="margin:0;">Signal Extraction</div>
              <p class="card-desc" style="margin:0;">Compute 55+ behavioral and linguistic signals</p>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:16px;">
            <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#5B8EFF);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;flex-shrink:0;">4</div>
            <div>
              <div class="card-title" style="margin:0;">AI Model Analysis</div>
              <p class="card-desc" style="margin:0;">Hybrid LSTM + Ensemble scoring</p>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:16px;">
            <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#5B8EFF);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;flex-shrink:0;">5</div>
            <div>
              <div class="card-title" style="margin:0;">Explainable Results</div>
              <p class="card-desc" style="margin:0;">Probability score + reasoning signals</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Analyzed Signals Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Signal Analysis</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">What We Analyze</h2>
      
      <div class="about-grid-3">
        <div class="card-glass">
          <div style="font-size:24px;margin-bottom:12px;">📊</div>
          <h3 class="card-title">Behavioral Signals</h3>
          <p class="card-desc">Posting frequency, account age, follower/following ratio, engagement consistency, and temporal patterns.</p>
        </div>
        <div class="card-glass">
          <div style="font-size:24px;margin-bottom:12px;">👤</div>
          <h3 class="card-title">Profile Signals</h3>
          <p class="card-desc">Username entropy, bio completeness, profile customization, verified status, and avatar patterns.</p>
        </div>
        <div class="card-glass">
          <div style="font-size:24px;margin-bottom:12px;">🔤</div>
          <h3 class="card-title">Language Signals</h3>
          <p class="card-desc">Repetitive phrases, template-like bios, character entropy, and automation pattern detection.</p>
        </div>
      </div>
    </div>

    <!-- How AI Decides Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Methodology</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">How the AI Decides</h2>
      
      <div class="card-glass">
        <h3 class="card-title" style="font-size:1.1rem;">Probabilistic Scoring, Not Certainty</h3>
        <p class="card-desc" style="margin-bottom:16px;">SecureProBot does not "know" if an account is a bot—it estimates probabilities based on observable patterns. The system analyzes signals independently and synthesizes them into a composite score.</p>
        
        <div style="margin-top:24px;padding-top:24px;border-top:1px solid rgba(255,255,255,0.1);">
          <h4 style="color:var(--text);font-weight:700;margin-bottom:12px;">Model Architecture</h4>
          <p class="card-desc">Hybrid LSTM (32 dims) + Ensemble (23 metadata dims) = 55-dimensional feature vector</p>
          <div style="display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;">
            <span class="tech-badge">LSTM Text Analysis</span>
            <span class="tech-badge">Random Forest</span>
            <span class="tech-badge">Extra-Trees</span>
            <span class="tech-badge">Feature Scaling</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Confidence Levels Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Interpretation</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Confidence Levels</h2>
      
      <div class="about-grid-3">
        <div class="card-glass">
          <div style="font-size:32px;font-weight:800;background:linear-gradient(135deg,var(--green),#10D981);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;">61–100%</div>
          <h4 style="color:var(--text);font-weight:700;margin-bottom:8px;">Likely Human</h4>
          <p class="card-desc">Account shows strong human-like signals. Behavior aligns with typical user patterns.</p>
        </div>
        <div class="card-glass">
          <div style="font-size:32px;font-weight:800;background:linear-gradient(135deg,var(--amber),#FBBF24);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;">31–60%</div>
          <h4 style="color:var(--text);font-weight:700;margin-bottom:8px;">Uncertain</h4>
          <p class="card-desc">Mixed signals detected. Account may need closer inspection or context review.</p>
        </div>
        <div class="card-glass">
          <div style="font-size:32px;font-weight:800;background:linear-gradient(135deg,var(--red),#F87171);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;">0–30%</div>
          <h4 style="color:var(--text);font-weight:700;margin-bottom:8px;">Likely Bot</h4>
          <p class="card-desc">Account exhibits significant bot-like patterns and suspicious signals.</p>
        </div>
      </div>

      <div class="card-glass" style="margin-top:24px;background:rgba(245,158,11,0.08);border-color:rgba(245,158,11,0.2);">
        <p class="card-desc" style="margin:0;"><strong>⚠️ Important:</strong> Scores are probabilistic estimates, not definitive proof. Context matters—dormant legitimate accounts, parody profiles, and new users may show bot-like signals.</p>
      </div>
    </div>

    <!-- Training & Data Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Research Foundation</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Training & Datasets</h2>
      
      <div class="about-grid-2">
        <div class="card-glass">
          <h3 class="card-title">Data Sources</h3>
          <p class="card-desc">Trained on public Twitter bot detection datasets including Cresci-2017, Varol-ICWSM, and Gilani-2017 corpora.</p>
        </div>
        <div class="card-glass">
          <h3 class="card-title">Methods</h3>
          <p class="card-desc">Feature engineering, dataset balancing, adversarial sample generation, and cross-validation for robustness.</p>
        </div>
      </div>
    </div>

    <!-- Privacy & Ethics Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Trust & Safety</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Privacy & Ethics</h2>
      
      <div class="about-grid-2">
        <div class="card-glass">
          <div style="font-size:20px;margin-bottom:12px;">🔒</div>
          <h3 class="card-title">Privacy-First</h3>
          <p class="card-desc">Only public profile data is analyzed. No passwords, private messages, or account logins required or stored.</p>
        </div>
        <div class="card-glass">
          <div style="font-size:20px;margin-bottom:12px;">🛡️</div>
          <h3 class="card-title">Ethical Use</h3>
          <p class="card-desc">Built for research and educational purposes. Designed to combat misinformation, not to target or harass users.</p>
        </div>
      </div>
    </div>

    <!-- Limitations Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Transparency</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Known Limitations</h2>
      
      <div class="card-glass" style="background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.2);">
        <h3 class="card-title">False Positives & Negatives</h3>
        <p class="card-desc" style="margin-bottom:16px;">No system is 100% accurate. The following scenarios may reduce confidence:</p>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:12px;">
          <div style="padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;">
            <div style="font-weight:600;color:var(--text);font-size:0.9rem;">Dormant Accounts</div>
            <p class="card-desc" style="margin:6px 0 0;font-size:0.85rem;">Old accounts with minimal activity</p>
          </div>
          <div style="padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;">
            <div style="font-weight:600;color:var(--text);font-size:0.9rem;">Parody Profiles</div>
            <p class="card-desc" style="margin:6px 0 0;font-size:0.85rem;">Intentionally absurd or comedic accounts</p>
          </div>
          <div style="padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;">
            <div style="font-weight:600;color:var(--text);font-size:0.9rem;">New Users</div>
            <p class="card-desc" style="margin:6px 0 0;font-size:0.85rem;">Recently created legitimate accounts</p>
          </div>
          <div style="padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;">
            <div style="font-weight:600;color:var(--text);font-size:0.9rem;">Corporate Accounts</div>
            <p class="card-desc" style="margin:6px 0 0;font-size:0.85rem;">Brand accounts with template-like bios</p>
          </div>
          <div style="padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;">
            <div style="font-weight:600;color:var(--text);font-size:0.9rem;">Multilingual Bios</div>
            <p class="card-desc" style="margin:6px 0 0;font-size:0.85rem;">Non-English text patterns</p>
          </div>
          <div style="padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;">
            <div style="font-weight:600;color:var(--text);font-size:0.9rem;">API Constraints</div>
            <p class="card-desc" style="margin:6px 0 0;font-size:0.85rem;">Limited historical tweet data</p>
          </div>
        </div>
      </div>
    </div>

    <!-- FAQ Section -->
    <div>
      <div class="section-label">Questions</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Frequently Asked Questions</h2>
      
      <div style="display:flex;flex-direction:column;gap:14px;">
        <details style="border:1px solid rgba(255,255,255,0.1);border-radius:var(--radius);padding:20px;background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));backdrop-filter:blur(10px);cursor:pointer;transition:all var(--transition);">
          <summary style="font-weight:700;color:var(--text);display:flex;justify-content:space-between;align-items:center;">Is the result always accurate?
            <span style="font-size:1.2rem;margin-left:12px;">▾</span>
          </summary>
          <p class="card-desc" style="margin-top:16px;margin-bottom:0;">No. SecureProBot provides probabilistic estimates based on available signals. The accuracy varies depending on account type, history, and context. It's a research tool, not a definitive classifier.</p>
        </details>

        <details style="border:1px solid rgba(255,255,255,0.1);border-radius:var(--radius);padding:20px;background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));backdrop-filter:blur(10px);cursor:pointer;transition:all var(--transition);">
          <summary style="font-weight:700;color:var(--text);display:flex;justify-content:space-between;align-items:center;">Does SecureProBot access private data?
            <span style="font-size:1.2rem;margin-left:12px;">▾</span>
          </summary>
          <p class="card-desc" style="margin-top:16px;margin-bottom:0;">No. Only public profile data is analyzed. SecureProBot does not access passwords, direct messages, followers lists, or any private information. It only fetches what appears on a public profile page.</p>
        </details>

        <details style="border:1px solid rgba(255,255,255,0.1);border-radius:var(--radius);padding:20px;background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));backdrop-filter:blur(10px);cursor:pointer;transition:all var(--transition);">
          <summary style="font-weight:700;color:var(--text);display:flex;justify-content:space-between;align-items:center;">Can legitimate humans be flagged as bots?
            <span style="font-size:1.2rem;margin-left:12px;">▾</span>
          </summary>
          <p class="card-desc" style="margin-top:16px;margin-bottom:0;">Yes. Dormant accounts, new users, sparse posters, or accounts with minimal bio information can receive higher bot scores. Context and manual review are always recommended.</p>
        </details>

        <details style="border:1px solid rgba(255,255,255,0.1);border-radius:var(--radius);padding:20px;background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));backdrop-filter:blur(10px);cursor:pointer;transition:all var(--transition);">
          <summary style="font-weight:700;color:var(--text);display:flex;justify-content:space-between;align-items:center;">What signals have the most weight?
            <span style="font-size:1.2rem;margin-left:12px;">▾</span>
          </summary>
          <p class="card-desc" style="margin-top:16px;margin-bottom:0;">Account age, posting frequency, follower-to-following ratio, and bio language patterns are among the strongest indicators. The hybrid model combines these to avoid over-weighting any single signal.</p>
        </details>

        <details style="border:1px solid rgba(255,255,255,0.1);border-radius:var(--radius);padding:20px;background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));backdrop-filter:blur(10px);cursor:pointer;transition:all var(--transition);">
          <summary style="font-weight:700;color:var(--text);display:flex;justify-content:space-between;align-items:center;">Does SecureProBot store analysis history?
            <span style="font-size:1.2rem;margin-left:12px;">▾</span>
          </summary>
          <p class="card-desc" style="margin-top:16px;margin-bottom:0;">No. Results are computed on-demand. No analysis records, usernames, or results are stored on the server. Each analysis is independent.</p>
        </details>

        <details style="border:1px solid rgba(255,255,255,0.1);border-radius:var(--radius);padding:20px;background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));backdrop-filter:blur(10px);cursor:pointer;transition:all var(--transition);">
          <summary style="font-weight:700;color:var(--text);display:flex;justify-content:space-between;align-items:center;">Is this tool peer-reviewed or published?
            <span style="font-size:1.2rem;margin-left:12px;">▾</span>
          </summary>
          <p class="card-desc" style="margin-top:16px;margin-bottom:0;">SecureProBot is a research-grade prototype built by University of Mindanao students. It uses established methodologies from published bot detection research (Cresci, Varol, Gilani) but is presented as an educational tool.</p>
        </details>
      </div>
    </div>
  </div>
</section>

<section id="page-about" class="page">
  <div class="section" style="padding-top:2rem;">
    <!-- Hero Mission Section -->
    <div class="about-hero">
      <div class="about-hero-content">
        <h1>Building Trust in Social Media</h1>
        <p>SecureProBot is a research-driven, explainable AI platform that identifies suspicious automated activity on Twitter/X using behavioral analysis and transparency-first machine learning.</p>
      </div>
    </div>

    <!-- Mission Statement -->
    <div class="about-grid-2" style="margin-bottom:3rem;">
      <div class="card-glass">
        <div class="section-label">Mission</div>
        <h3 class="card-title">Why We Built SecureProBot</h3>
        <p class="card-desc">Misinformation, spam, and automated manipulation damage public discourse. We built SecureProBot to help researchers, developers, and users surface suspicious automation with transparent, explainable AI.</p>
      </div>
      <div class="card-glass">
        <div class="section-label">Research Goal</div>
        <h3 class="card-title">Our Focus</h3>
        <p class="card-desc">This project explores interpretable machine learning for social bot detection, emphasizing transparency, robustness, and adversarial resistance. We prioritize explainability over black-box accuracy.</p>
      </div>
    </div>

    <!-- Tech Stack Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Technology</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Our Tech Stack</h2>
      
      <div class="about-grid-2">
        <div class="card-glass">
          <h3 class="card-title">Backend & ML</h3>
          <div class="badge-group">
            <span class="tech-badge">Python</span>
            <span class="tech-badge">Flask</span>
            <span class="tech-badge">Random Forest</span>
            <span class="tech-badge">Extra-Trees</span>
            <span class="tech-badge">LSTM</span>
          </div>
        </div>
        <div class="card-glass">
          <h3 class="card-title">Frontend & APIs</h3>
          <div class="badge-group">
            <span class="tech-badge">HTML5</span>
            <span class="tech-badge">CSS3</span>
            <span class="tech-badge">JavaScript</span>
            <span class="tech-badge">GetXAPI</span>
            <span class="tech-badge">Twitter API</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Highlights Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Highlights</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Why SecureProBot Stands Out</h2>
      
      <div class="about-grid-4">
        <div class="card-glass">
          <div class="highlight-stat">55+</div>
          <div class="highlight-label">Extracted Signals</div>
        </div>
        <div class="card-glass">
          <div class="highlight-stat">Hybrid</div>
          <div class="highlight-label">AI Detection</div>
        </div>
        <div class="card-glass">
          <div class="highlight-stat">100%</div>
          <div class="highlight-label">Explainable</div>
        </div>
        <div class="card-glass">
          <div class="highlight-stat">Adversarial</div>
          <div class="highlight-label">Hardened</div>
        </div>
      </div>
    </div>

    <!-- Roadmap Section -->
    <div style="margin-bottom:3rem;">
      <div class="section-label">Future</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Product Roadmap</h2>
      
      <div class="card-glass">
        <div class="roadmap-item">
          <div class="roadmap-dot"></div>
          <div class="roadmap-content">
            <h4>Network Graph Analysis</h4>
            <p>Detect coordinated bot networks and influence patterns.</p>
          </div>
        </div>
        <div class="roadmap-item">
          <div class="roadmap-dot"></div>
          <div class="roadmap-content">
            <h4>Multilingual Support</h4>
            <p>Extend detection to non-English profiles and text patterns.</p>
          </div>
        </div>
        <div class="roadmap-item">
          <div class="roadmap-dot"></div>
          <div class="roadmap-content">
            <h4>Browser Extension</h4>
            <p>Real-time bot detection directly in Twitter/X interface.</p>
          </div>
        </div>
        <div class="roadmap-item">
          <div class="roadmap-dot"></div>
          <div class="roadmap-content">
            <h4>API Dashboard</h4>
            <p>Comprehensive analytics and usage dashboard for developers.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Team Section -->
    <div>
      <div class="section-label">Team</div>
      <h2 style="font-family:var(--font-display);font-size:1.4rem;color:var(--text);margin-bottom:1.5rem;">Meet the Developers</h2>
      
      <div class="about-grid-2">
        <div class="team-card">
          <div class="avatar">JC</div>
          <div class="team-name">Joevan B. Capote</div>
          <div class="team-role">Backend Development</div>
          <div style="display:flex;gap:6px;margin-bottom:10px;justify-content:center;flex-wrap:wrap;">
            <span class="tech-badge" style="background:rgba(16,185,129,0.1);border-color:rgba(16,185,129,0.3);color:var(--green);">ML Pipeline</span>
            <span class="tech-badge" style="background:rgba(16,185,129,0.1);border-color:rgba(16,185,129,0.3);color:var(--green);">API Integration</span>
          </div>
          <div class="team-details">
            <strong>B.S. Computer Science</strong><br/>
            University of Mindanao
          </div>
          <div class="team-email">j.capote.545089@umindanao.edu.ph</div>
        </div>
        <div class="team-card">
          <div class="avatar" style="background:linear-gradient(135deg,#F59E0B,#FBBF24);">MC</div>
          <div class="team-name">Mheil Andrei N. Cenita</div>
          <div class="team-role">Full Stack Developer</div>
          <div style="display:flex;gap:6px;margin-bottom:10px;justify-content:center;flex-wrap:wrap;">
            <span class="tech-badge" style="background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.3);color:var(--amber);">UI/UX Design</span>
            <span class="tech-badge" style="background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.3);color:var(--amber);">Visualization</span>
          </div>
          <div class="team-details">
            <strong>B.A. Multimedia Arts</strong><br/>
            University of Mindanao
          </div>
          <div class="team-email">m.cenita.545045@umindanao.edu.ph</div>
        </div>
      </div>
    </div>
  </div>
</section>


<footer>
  <div class="footer">
    <div class="footer-inner">
      <div class="footer-brand">
        <div class="footer-brand-name"><div class="logo-icon" style="width:28px;height:28px;font-size:13px;">🛡</div>SecureProBot</div>
        <p class="footer-brand-desc">AI-powered Twitter/X bot detection for researchers, developers, and platform integrity teams.</p>
      </div>
    <div class="footer-links-col"><h4>Product</h4><ul><li><a href="#" onclick="setPage('home')">Analyze</a></li><li><a href="#" onclick="setPage('results')">Results</a></li><li><a href="#" onclick="setPage('docs')">Documentation</a></li><li><a href="#" onclick="setPage('about')">About</a></li></ul></div>
      <div class="footer-links-col"><h4>Research</h4><ul><li><a href="#">API Docs</a></li><li><a href="#">Dataset Sources</a></li><li><a href="#">Publications</a></li></ul></div>
      <div class="footer-links-col"><h4>Legal</h4><ul><li><a href="#">Privacy Policy</a></li><li><a href="#">Terms of Use</a></li><li><a href="#">Contact</a></li></ul></div>
    </div>
    <div class="footer-bottom">
      <span>SecureProBot © 2025 — For research and educational purposes only</span>
      <span>Powered by Flask · Scikit-learn · TensorFlow</span>
    </div>
  </div>
</footer>

<script>
/* Theme */
const themeBtn = document.getElementById('themeBtn');
function setTheme(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('spb-theme',t);themeBtn.textContent=t==='dark'?'🌙':'☀️';}
function toggleTheme(){setTheme(document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark');}
(function(){setTheme(localStorage.getItem('spb-theme')||'dark');})();

const MODEL_EVIDENCE = {
  validationAuc: '0.9882',
  validationAccuracy: '94.42%',
  cvAuc: '0.9929 ± 0.0028',
  selectedFeatures: [
    'statuses_count', 'followers_count', 'friends_count', 'favourites_count',
    'listed_count', 'verified', 'description_length', 'description_entropy',
    'screen_name_entropy', 'tweet_freq', 'followers_friends_ratio', 'has_description',
    'user_age', 'followers_growth_rate', 'friends_growth_rate', 'favourites_growth_rate',
    'listed_growth_rate', 'name_entropy', 'num_digits_in_name', 'num_digits_in_screen_name',
    'screen_name_freq', 'name_sim', 'default_profile'
  ]
};

/* Pages */
function setPage(page){
  document.querySelectorAll('.page').forEach(el=>el.classList.remove('active'));
  const target=document.getElementById('page-'+page) || document.getElementById(page);
  if(target) target.classList.add('active');
  document.querySelectorAll('.nav-link').forEach(el=>el.classList.remove('active'));
  const navBtn=document.querySelector(`.nav-link[data-page="${page}"]`);
  if(navBtn) navBtn.classList.add('active');
}
document.querySelectorAll('.nav-link').forEach(btn=>{
  btn.addEventListener('click',()=>setPage(btn.dataset.page));
});

/* Toast */
function showToast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3500);}

/* Scan overlay */
const STEPS=['Fetching account data via API…','Extracting 23 metadata features…','Running LSTM feature extractor…','Scoring with RF+ET ensemble…','Compiling authenticity report…'];
function showScan(){
  const overlay=document.getElementById('scanOverlay');const stepEl=document.getElementById('scanStep');
  overlay.classList.add('active');let i=0;
  return setInterval(()=>{stepEl.textContent=STEPS[i%STEPS.length];i++;},700);
}
function hideScan(iv){clearInterval(iv);document.getElementById('scanOverlay').classList.remove('active');}

/* Render results */
function renderResults(data){
  const section=document.getElementById('results');
  const title=document.getElementById('resultsTitle');
  const grid=document.getElementById('resultsGrid');
  const isBot=data.label==='BOT';
  const botLeaning=isBot;
  let vc='var(--green)';
  if(data.risk_level==='HIGH') vc='var(--red)';
  else if(data.risk_level==='MEDIUM') vc='var(--amber)';
  const r=62,circ=2*Math.PI*r;
  let riskBg='rgba(16,185,129,0.12)';let riskFg='var(--green)';
  if(data.risk_level==='HIGH'){riskBg='rgba(239,68,68,0.12)';riskFg='var(--red)';}
  if(data.risk_level==='MEDIUM'){riskBg='rgba(245,158,11,0.12)';riskFg='var(--amber)';}
  const insHtml=data.insights.map(i=>`<div class="insight-item"><div class="insight-icon ${i.type}">${i.icon}</div><div class="insight-text">${i.text}</div></div>`).join('');
    const metricInfo={
        "Posting Frequency":{
            desc:"Average posts per day.",
            why:"Bots often post at unusually high and consistent rates."
        },
        "Username Entropy":{
            desc:"How random the username appears.",
            why:"Bot names often look auto-generated with odd character patterns."
        },
        "Bio Completeness":{
            desc:"How much profile info is filled out.",
            why:"Real users usually add personal or descriptive bios."
        },
        "Follower / Following Ratio":{
            desc:"Compares followers to following count.",
            why:"Mass-following behavior is common in spam or automated accounts."
        }
    };
    const behHtml=data.behaviors.map(b=>{
        // Determine color based on status badge
        let fc = 'var(--green)';
        if(b.status.includes('⚡')) fc = 'var(--red)';           // Danger = Red
        else if(b.status.includes('⚠')) fc = 'var(--amber)';     // Warning = Amber
        else if(b.status.includes('✓')) fc = 'var(--green)';     // Good = Green
        
        const info=metricInfo[b.label]||{desc:"Behavioral signal",why:"This metric can indicate automated patterns."};
        return`<div class="behavior-item">
            <div class="behavior-top">
                <span class="behavior-label">${b.label}
                    <span class="info-tip" data-tip="${info.desc} Why it matters: ${info.why}">i</span>
                </span>
                <span class="behavior-value" style="font-weight:600;color:${fc};">${b.status}</span>
            </div>
            <div class="metric-desc">${b.value}</div>
            <div class="metric-why">Why it matters: ${info.why}</div>
            <div class="behavior-track">
                <div class="behavior-fill" style="width:${Math.min(b.score,100)}%;background:${fc};" data-target="${Math.min(b.score,100)}"></div>
            </div>
        </div>`;
    }).join('');
  const verdictLabel=isBot?'⚠ BOT DETECTED':(botLeaning?'⚠ BOT-LEANING':'✓ LIKELY HUMAN');
    const confidenceLabel=data.confidence>=80?'High confidence':(data.confidence>=60?'Moderate confidence':'Low confidence');
    const positive=data.insights.filter(i=>i.type==='positive').slice(0,3);
    const warning=data.insights.filter(i=>i.type==='warning').slice(0,2);
    const danger=data.insights.filter(i=>i.type==='danger').slice(0,2);
    const summaryParts=[];
    if(positive.length){summaryParts.push(`Strong human-like signals include ${positive.map(i=>i.text.toLowerCase()).join(', ')}.`);} 
    if(danger.length||warning.length){summaryParts.push(`Potential concerns: ${(danger.concat(warning)).map(i=>i.text.toLowerCase()).join(', ')}.`);} 
    summaryParts.push('AI confidence is probabilistic and based on public signals, not private data.');
  const featureChips=MODEL_EVIDENCE.selectedFeatures.slice(0,12).map(feature=>`<span class="tech-badge">${feature}</span>`).join('');
  grid.innerHTML=`
    <div class="result-card fade-in-up">
      <div class="verdict-circle">
        <svg class="verdict-svg" width="140" height="140" viewBox="0 0 140 140">
          <circle cx="70" cy="70" r="${r}" fill="none" stroke="var(--surface2)" stroke-width="10"/>
          <circle cx="70" cy="70" r="${r}" fill="none" stroke="${vc}" stroke-width="10"
            stroke-dasharray="${circ.toFixed(1)}" stroke-dashoffset="${circ.toFixed(1)}" stroke-linecap="round" id="verdictArc"/>
        </svg>
        <div class="verdict-inner">
          <div class="verdict-pct" style="color:${vc}" id="verdictPct">0%</div>
          <div class="verdict-tag">CONFIDENCE</div>
        </div>
      </div>
      <div class="verdict-label" style="color:${vc}">${verdictLabel}</div>
      <div class="verdict-conf">${data.confidence}% confidence · @${data.screen_name}</div>
      <div class="risk-row"><span class="risk-row-label">Risk Level</span><span class="risk-badge" style="background:${riskBg};color:${riskFg};">${data.risk_level}</span></div>
      <div class="risk-row"><span class="risk-row-label">Authenticity Score</span><span style="font-weight:700;color:${isBot?'var(--red)':'var(--green)'};">${data.authenticity_score}/100</span></div>
      <div class="risk-row"><span class="risk-row-label">Engine</span><span style="font-size:0.78rem;color:var(--text3);">${data.model_used}</span></div>
      <div class="confidence-pill">${confidenceLabel} · ${data.confidence}% confidence</div>
    </div>
    <div style="display:flex;flex-direction:column;gap:16px;">
            <div class="result-card fade-in-up summary-card" style="animation-delay:0.05s;">
                <h3 style="font-family:var(--font-display);font-size:0.9rem;letter-spacing:0.05em;text-transform:uppercase;color:var(--text2);margin-bottom:0.8rem;">Why This Account Was Classified as ${botLeaning?'Bot-Leaning':'Human'}</h3>
                <p class="summary-text">${summaryParts.join(' ')}</p>
            </div>
            <div class="result-card fade-in-up" style="animation-delay:0.08s;">
                <h3 style="font-family:var(--font-display);font-size:0.85rem;letter-spacing:0.05em;text-transform:uppercase;color:var(--text2);margin-bottom:0.7rem;">Model Evidence</h3>
                <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-bottom:12px;">
                  <div class="risk-row" style="margin-bottom:0;"><span class="risk-row-label">Validation AUC</span><span style="font-weight:700;color:var(--green);">${MODEL_EVIDENCE.validationAuc}</span></div>
                  <div class="risk-row" style="margin-bottom:0;"><span class="risk-row-label">Validation Accuracy</span><span style="font-weight:700;color:var(--accent);">${MODEL_EVIDENCE.validationAccuracy}</span></div>
                  <div class="risk-row" style="margin-bottom:0;"><span class="risk-row-label">5-Fold CV AUC</span><span style="font-weight:700;color:var(--amber);">${MODEL_EVIDENCE.cvAuc}</span></div>
                </div>
                <div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:8px;">${featureChips}</div>
            </div>
            <div class="result-card fade-in-up" style="animation-delay:0.1s;">
                <h3 style="font-family:var(--font-display);font-size:0.85rem;letter-spacing:0.05em;text-transform:uppercase;color:var(--text2);margin-bottom:0.7rem;">How The AI Decided</h3>
                <div class="signal-chips">
                    ${positive.map(i=>`<span class="chip good">${i.text}</span>`).join('')}
                    ${warning.map(i=>`<span class="chip warn">${i.text}</span>`).join('')}
                    ${danger.map(i=>`<span class="chip bad">${i.text}</span>`).join('')}
                </div>
            </div>
      <div class="result-card fade-in-up" style="animation-delay:0.1s;">
        <h3 style="font-family:var(--font-display);font-size:0.85rem;letter-spacing:0.05em;text-transform:uppercase;color:var(--text2);margin-bottom:1rem;">🔬 AI Insights</h3>
        <div class="insights-list">${insHtml}</div>
      </div>
      <div class="result-card fade-in-up" style="animation-delay:0.2s;">
        <h3 style="font-family:var(--font-display);font-size:0.85rem;letter-spacing:0.05em;text-transform:uppercase;color:var(--text2);margin-bottom:1rem;">📡 Behavioral Signals</h3>
        <p class="behavior-explain">Behavioral signals are human-readable indicators (posting rate, entropy, ratios) used to support the decision. They help explain the score rather than replace it.</p>
        <div class="behavior-grid">${behHtml}</div>
      </div>
            <div class="result-card fade-in-up" style="animation-delay:0.25s;">
                <h3 style="font-family:var(--font-display);font-size:0.85rem;letter-spacing:0.05em;text-transform:uppercase;color:var(--text2);margin-bottom:0.7rem;">Trust & Transparency</h3>
                <div class="trust-list">
                    <div class="trust-item">✓ We analyze public account metadata and behavioral signals only.</div>
                    <div class="trust-item">✓ Private messages, passwords, and logins are never accessed.</div>
                    <div class="trust-item">⚠ Results are probabilistic and may be incorrect for edge cases.</div>
                </div>
            </div>
      ${data.note ? `<div class="result-card fade-in-up" style="animation-delay:0.3s;border-color:rgba(245,158,11,0.3);"><p style="font-size:0.83rem;color:var(--amber);">ℹ ${data.note}</p></div>` : ''}
    </div>`;

  setPage('results');
  title.textContent=`Report: @${data.screen_name}`;
  requestAnimationFrame(()=>{
    setTimeout(()=>{
      const arc=document.getElementById('verdictArc');
      const pctEl=document.getElementById('verdictPct');
      if(arc){const tgt=circ-(data.confidence/100)*circ;arc.style.transition='stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)';arc.style.strokeDashoffset=tgt;}
      if(pctEl)animateNumber(pctEl,0,data.confidence,1200,'%');
      document.querySelectorAll('.behavior-fill').forEach(el=>{const t=el.getAttribute('data-target');if(t)el.style.width=t+'%';});
    },80);
  });
}

function animateNumber(el,from,to,dur,sfx=''){
  const start=performance.now();
  (function step(ts){const p=Math.min((ts-start)/dur,1);const e=1-Math.pow(1-p,3);el.textContent=Math.round(from+(to-from)*e)+sfx;if(p<1)requestAnimationFrame(step);})(start);
}

/* ── Twitter API Analysis ── */
function analyzeViaAPI(){
  const raw=document.getElementById('apiUsername').value.trim();
  if(!raw){showToast('Please enter a username or URL.');return;}
  const iv=showScan();
  fetch('/analyze-twitter-api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:raw})})
  .then(r=>r.json()).then(data=>{hideScan(iv);if(data.error){showToast('Error: '+data.error);return;}renderResults(data);})
  .catch(()=>{hideScan(iv);showToast('Network error — please try again.');});
}


window.addEventListener('scroll',()=>{const nav=document.querySelector('.nav');if(window.scrollY>20)nav.style.borderBottomColor='var(--border)';});
</script>
</body>
</html>"""


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True) or {}
        account = {
            'screen_name':      str(data.get('screen_name', 'unknown')).strip() or 'unknown',
            'name':             str(data.get('name', '')),
            'verified':         bool(data.get('verified', False)),
            'followers_count':  int(data.get('followers_count', 0) or 0),
            'friends_count':    int(data.get('friends_count', data.get('following_count', 0)) or 0),
            'statuses_count':   int(data.get('statuses_count', 0) or 0),
            'favourites_count': int(data.get('favourites_count', 0) or 0),
            'listed_count':     int(data.get('listed_count', 0) or 0),
            'description':      str(data.get('description', '')),
            'user_age':         int(data.get('user_age', data.get('account_age_days', 365)) or 365),
            'default_profile':  bool(data.get('default_profile', False)),
        }
        return jsonify(analyze_account(account))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-twitter-api', methods=['POST'])
def analyze_twitter_api():
    """Fetch account from GetXAPI then run the model."""
    try:
        data         = request.get_json(force=True) or {}
        bearer_token = TWITTER_BEARER_TOKEN
        url          = data.get('url', '').strip()
        debug        = bool(data.get('debug', False))

        if not bearer_token:
            return jsonify({"error": "TWITTER_BEARER_TOKEN is not configured on the server. Add it to your .env file."}), 400
        if not url:
            return jsonify({"error": "url / username is required"}), 400

        username = _parse_twitter_url(url)
        if not username:
            return jsonify({"error": "Could not extract a valid username"}), 400

        # Fetch from Twitter
        try:
            account = fetch_twitter_account(username, bearer_token)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        result = analyze_account(account)
        if debug:
            result['debug_account'] = account
            result['debug_metadata_cols'] = METADATA_COLS
            result['debug_metadata_features'] = extract_metadata_features(account).flatten().tolist()
        result['api_source'] = 'GetXAPI'
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-link', methods=['POST'])
def analyze_link():
    try:
        data     = request.get_json(force=True) or {}
        url      = data.get('url', '').strip()
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        username = _parse_twitter_url(url)
        if not username:
            return jsonify({"error": "Could not extract a valid username"}), 400

        account = {
            'screen_name': username, 'name': '', 'verified': False,
            'followers_count': 0, 'friends_count': 1, 'statuses_count': 0,
            'favourites_count': 0, 'listed_count': 0, 'description': '',
            'user_age': 365, 'default_profile': False,
        }
        result = analyze_account(account)
        result['note']       = 'Username-only analysis — use the Twitter API tab for full accuracy.'
        result['confidence'] = round(result['confidence'] * 0.5, 1)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    bundle, extractor = load_model_artifacts()
    return jsonify({
        "status":         "ok",
        "bundle_loaded":  bundle is not None,
        "extractor_loaded": extractor is not None,
        "model_type":     _model_type,
        "bundle_path":    str(BUNDLE_PATH),
        "tf_available":   TF_AVAILABLE,
        "requests_available": REQUESTS_AVAILABLE,
    })


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    print("=" * 62)
    print("  SecureProBot — AI Twitter/X Bot Detection  v2.1")
    print("=" * 62)
    load_config()
    bundle, extractor = load_model_artifacts()
    if bundle is not None and extractor is not None:
        print(f"  ✓ Hybrid LSTM + Ensemble model loaded")
        print(f"    Features: 32-dim LSTM + {len(METADATA_COLS)}-dim metadata")
    elif bundle is not None:
        print(f"  ⚠ Ensemble loaded but Keras extractor missing")
        print(f"    Expected: {EXTRACTOR_PATH}")
    else:
        print("  ⚠ No model found — using heuristic engine")
        print(f"    Expected: {BUNDLE_PATH}")
    print("  → http://127.0.0.1:5000")
    print("=" * 62)
    app.run(debug=True, host='0.0.0.0', port=5000)