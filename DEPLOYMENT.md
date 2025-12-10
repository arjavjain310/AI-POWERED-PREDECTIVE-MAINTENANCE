# Deployment Guide

## 🚀 Deploy Your Predictive Maintenance Dashboard

### Option 1: Streamlit Cloud (Recommended - FREE & Easiest)

Streamlit Cloud is the easiest way to deploy your dashboard for free.

#### Steps:

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Deploy Your App:**
   - Click "New app"
   - Select your repository: `arjavjain310/AI-POWERED-PREDECTIVE-MAINTENANCE`
   - Main file path: `app/dashboard.py`
   - Branch: `main`
   - Click "Deploy"

3. **Your app will be live at:**
   - `https://ai-powered-predective-maintenance.streamlit.app`
   - (or similar URL provided by Streamlit)

#### Requirements:
- ✅ Your code is already on GitHub
- ✅ `requirements.txt` exists
- ✅ Dashboard file is at `app/dashboard.py`
- ✅ All dependencies are listed

---

### Option 2: Railway (Alternative)

1. Go to: https://railway.app
2. Sign up with GitHub
3. New Project → Deploy from GitHub
4. Select your repository
5. Add start command: `streamlit run app/dashboard.py --server.port $PORT`
6. Deploy

---

### Option 3: Render

1. Go to: https://render.com
2. Sign up with GitHub
3. New Web Service
4. Connect your repository
5. Build command: `pip install -r requirements.txt`
6. Start command: `streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0`
7. Deploy

---

### Option 4: Heroku

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: streamlit run app/dashboard.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

---

## 📝 Pre-Deployment Checklist

- [x] Code pushed to GitHub
- [x] requirements.txt exists
- [x] Dashboard file is ready
- [x] .streamlit/config.toml configured
- [ ] Data file path configured (if using cloud storage)

## 🔧 Configuration for Deployment

### For Streamlit Cloud:
- No additional config needed
- Data files should be in the repository or use cloud storage
- Large data files (>100MB) should be excluded via .gitignore

### Data Handling:
Since your data file is large, consider:
1. **Option A:** Generate data on first run (add to dashboard)
2. **Option B:** Use cloud storage (S3, Google Drive, etc.)
3. **Option C:** Include sample data in repo (smaller subset)

---

## 🎯 Recommended: Streamlit Cloud

**Why Streamlit Cloud?**
- ✅ Free forever
- ✅ Automatic deployments from GitHub
- ✅ No credit card required
- ✅ Easy to set up (5 minutes)
- ✅ Custom domain support
- ✅ Perfect for final year projects

**Get started:** https://share.streamlit.io/

