# CapRover Deployment Guide

## Prerequisites

1. CapRover VPS server set up and running
2. Git repository with your code
3. OpenAI API key

## Deployment Steps

### 1. Prepare Your Files

Make sure these files are in your repository:
- `app.py` - Streamlit application
- `config.py` - Configuration file
- `captain-definition` - CapRover definition
- `Dockerfile` - Docker build instructions
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (or set in CapRover dashboard)
- `constitution.pdf` - PDF file (if needed)
- `vector_store/` - Pre-built vector store (IMPORTANT!)

### 2. Build Vector Store (Before Deployment)

**Option A: Build locally and upload**
```bash
# On your local machine
python ingest.py

# Commit the vector_store folder to git
git add vector_store/
git commit -m "Add vector store"
git push
```

**Option B: Build on server (after first deployment)**
- Deploy the app first
- SSH into your CapRover server
- Navigate to the app directory
- Run `python ingest.py` inside the container

### 3. Set Environment Variables in CapRover

In CapRover dashboard:
1. Go to your app settings
2. Navigate to "App Configs" → "Environment Variables"
3. Add:
   ```
   OPENAI_API_KEY=your-actual-api-key-here
   ```

### 4. Deploy to CapRover

**Method 1: Git Deployment (Recommended)**
1. In CapRover dashboard, create a new app
2. Go to "Deployment" tab
3. Enable "Git Deployment"
4. Add your Git repository URL
5. Set branch to `main` or `master`
6. Click "Save & Update"

**Method 2: Manual Deployment**
1. In CapRover dashboard, create a new app
2. Go to "Deployment" tab
3. Upload your project as a tar/zip file
4. CapRover will automatically detect `captain-definition` and build

### 5. Verify Deployment

1. Check app logs in CapRover dashboard
2. Visit your app URL (provided by CapRover)
3. Test the chat interface

## Important Notes

- **Port**: The Dockerfile exposes port 80 (CapRover requirement)
- **Vector Store**: Must exist before app starts, or app will show error
- **Environment Variables**: Set `OPENAI_API_KEY` in CapRover dashboard
- **Memory**: Ensure your VPS has enough RAM (recommended: 2GB+)

## Troubleshooting

### App shows "Vector store not found"
- Run `python ingest.py` to create the vector store
- Make sure `vector_store/` folder is in your repository

### App crashes on startup
- Check logs in CapRover dashboard
- Verify `OPENAI_API_KEY` is set correctly
- Ensure all dependencies are in `requirements.txt`

### Slow responses
- Consider using a larger VPS instance
- Check network connectivity to OpenAI API

