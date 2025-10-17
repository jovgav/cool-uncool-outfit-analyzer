# Deployment Guide

## Local Development

The application runs locally on your machine. Follow the Quick Start guide in README.md.

## Cloud Deployment Options

### Option 1: Heroku (Recommended for beginners)

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Or download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

3. **Add Procfile**
   Create `Procfile` in root directory:
   ```
   web: python app.py
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

### Option 2: Railway

1. **Connect GitHub Repository**
   - Go to [Railway.app](https://railway.app)
   - Connect your GitHub account
   - Select your repository

2. **Configure Environment**
   - Add environment variables if needed
   - Railway will auto-detect Python and install dependencies

3. **Deploy**
   - Railway automatically deploys on git push

### Option 3: Render

1. **Create Web Service**
   - Go to [Render.com](https://render.com)
   - Connect GitHub repository
   - Choose "Web Service"

2. **Configure**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`

### Option 4: Google Cloud Platform

1. **Install Google Cloud SDK**
2. **Create App Engine App**
   ```bash
   gcloud app create
   ```

3. **Create app.yaml**
   ```yaml
   runtime: python39
   instance_class: F1
   automatic_scaling:
     min_instances: 1
     max_instances: 1
   ```

4. **Deploy**
   ```bash
   gcloud app deploy
   ```

## Important Notes

### Model Files
- Your trained model (`cool_uncool_model.pth`) is large (~15MB)
- Consider using cloud storage for model files
- Update `app.py` to load models from cloud storage

### Environment Variables
For production, set these environment variables:
- `FLASK_ENV=production`
- `PORT=5000` (or your platform's port)

### Database
- Current setup uses JSON files for labels
- For production, consider using a proper database
- SQLite is a good starting point

### Security
- Add authentication if needed
- Implement rate limiting
- Validate file uploads

## Troubleshooting

### Common Issues

1. **Port Issues**
   - Heroku uses `PORT` environment variable
   - Update `app.py` to use `os.environ.get('PORT', 3000)`

2. **Model Loading**
   - Ensure model file is included in deployment
   - Check file paths are correct

3. **Dependencies**
   - Some packages might not be available on all platforms
   - Test locally first

### Performance Optimization

1. **Model Optimization**
   - Use smaller models for faster inference
   - Consider quantization
   - Use GPU if available

2. **Caching**
   - Implement result caching
   - Use Redis or similar for production

3. **CDN**
   - Use CDN for static files
   - Optimize images

## Monitoring

### Logging
Add logging to track usage:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Analytics
Consider adding:
- Usage analytics
- Error tracking (Sentry)
- Performance monitoring

## Next Steps

1. **Choose a platform** based on your needs
2. **Test locally** with production-like settings
3. **Deploy to staging** environment first
4. **Monitor performance** and usage
5. **Iterate and improve**

Remember: Start simple and scale as needed!
