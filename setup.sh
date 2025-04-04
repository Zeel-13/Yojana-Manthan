#!/bin/bash

echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

echo "📦 Downloading NLTK resources..."
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"

echo "✅ Setup complete. You're ready to run the app!"
