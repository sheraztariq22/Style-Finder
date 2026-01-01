# Fashion Style Analyzer with Google Gemini

A multimodal RAG (Retrieval-Augmented Generation) pipeline that combines computer vision with Google Gemini AI to analyze fashion images and provide detailed product recommendations.

## Requirements

- **Python 3.11.3**
- Google API Key (free tier available)
- Dataset file: `swift-style-embeddings.pkl`

## Features

- **Image Encoding**: Uses ResNet50 to convert fashion images into vector embeddings
- **Similarity Matching**: Finds visually similar items using cosine similarity
- **AI-Powered Analysis**: Leverages Google Gemini's vision capabilities for detailed fashion analysis
- **Interactive Interface**: User-friendly Gradio web interface

## Project Structure

```
STYLE-FINDER/
├── models/
│   ├── image_processor.py    # Image encoding and similarity matching
│   └── llm_service.py         # Google Gemini API integration
├── utils/
│   └── helpers.py             # Utility functions
├── examples/                   # Example fashion images
├── app.py                     # Main application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## Installation

### 1. Clone or download the project

### 2. Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy your API key

### 5. Set up environment variables

Create a `.env` file in the root directory:

```bash
# Copy the example file
cp .env.example .env

# Or create manually on Windows
copy .env.example .env
```

Edit the `.env` file and add your API key:

```
GOOGLE_API_KEY=your-actual-api-key-here
```

**Important:** Never commit your `.env` file to Git! It's already in `.gitignore`.

### 6. Verify your dataset

Ensure `swift-style-embeddings.pkl` is in the root directory with these columns:
- `Image URL`: URL of the fashion image
- `Item Name`: Name of the clothing item
- `Price`: Price of the item
- `Link`: Purchase link
- `Embedding`: Pre-computed feature vector

## Usage

### Run the application

```bash
python app.py
```

The application will:
1. Load the dataset
2. Initialize the Gemini API client
3. Launch at `http://127.0.0.1:5000`
4. Optionally create a public share link

### Using the interface

1. **Upload an image**: Click the upload area or drag and drop a fashion image
2. **Or use examples**: Click one of the example buttons to load a pre-loaded image
3. **Analyze**: Click "Analyze Style" button
4. **View results**: Get detailed fashion analysis with product recommendations

## How It Works

### 1. Image Encoding (ResNet50)
```python
# Image → Feature Vector (2048 dimensions)
user_encoding = image_processor.encode_image(image_path, is_url=False)
```

### 2. Similarity Matching (Cosine Similarity)
```python
# Find most similar items in database
closest_row, similarity_score = image_processor.find_closest_match(
    user_encoding['vector'], 
    dataset
)
```

### 3. AI Analysis (Google Gemini)
```python
# Generate detailed fashion response
bot_response = llm_service.generate_fashion_response(
    user_image_base64=user_encoding['base64'],
    matched_row=closest_row,
    all_items=all_items,
    similarity_score=similarity_score
)
```

### 4. Response Formatting
The AI response includes:
- Clothing item descriptions
- Color and pattern analysis
- Style categorization
- Product recommendations with prices and purchase links

## Configuration

Edit `config.py` to customize behavior:

```python
# Model selection
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Latest vision model

# Generation parameters
TEMPERATURE = 0.2  # Lower = more deterministic (0.0-1.0)
TOP_P = 0.6        # Nucleus sampling (0.0-1.0)
MAX_TOKENS = 2000  # Maximum response length

# Similarity threshold
SIMILARITY_THRESHOLD = 0.8  # 0.0-1.0 (higher = stricter matching)

# Image processing
IMAGE_SIZE = (224, 224)  # ResNet50 input size
```

## Troubleshooting

### API Key Issues

**Error: "Google API key not found"**
```bash
# Check if .env file exists
ls -la .env  # macOS/Linux
dir .env     # Windows

# Verify content
cat .env     # macOS/Linux
type .env    # Windows

# Should contain:
# GOOGLE_API_KEY=your-key-here
```

**Error: "API key not valid"**
- Regenerate your API key at https://makersuite.google.com/app/apikey
- Ensure no extra spaces in `.env` file
- Try using quotes: `GOOGLE_API_KEY="your-key-here"`

### Installation Issues

**Error: "No module named 'google.generativeai'"**
```bash
pip install google-generativeai==0.8.3
```

**Error: "torch not found" or CUDA issues**
```bash
# CPU-only version (lighter)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

**Error with Python version**
```bash
# Verify Python version
python --version  # Should show 3.11.3

# If wrong version, use:
python3.11 -m venv venv
```

### Runtime Issues

**Error: "Dataset file not found"**
- Ensure `swift-style-embeddings.pkl` is in the root directory
- Check file permissions

**Slow processing**
- First run downloads ResNet50 model (~100MB)
- Subsequent runs will be faster
- GPU acceleration used if available

**Incomplete responses**
- Check API quota at Google AI Studio
- Increase `MAX_TOKENS` in config.py
- Check internet connection

### Port Already in Use

```bash
# Error: "Address already in use"
# Change port in app.py:
demo.launch(server_port=5001)  # Try different port
```

## API Costs

Google Gemini pricing (December 2024):

| Model | Free Tier | Paid Tier |
|-------|-----------|-----------|
| Gemini 2.0 Flash | 15 RPM, 1500 RPD | Pay per request |
| Gemini Pro Vision | 15 RPM, 1500 RPD | Pay per request |

**RPM** = Requests Per Minute  
**RPD** = Requests Per Day

Check current pricing: https://ai.google.dev/pricing

## Development

### Project Dependencies

```bash
# Core ML
torch==2.5.1              # PyTorch
torchvision==0.20.1       # Vision models
transformers==4.46.3      # NLP models

# AI API
google-generativeai==0.8.3  # Gemini API

# Data Science
pandas==2.2.3             # Data manipulation
numpy==1.26.4            # Numerical operations
scikit-learn==1.5.2      # ML utilities

# Web Interface
gradio==5.22.0           # UI framework

# Utilities
python-dotenv==1.0.1     # Environment variables
requests==2.32.0         # HTTP requests
```

### Adding New Features

To customize the fashion analysis prompts, edit `models/llm_service.py`:

```python
def generate_fashion_response(self, ...):
    # Modify this section for custom prompts
    assistant_prompt = (
        f"Your custom prompt here..."
    )
```

### Testing

```bash
# Test with example images
python app.py

# Then in browser:
# 1. Click "Use Example 1"
# 2. Click "Analyze Style"
# 3. Verify output
```

## Best Practices

1. **API Key Security**
   - Never commit `.env` to version control
   - Use `.gitignore` to exclude sensitive files
   - Rotate API keys periodically

2. **Rate Limiting**
   - Implement delays for batch processing
   - Monitor API usage in Google AI Studio
   - Handle rate limit errors gracefully

3. **Image Quality**
   - Use high-resolution images (min 224x224)
   - Ensure good lighting and clear visibility
   - Avoid heavily filtered or edited images

4. **Error Handling**
   - Always check API responses
   - Implement fallback mechanisms
   - Log errors for debugging

## Contributing

Feel free to fork and submit pull requests!

## License

This project is provided for educational purposes.

## Support

For issues:
1. Check the Troubleshooting section
2. Review logs in terminal
3. Verify API key and quota
4. Check Google AI Studio status

## Acknowledgments

- **Google Gemini** for multimodal AI capabilities
- **PyTorch** for deep learning framework
- **Gradio** for the web interface
- **ResNet50** for image feature extraction