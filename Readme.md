python3.11 -m venv venv
source venv/bin/activate # activate venv
pip install -r requirements.txt

To Access the dataset:
wget -O swift-style-embeddings.pkl https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-with-embeddings.pkl

This configuration file defines:

Model and API settings: Specifies which Llama model to use and connection details
Image processing parameters: Sets the standard size and normalization values for image preprocessing
Matching thresholds: Determines when an image match is considered "good enough"
Search settings: Controls how many alternative products to retrieve