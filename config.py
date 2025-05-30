import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_INDEX = os.getenv("ES_INDEX_NAME", "valley_air_documents")
ES_CERT_FINGERPRINT = os.getenv("ES_CERT_FINGERPRINT")

IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_CLOUD_ENDPOINT = os.getenv("IBM_CLOUD_ENDPOINT")
IBM_CLOUD_PROJECT_ID = os.getenv("IBM_CLOUD_PROJECT_ID")

WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID") 