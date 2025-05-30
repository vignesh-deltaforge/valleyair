from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from config import IBM_CLOUD_API_KEY, IBM_CLOUD_ENDPOINT, IBM_CLOUD_PROJECT_ID, WATSONX_URL, WATSONX_PROJECT_ID

# Set up the LLM
llm = WatsonxLLM(
    model_id="ibm/granite-3-3-8b-instruct",
    url=WATSONX_URL,
    project_id=WATSONX_PROJECT_ID,
    params={
        "decoding_method": "sample",
        "max_new_tokens": 512,
        "min_new_tokens": 1,
        "temperature": 0.5,
        "top_k": 50,
        "top_p": 0.9,
    }
)

# Set up the embedding model (ibm/slate-125m-english-rtrvr-v2)
embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 500,
    EmbedParams.RETURN_OPTIONS: {'input_text': True}
}
ibm_embedding = Embeddings(
    model_id="ibm/slate-125m-english-rtrvr-v2",
    credentials={"url": IBM_CLOUD_ENDPOINT, "apikey": IBM_CLOUD_API_KEY},
    params=embed_params,
    project_id=IBM_CLOUD_PROJECT_ID
)

class IBMEmbeddingWrapper:
    def embed_documents(self, texts):
        return ibm_embedding.embed_documents(texts=texts)
    def embed_query(self, text):
        return ibm_embedding.embed_documents(texts=[text])[0] 