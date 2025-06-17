from langchain_ibm import WatsonxLLM
from dotenv import load_dotenv
import os

load_dotenv()
project_id = os.environ["WATSONX_PROJECT_ID"]
watsonx_url = os.environ["WATSONX_URL"]

params = {
    'max_new_tokens': 500,
    'stop_sequences': ['Human:']
}

granite38b = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url=watsonx_url,
    project_id=project_id,
    params=params,
)

