import ollama
import requests
import json

CONST_N_CTX = 14000
CONST_MAX_CTX = 500

# Define the URL of the Ollama REST API
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Specify the model and the prompt
payload = {
    "model": "hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q8_0",  # Change this to your preferred model (e.g., 'mistral', 'gemma')
    "prompt": "Tell me a fun fact about space.",
    "stream": False
}

from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

Prompt="why is sky bllue"

try:
        gen_opts = {
            "num_predict": CONST_MAX_CTX,
            "num_ctx": CONST_N_CTX,
            "temperature": 0.6,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
        }
        output = client.generate(
            model="gemma3:12b-it-qat", prompt=Prompt, options=gen_opts
        )
        print(output)
except Exception as e:
        print(e)

# # Send the request to the Ollama server
# try:
#     response = requests.post(OLLAMA_API_URL, json=payload)
#
#     # Check if the request was successful
#     if response.status_code == 200:
#         data = response.json()
#         print("Response from Ollama:")
#         print(data["response"])
#     else:
#         print(f"Error: {response.status_code} - {response.text}")
#
# except requests.exceptions.RequestException as e:
#     print(f"Request failed: {e}")
