import os
from dotenv import load_dotenv


load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN was not found")

print("Hugging Face token loaded:", bool(token))

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=token,
    temperature=0.1,
    max_new_tokens=512
)

response = llm.invoke("Say hello in one sentence.")

print("Hugging Face response:")
print(response.content)