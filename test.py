import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load API key 
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")

# Check API key
if not api_key:
    raise ValueError("Gemini API key not found! Check your .env file.")

print("API key loaded successfully!")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send test question
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is 2 + 2? Respond with only the answer."
)

# Response 
print("Gemini response:", response.text)