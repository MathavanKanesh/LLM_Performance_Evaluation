import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from datasets import load_dataset

if os.path.exists("gemini_responses.csv"):
    gemini_answers = pd.read_csv("gemini_responses.csv").to_dict("records")
else:
    gemini_answers = []

dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

test_data = dataset["test"]

trainingset = dataset["train"]
testingset = dataset["test"]

print("Length of training set:", len(trainingset))
print("Length of testing set:",  len(testingset))

# Load API key 
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key = api_key)



#Testing loop on gemini with retry logic
for i in range(len(gemini_answers), len(testingset)):
    print(f"Sending question {i + 1} to Gemini...")
    question = testingset[i]["question"]
    answer = testingset[i]["answer_idx"]
    options = testingset[i]["options"]
    prompt =  f"""
Answer the following medical question. Select the letter corresponding to the correct answer. No justification, just the letter.

"Question: ", {question}
"options:", {options}

"""

    question = testingset[i]["question"]
    answer = testingset[i]["answer_idx"]
    options = testingset[i]["options"]

    prompt = f"""
Answer the following medical question. Select the letter corresponding to the correct answer. No justification, just the letter.

Question: {question}
Options: {options}
"""

    # Gemini Response with retry logic, ensuring it doesn't exceed request limit 
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )
            break
        except Exception:
            if attempt < 2:
                time.sleep(10)
            else:
                raise

    gemini_answers.append({
        "question": question,
        "correct_answer": answer,
        "gemini_response": response.text
    })
    # Save after every successful question in case gemini crashes 
    df = pd.DataFrame(gemini_answers)
    df.to_csv("gemini_responses.csv", index=False)