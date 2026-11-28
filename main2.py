import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

if os.path.exists("OPENAI_responses.csv"):
    NVIDIA_answers = pd.read_csv("OPENAI_responses.csv").to_dict("records")
else:
    NVIDIA_answers = []

dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

test_data = dataset["test"]

trainingset = dataset["train"]
testingset = dataset["test"]

print("Length of training set:", len(trainingset))
print("Length of testing set:",  len(testingset))

# Load API key 
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


#Testing loop with retry logic
for i in range(len(NVIDIA_answers), len(testingset)):
    print(f"Sending question {i + 1} ...")
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

    #  Response with retry logic, ensuring it doesn't exceed request limit 
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b:free",
                messages=[
                    {"role": "user", "content": prompt}
                    ]
                    )
            break
        except Exception:
            if attempt < 2:
                time.sleep(10)
            else:
                raise

    NVIDIA_answers.append({
        "question": question,
        "correct_answer": answer,
        "OPENAI_response": response.choices[0].message.content
    })
    # Save after every successful question in case  crashes 
    df = pd.DataFrame(NVIDIA_answers)
    df.to_csv("NVIDIA_answers.csv", index=False)