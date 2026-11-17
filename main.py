import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from datasets import load_dataset

dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

test_data = dataset["test"]

print("Number of test questions:", len(test_data))

trainingset = dataset["train"]
validationset = dataset["validation"]
testingset = dataset["test"]

print(len(trainingset))
print(len(validationset))
print(len(testingset))

# Load API key 
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key = api_key)


gemini_answers = []

#Testing loop on gemini
for i in range(len(testingset)):

    question = testingset[i]["question"]
    answer = testingset[i]["answer_idx"]
    options = testingset[i]["options"]
    prompt =  f"""
Answer the following medical question. Select the letter corresponding to the correct answer. No justification, just the letter.

"Question: ", {question}
"answer:" , {answer}
"options:", {options}

"""
#Gemini Response 
response = client.models.generate_content(
    model="gemini-3.8-flash",
        contents= prompt
)

gemini_answers.append({
    "question": question,
    "correct_answer": answer,
    "gemini_response": response.text
})

df = pd.DataFrame(gemini_answers)
df.to_csv("gemini_responses.csv")






