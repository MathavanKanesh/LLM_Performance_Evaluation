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

question1 = trainingset[0]["question"]
answer1 = trainingset[0]["answer_idx"]
options1 = trainingset[0]["options"]

prompt1 = "Answer the following medical question:", question1, "Your options are:", options1, ".", "Select the correct answer" # Format prompt 
print(prompt1) 

