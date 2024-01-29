import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from  sklearn.metrics import accuracy_score, f1_score
tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold')
model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold')
model.eval()

df_cn = pd.read_csv("C:/school things/Courses/2023fall/Data and social media analysis/Finanal project/COLD/train.csv",sep = ",")
labels = ['0','1']
#1 offensive 0 not offensive
def predict_offensiveness(text):
    model_input = tokenizer(text, return_tensors='pt',padding = True)
    model_output = model(**model_input, return_dict = False)
    prediction = torch.argmax(model_output[0].cpu(),dim=-1).item()
    return prediction
df_cn["pred"]=df_cn['TEXT'].apply(predict_offensiveness)
print("predicted finished")
ground_truth_labels = df_cn['label'].tolist()
predicted_labels = df_cn['pred'].tolist()

accuracy = accuracy_score(ground_truth_labels,predicted_labels)
f1 = f1_score(ground_truth_labels, predicted_labels,average = 'weighted')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')

#predicted finished
#Accuracy: 0.9652
#F1 Score: 0.9653