import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
from scipy.special import softmax

# Sample DataFrame

df = pd.read_csv("/content/olid-training-v1.0.tsv",sep='\t')
labels = ['NOT', 'OFF']
# Function to preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Preprocess tweets in the DataFrame
df['tweet'] = df['tweet'].apply(preprocess)
task='offensive'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL)

# Create lists to store predictions and true labels
predicted_labels = []
true_labels = []

# Iterate through the DataFrame and make predictions
for index, row in df.iterrows():
    text = row['tweet']
    true_label = row['subtask_a']

    # Tokenize and make predictions
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Get predicted label
    predicted_label_index = np.argmax(scores)
    predicted_label = labels[predicted_label_index]

    # Store predicted and true labels
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

# Calculate accuracy and F1 score
accuracy = accuracy_score(true_labels, predicted_labels)
f1_score = f1_score(true_labels, predicted_labels, pos_label='OFF')

#accuracy = 0.8567220543806646
#f1 = 0.78465206039278