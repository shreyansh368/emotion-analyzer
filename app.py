import os
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load BERT-based model
MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Flask setup
app = Flask(__name__)

def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)

    # Softmax on the output logits to get probabilities
    scores = output.logits[0].numpy()
    scores = np.exp(scores) / np.sum(np.exp(scores))  # softmax
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    
    # Get the sentiment with the highest probability
    sentiment = {labels[i]: float(scores[i]) for i in range(len(scores))}
    top = max(sentiment, key=sentiment.get)
    return f"{top} ({round(sentiment[top]*100, 2)}%)"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        text = request.form["text"]
        sentiment = analyze_sentiment(text)
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    # Get the port number from the environment variable (Render will provide this)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
