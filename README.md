# Financial News Sentiment Prediction

This project explores how finance-related tweets reflect investor mood. I started with simple RNN models to understand the basics of sequence modeling, then moved on to BERT-based fine-tuning to improve both accuracy and generalization. The goal was to compare classic deep learning approaches with transformer-based models on short, noisy financial text.

## Repository
- Correct repo name: `DS_Financial-NEWS-Sentiment-Prediction`
- GitHub repo: https://github.com/your-username/DS_Financial-NEWS-Sentiment-Prediction

## Project Overview
This project builds a finance-focused sentiment classifier for short financial texts, with labels for **Bearish**, **Bullish**, and **Neutral**. The models are designed to handle market-related tweets and headlines, where the language is short, noisy, and often full of jargon.

This project includes:
- Baseline deep learning text classifiers using **Embedding + RNN / LSTM / GRU**
- Optional BERT-based fine-tuning for improved performance
- Evaluation using **accuracy**, **macro F1**, and **confusion matrix**
- A simple **Streamlit dashboard** for live inference
- A notebook that documents the end-to-end workflow from data cleaning to model comparison

## Why I Built This
I wanted to explore how investor sentiment on Twitter reflects market mood. Starting with RNNs helped me understand the basics of sequence modeling, but they struggled with longer dependencies and more nuanced wording. Fine-tuning BERT gave a clear boost in F1-score and handled financial jargon much better.

## Challenges and Learnings
- Cleaning tweets was tricky because financial jargon, ticker symbols, and sarcasm often confused the model.
- Neutral tweets were some of the hardest examples to classify because they often blend optimism and caution without an explicit direction.
- RNNs trained faster and were easier to debug, but BERT generalized better on ambiguous language.
- I also learned that a lot of errors came from sarcasm and short bursts of market commentary, not just weak model architecture.

## Dataset
Dataset: Twitter Financial News Sentiment on Hugging Face
- `zeroshot/twitter-financial-news-sentiment`
- Dataset page: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

This dataset is useful for this task because it contains short financial texts with sentiment labels that align well with market commentary and headline-style language.

## Files
- `Financial_News_Sentiment_Classification.ipynb`: main notebook covering data loading, preprocessing, baseline models, and optional BERT fine-tuning
- `streamlit_app.py`: lightweight Streamlit app for sentiment prediction
- `requirements.txt`: Python package dependencies
- `best_lstm_baseline.pt`, `best_lstm.pt`, `best_gru.pt`, `best_rnn.pt`: trained model checkpoints
- `vocab.json`: token vocabulary used by the baseline models
- `.gitignore`: files and folders to ignore in Git

## Setup
1. Create and activate a Python environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Launch the notebook:
   ```powershell
   jupyter notebook
   ```

## Usage
- Open `Financial_News_Sentiment_Classification.ipynb` to train and evaluate the baseline models.
- If you train a model, save it with a descriptive name such as `best_model.pt` and export the vocabulary if needed.
- Start the Streamlit dashboard:
  ```powershell
  streamlit run streamlit_app.py
  ```

## Model comparison and observations
The baseline sequence models give a strong starting point for this task, but the transformer approach usually improves the handling of context and subtle language. Looking at the predictions, I noticed that subtle wording mattered a lot: tweets with sarcasm, hedging language, or understatement were frequently mistaken for **Neutral** when they were actually intended to be **Bearish** or **Bullish**.

For example:
- **Bearish tweets with sarcasm or a cautious tone** were often predicted as Neutral
- **Buzzword-heavy headlines** sometimes confused the model when the sentiment was implied rather than directly stated
- **Neutral** statements involving earnings or macro news were harder to separate from weak bullish or bearish signals

A good example is a tweet like: *“Great job $AAPL, losing billions again!”* This is clearly negative in tone, but the model sometimes treats it as Neutral because sarcasm and irony are difficult for short-text sentiment systems to detect.

This kind of error analysis is useful because it shows that the toughest cases do not always come from raw model weakness; they often come from ambiguous finance language and the lack of deeper market context in short tweets.

## Sample outputs
Below is an example of the kind of input/output the app is designed to handle:

```text
Input: "Stocks dip as inflation fears rise around tech earnings"
Predicted sentiment: Bearish

Input: "Chipmakers rally as AI spending outlook beats forecasts"
Predicted sentiment: Bullish

Input: "The company says it is monitoring demand conditions"
Predicted sentiment: Neutral
```

The Streamlit dashboard also visualizes confidence scores as a probability bar chart for each class.

## Screenshots
Add screenshots like these to make the project feel more hands-on:
- Confusion matrix from model evaluation
- Streamlit dashboard UI showing prediction inputs and output probabilities
- Example prediction: *Tweet: “$TSLA is doomed” → Predicted: Bearish*

```markdown
![Confusion Matrix](images/confusion_matrix.png)
![Streamlit Dashboard](images/streamlit_dashboard.png)
![Example Prediction](images/example_prediction.png)
```

## Notes
- The notebook includes a full pipeline from text cleaning to model evaluation.
- The Streamlit app uses the trained model artifacts if available.
- In my experience, the hardest part of the workflow was cleaning the text without accidentally removing financial meaning, especially around patterns like ticker symbols, currencies, and market shorthand.

## Deliverables
- Baseline RNN/LSTM/GRU sentiment classifiers
- Optional BERT-based sentiment model
- Model comparison notebook with evaluation metrics
- Streamlit interface for interactive prediction
- Documented workflow and reflection on model behavior
