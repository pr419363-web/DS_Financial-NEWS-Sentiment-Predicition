# Financial News Sentiment Prediction using Deep Learning & BERT

## Project Overview
Build a finance-focused tweet sentiment classifier that labels tweets as **Bearish**, **Bullish**, or **Neutral**.

This project includes:
- Baseline deep learning text classifiers using **Embedding + RNN / LSTM / GRU**
- Optional BERT-based fine-tuning for improved performance
- Evaluation using **accuracy**, **macro F1**, and **confusion matrix**
- A simple **Streamlit dashboard** for live inference
- Modular code and experiment documentation in a Jupyter Notebook

## Files
- `Financial_News_Sentiment_Classification.ipynb`: main notebook covering data loading, preprocessing, baseline models, and optional BERT fine-tuning
- `streamlit_app.py`: lightweight Streamlit app for sentiment prediction
- `requirements.txt`: Python package dependencies
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
- Run `Financial_News_Sentiment_Classification.ipynb` to train and evaluate baseline models.
- If you train a model, save it with a descriptive name such as `best_model.pt` and export the vocabulary if needed.
- Start the Streamlit dashboard:
  ```powershell
  streamlit run streamlit_app.py
  ```

## Notes
- The dataset is loaded directly from Hugging Face: `zeroshot/twitter-financial-news-sentiment`.
- The notebook includes a full pipeline from text cleaning to model evaluation.
- The Streamlit app uses the trained model artifacts if available.
