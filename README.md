# Multi-Model Sentiment & Capital Intel System (Streamlit)

## Files in this zip (put them in your repo's main folder)
- sentiment_capital_app.py  - the app (REPLACES your old one; keep a backup copy)
- model_engine.py           - loads your model files
- requirements.txt          - set tensorflow== to YOUR version (pip show tensorflow)
- .streamlit/config.toml    - dark theme
- shrink_models.py          - optional: makes .h5 files smaller (run locally, don't need to upload)

## Keep your own files next to them (same folder)
text_preprocessing.py, text_cleaner.pkl, tfidf.pkl,
sentiment_lr_model.pkl, capital_lr_model.pkl, sentiment_encoder.pkl, capital_encoder.pkl,
lstm_tokenizer.pkl, lstm_params.pkl, best_sentiment_lstm.h5 (or sentiment_lstm_model.h5),
best_capital_lstm.h5, sentiment_lstm_encoder.pkl, capital_lstm_encoder.pkl,
federated_sentiment_model.pkl (optional - adds a 3rd engine)

## Run
streamlit run sentiment_capital_app.py
Streamlit Cloud: Main file path = sentiment_capital_app.py

## If a prediction looks wrong / fails
Open "Model diagnostics" at the bottom of the app - it lists every file, its size and what is inside.
Sidebar (arrow top-left): LSTM padding must match your training code (post/pre).
