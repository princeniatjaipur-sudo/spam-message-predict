# SpamShield AI

SpamShield AI is an explainable spam detection application built with Python, scikit-learn, and Streamlit. It loads a labeled SMS dataset, trains a Multinomial Naive Bayes classifier with TF-IDF features, predicts whether incoming messages are spam or ham, scores risk using transparent heuristics, and stores prediction history and user feedback locally.

## Architecture and implementation plan

1. Dataset validation and cleaning
   - Download or load the UCI SMS Spam Collection.
   - Validate missing values, duplicate rows, label consistency, and required columns.
   - Standardize labels and tokenize text without introducing leakage.

2. Model training and evaluation
   - Use a TF-IDF vectorizer and Multinomial Naive Bayes pipeline.
   - Split data into train/test sets with stratification.
   - Measure accuracy, precision, recall, F1, and confusion matrix.
   - Keep the test set isolated for final evaluation.

3. Prediction and explainability
   - Score message risk using model probability plus security heuristics.
   - Surface warning signals such as urgency, suspicious URLs, and requests for sensitive information.
   - Show imported words and suspicious patterns without claiming certainty.

4. Local persistence and feedback-learning lab
   - Save prediction history with SQLite.
   - Record user feedback separately from the original training data.
   - Re-train only after approval and only if the updated model meets performance requirements.

5. Streamlit application
   - Build a polished interface with Home, Message Analyzer, Bulk Scanner, Dashboard, History, Learning Lab, and Model Information pages.
   - Keep all local data storage on-device and avoid logging raw message content beyond what is needed for the app.

## Project structure

```text
spamshield-ai/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── train_model.py
│   ├── predict.py
│   ├── explainability.py
│   ├── risk_scoring.py
│   ├── feedback.py
│   ├── database.py
│   └── __init__.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_data_validation.py
└── .venv/
```

## Dataset setup

The project uses the UCI SMS Spam Collection dataset. It is publicly available from the University of California, Irvine, and is a standard benchmark for SMS spam detection.

Download and prepare it with:

```bash
python src/train_model.py --download
```

This command downloads the zipped dataset from the official UCI archive, extracts the file, and stores it in the `data` folder as `SMSSpamCollection`.

The required format is a two-column tab-delimited file with a label and message text, for example:

```text
ham	Go to the meeting at 10am.
spam	WIN A FREE CASH PRIZE NOW!
```

If the file is missing, empty, or malformed, the application raises a clear validation error.

## Installation

```bash
cd "c:\Users\C2\Desktop\spam message project"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training the model

Train the baseline model and save it to `models/spam_model.joblib`:

```bash
python src/train_model.py
```

Optional flags:

```bash
python src/train_model.py --dataset data/SMSSpamCollection --output models/spam_model.joblib --force
```

This script performs:

- dataset loading and validation
- cleaning and deduplication
- train/test splitting with stratification
- TF-IDF transformation and Multinomial Naive Bayes training
- evaluation of accuracy, precision, recall, F1, and confusion matrix
- model persistence with joblib

## Running the Streamlit app

```bash
streamlit run app.py
```

The UI includes:

- Home
- Message Analyzer
- Bulk Scanner
- Analytics Dashboard
- Prediction History
- Learning Lab
- Model Information

## Testing

Run the automated test suite:

```bash
pytest -q
```

## Notes on model reliability

This project intentionally treats predictions as decision support tools rather than guarantees. Spam detection models can make mistakes. The project distinguishes between:

- model output
- heuristic security indicators
- human review and feedback

The risk score is a transparent score from 0 to 100, not a calibrated probability of fraud. It combines model confidence with evidence from suspicious text patterns.

## Sample predictions

The examples below are clearly marked as examples and not real system decisions.

- Example: "WIN a FREE vacation now!" -> likely Spam
- Example: "Can we meet tomorrow at 3pm?" -> likely Ham

## Troubleshooting

- If the dataset is missing: run `python src/train_model.py --download`.
- If joblib fails to load the model: retrain with `python src/train_model.py`.
- If Streamlit cannot find dependencies: activate the virtual environment and reinstall requirements.
- If CSV uploads fail: ensure the file is a UTF-8 CSV with a text column and a reasonable size.

## Privacy notice

Messages handled by the app may contain personal information. The project stores prediction records and approved feedback locally by default, avoids unnecessary raw log storage, and provides controls for deleting records and clearing history.

## Model limitations

- The system is trained on a public SMS dataset and may not generalize to all message contexts.
- Short messages or ambiguous wording can produce false positives and false negatives.
- Rule-based indicators are useful for explainability but do not prove malicious intent.
- Feedback-based retraining should be reviewed carefully before deployment.

This application is designed as a portfolio-grade demo and as a foundation for a practical message-security platform.
