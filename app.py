import streamlit as st
import pandas as pd
import joblib
import re
from io import StringIO
import matplotlib.pyplot as plt

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def handle_negation(text):
    # Define negation words and special modifiers
    negation_words = ['not', 'no', 'never', "n't", 'neither', 'nor']
    intensifiers = ['very', 'really', 'extremely', 'highly']

    # Convert contractions
    text = re.sub(r"n't", " not", text.lower())
    words = text.split()

    # Handle negation
    negated = False
    result = []
    for i, word in enumerate(words):
        if word in negation_words:
            negated = True
            continue
        if word in intensifiers:
            result.append(word)
            continue
        if negated:
            # Map negated sentiments
            if word in ['good', 'great', 'excellent']:
                result.append('bad')
            elif word in ['bad', 'poor', 'terrible']:
                result.append('average')
            else:
                result.append(word)
            negated = False
        else:
            result.append(word)

    return ' '.join(result)


def predict_sentiment_with_confidence(text):
    # Handle negation first
    processed_text = handle_negation(text)

    # Transform the text
    text_vectorized = vectorizer.transform([processed_text])

    # Get prediction and probability scores
    prediction = model.predict(text_vectorized)[0]
    proba = model.predict_proba(text_vectorized)[0]
    confidence = max(proba) * 100

    return prediction, confidence


def process_csv(df):
    if 'review_text' not in df.columns:
        st.error("CSV must contain a column named 'review_text'")
        return None

    # Process each review and get predictions with confidence
    predictions = []
    confidences = []

    for review in df['review_text']:
        pred, conf = predict_sentiment_with_confidence(review)
        predictions.append(pred)
        confidences.append(conf)

    df['predicted_sentiment'] = predictions
    df['confidence'] = confidences
    return df


def plot_sentiment_distribution(predictions):
    sentiment_counts = pd.Series(predictions).value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Predicted Sentiments')
    return fig


# Set page title and style
st.title('Advanced Sentiment Analysis Predictor')

# Create tabs
tab1, tab2 = st.tabs(["Single Review", "Bulk Prediction"])

# Single Review Prediction
with tab1:
    st.header("Single Review Prediction")
    review_text = st.text_area("Enter your review:", height=100)

    if st.button("Predict Sentiment"):
        if review_text.strip() != "":
            prediction, confidence = predict_sentiment_with_confidence(review_text)

            # Display results with confidence
            st.write("Predicted Sentiment:", prediction)
            st.write(f"Confidence: {confidence:.2f}%")

            # Add confidence meter
            st.progress(confidence / 100)
        else:
            st.warning("Please enter a review text")

# Bulk Prediction
with tab2:
    st.header("Bulk Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            result_df = process_csv(df)

            if result_df is not None:
                st.write("Predictions:")
                st.dataframe(result_df)

                # Display sentiment distribution chart
                st.subheader("Sentiment Distribution")
                fig = plot_sentiment_distribution(result_df['predicted_sentiment'])
                st.pyplot(fig)

                # Add download button for results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
