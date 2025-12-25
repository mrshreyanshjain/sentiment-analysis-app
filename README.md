🧠 Sentiment Analysis Web App

This project is a simple and interactive Streamlit web application that uses a machine learning model to predict the sentiment of product reviews.

Supported Sentiment Categories:
- Excellent
- Good
- Neutral
- Bad

🚀 Features
- Accepts user text or CSV file input
- Preprocesses text using NLTK (tokenization, stopword removal, lemmatization)
- Predicts sentiment using a trained Naive Bayes model
- Returns predictions in table and downloadable CSV format
- Deployable on Streamlit Cloud

📁 Folder Structure
sentiment-app/
sentiment-analysis-app/
├── model.ipynb       			✅ Train the model in Jupyter
├── test.py                 		✅ Script to test the trained model
├── streamlit_server.py     		✅ Streamlit app to deploy the model
├── sentiment_model.pkl     		✅ Trained Naive Bayes model
├── vectorizer.pkl          		✅ Trained TF-IDF vectorizer
├── product_reviews_balanced_1MF.csv   ✅ Optional test input file
├── requirements.txt        		✅ All dependencies
└── README.md              		✅ Project overview


⚙️ Setup Instructions

1. Clone the Repository
   git clone https://github.com/your-username/sentiment-analysis-app.git
   cd sentiment-analysis-app

2. Create Virtual Environment (optional)
   python -m venv .venv
   .venv\Scripts\activate   (Windows)
   source .venv/bin/activate  (Mac/Linux)

3. Install Dependencies
   pip install -r requirements.txt

▶️ Run Locally
   streamlit run app.py

☁️ Deploy to Streamlit Cloud
1. Push this folder to GitHub
2. Go to https://streamlit.io/cloud
3. Click “Create App”
4. Select your repo, branch, and app.py
5. Deploy and get your public link

📄 Sample Input
"This product is absolutely amazing! I’m very satisfied."

✅ Output:
Sentiment: Excellent

