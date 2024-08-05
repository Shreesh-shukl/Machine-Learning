import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import random

# Download NLTK resources (if not already downloaded)
nltk.download('movie_reviews')
nltk.download('punkt')

# Load movie reviews dataset
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Convert documents to DataFrame
data = pd.DataFrame(documents, columns=['review', 'sentiment'])

# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Keep only alphabetic tokens
    return ' '.join(tokens)

data['review'] = data['review'].apply(preprocess_text)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Example inputs for testing
inputs = [
    "This movie was amazing! The acting was superb and the storyline was engaging.",
    "I regret watching this film. The acting was terrible and the plot was boring."
]

# Process and predict sentiments for inputs
print("\nExample Predictions:")
for input_text in inputs:
    input_text_processed = preprocess_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([input_text_processed])
    prediction = rf_classifier.predict(input_vectorized)[0]
    
    print(f"Input Text: {input_text}")
    print(f"Predicted Sentiment: {'Positive' if prediction == 'pos' else 'Negative'}")
    print()

# Save the model
# joblib.dump(rf_classifier, 'sentiment_model.pkl')
