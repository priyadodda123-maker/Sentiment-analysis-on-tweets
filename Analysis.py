# Step 1: Import libraries
import pandas as pd
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Sample dataset
data = {
    'review': [
        'I love this product',
        'This is the worst item ever',
        'Amazing quality and great service',
        'Very bad experience',
        'I am very happy with this',
        'I hate this product'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Step 3: Text cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['review'].apply(clean_text)

# Step 4: Convert labels
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['sentiment'], test_size=0.2)

# Step 6: TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 8: Test accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: Predict new review
def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return "Positive 😊" if pred[0] == 1 else "Negative 😞"

# Example
print(predict_sentiment("This product is awesome"))
print(predict_sentiment("Very bad and disappointing"))
output:
Accuracy:0.0
Positive 😊
Positive 😊
