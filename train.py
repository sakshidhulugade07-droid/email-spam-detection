import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_model():
    dataset_path = 'dataset.csv'
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Error: {dataset_path} not found. Please create it first.")
        
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    X = df['Message']
    y = df['Category']  # Labels: 'spam' or 'ham'
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building and training the ML pipeline...")
    # Pipeline handles text vectorization seamlessly inside the model file
    model_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english', lowercase=True)),
        ('nb', MultinomialNB())
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    predictions = model_pipeline.predict(X_test)
    print(f"Model Training Complete. Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    
    # Save the model pipeline file
    model_filename = 'spam_detector_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model_pipeline, file)
        
    print(f"Saved trained model to '{model_filename}'")

if __name__ == "__main__":
    train_model()