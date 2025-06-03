import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def load_processed_data():
    """Carrega os dados pré-processados"""
    train_path = os.path.join('data', 'processed', 'train_processed.csv')
    test_path = os.path.join('data', 'processed', 'test_processed.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def train_tfidf_model():
    """Treina e salva modelo TF-IDF + Regressão Logística"""
    train_df, test_df = load_processed_data()
    
    # Criar pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Treinar
    model.fit(train_df['cleaned_text'], train_df['sentiment'])
    
    # Salvar modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'tfidf_model.pkl'))
    
    return model

if __name__ == "__main__":
    model = train_tfidf_model()
    print("Modelo TF-IDF treinado e salvo em models/tfidf_model.pkl")