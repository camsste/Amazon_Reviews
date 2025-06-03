import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data():
    """Carrega os dados originais"""
    train_path = os.path.join('data', 'raw', 'train.csv')
    test_path = os.path.join('data', 'raw', 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def preprocess_text(text):
    """Função para pré-processamento de texto individual"""
    if not isinstance(text, str):
        return ""
    
    # Limpeza básica
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remoção de stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    
    # Lematização
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def preprocess_data(train_df, test_df):
    """Pré-processamento completo dos dados"""
    # Converter polaridade para 0 (negativo) e 1 (positivo)
    train_df['sentiment'] = train_df['polarity'].apply(lambda x: 0 if x == 1 else 1)
    test_df['sentiment'] = test_df['polarity'].apply(lambda x: 0 if x == 1 else 1)
    
    # Combinar título e texto
    train_df['full_text'] = train_df['title'] + ' ' + train_df['text']
    test_df['full_text'] = test_df['title'] + ' ' + test_df['text']
    
    # Pré-processar texto
    train_df['cleaned_text'] = train_df['full_text'].apply(preprocess_text)
    test_df['cleaned_text'] = test_df['full_text'].apply(preprocess_text)
    
    # Salvar dados processados
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    train_df.to_csv(os.path.join('data', 'processed', 'train_processed.csv'), index=False)
    test_df.to_csv(os.path.join('data', 'processed', 'test_processed.csv'), index=False)
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    print("Pré-processamento concluído! Dados salvos em data/processed/")