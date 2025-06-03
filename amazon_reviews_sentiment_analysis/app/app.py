import streamlit as st
import joblib
import pandas as pd
from src.preprocessing import preprocess_text

# Configuração da página
st.set_page_config(page_title="Amazon Reviews Classifier", layout="wide")

@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    try:
        return joblib.load('models/tfidf_model.pkl')
    except:
        st.error("Modelo não encontrado. Por favor, treine o modelo primeiro.")
        return None

def main():
    st.title("📊 Amazon Reviews Sentiment Analysis")
    st.write("Classifique reviews da Amazon como positivas ou negativas")
    
    model = load_model()
    
    # Entrada de texto
    review_title = st.text_input("Título da Review:")
    review_text = st.text_area("Texto da Review:", height=150)
    
    if st.button("Classificar Sentimento"):
        if review_text:
            # Combinar e pré-processar
            full_text = f"{review_title} {review_text}"
            processed_text = preprocess_text(full_text)
            
            # Prever
            if model:
                prediction = model.predict([processed_text])[0]
                proba = model.predict_proba([processed_text])[0]
                
                # Mostrar resultados
                if prediction == 1:
                    st.success(f"✅ Positivo ({(proba[1]*100):.1f}% de confiança)")
                else:
                    st.error(f"❌ Negativo ({(proba[0]*100):.1f}% de confiança)")
                
                # Gráfico de probabilidade
                st.subheader("Probabilidades:")
                proba_df = pd.DataFrame({
                    'Sentimento': ['Negativo', 'Positivo'],
                    'Probabilidade': [proba[0], proba[1]]
                })
                st.bar_chart(proba_df.set_index('Sentimento'))
        else:
            st.warning("Por favor, insira o texto da review para classificar.")

if __name__ == "__main__":
    main()