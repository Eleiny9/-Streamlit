import streamlit as st
import joblib
import numpy as np

# Cargar el modelo 
@st.cache_resource
def load_model():
    return joblib.load("tree_default_42.pkl")

model = load_model()

st.title("Clasificador de Vinos")
st.write("Introduce las características químicas del vino para predecir su clase.")

alcohol = st.slider("Alcohol", 10.0, 15.0, step=0.1, value=13.0)
ash = st.slider("Cenizas (Ash)", 1.0, 3.5, step=0.1, value=2.5)
flavanoids = st.slider("Flavonoides", 0.0, 5.0, step=0.1, value=2.5)
color_intensity = st.slider("Intensidad de color", 1.0, 13.0, step=0.1, value=5.0)
proline = st.slider("Proline", 200.0, 1700.0, step=50.0, value=750.0)

# Botón para realizar predicción
if st.button("Predecir"):
    try:
        features = np.array([[alcohol, ash, flavanoids, color_intensity, proline]])
        
        prediction = model.predict(features)[0]

        wine_classes = ["Clase 0", "Clase 1", "Clase 2"]
        result = wine_classes[prediction]
        st.subheader("Resultado de la predicción:")
        st.write(f"El vino pertenece a: **{result}**")
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
