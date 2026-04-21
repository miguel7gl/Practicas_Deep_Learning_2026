import streamlit as st
import requests

# --- Configuración de la página ---
st.set_page_config(
    page_title="Chachi Pistachi AI",
    layout="centered"
)

# --- CSS Minimalista y Elegante ---
st.markdown("""
<style>
    /* Tipografía y fondo */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #fafafa;
    }

    /* Título sobrio */
    .main-title {
        color: #1a1a1a;
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .main-subtitle {
        color: #666666;
        font-size: 16px;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Tarjetas de contenido */
    div.stFileUpload {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
    }

    /* Botón personalizado (Sin iconos) */
    div.stButton > button:first-child {
        background-color: #2b2b2b;
        color: #ffffff;
        border-radius: 4px;
        border: none;
        width: 100%;
        height: 45px;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: background 0.2s ease;
    }

    div.stButton > button:first-child:hover {
        background-color: #444444;
        border: none;
        color: #ffffff;
    }

    /* Contenedor de resultados */
    .result-box {
        background-color: #ffffff;
        border: 1px solid #2ecc71;
        border-left: 4px solid #2ecc71;
        padding: 24px;
        border-radius: 4px;
        margin-top: 20px;
    }

    .result-label {
        color: #27ae60;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .result-value {
        color: #1a1a1a;
        font-size: 24px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Contenido ---
st.markdown('<h1 class="main-title">Chachi Pistachi AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Clasificación de imágenes mediante redes neuronales</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Seleccionar archivo de imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen centrada y limpia
    col_img_1, col_img_2, col_img_3 = st.columns([1, 2, 1])
    with col_img_2:
        st.image(uploaded_file, use_column_width=True)
    
    st.markdown("---")
    
    if st.button("Ejecutar Predicción"):
        file_bytes = uploaded_file.getvalue()
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
        
        try:
            with st.spinner('Procesando datos...'):
                response = requests.post("http://localhost:8000/predict", files=files)
                response.raise_for_status()
                result = response.json()

            # --- Diseño de Resultado Minimalista ---
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Resultado del análisis</div>
                <div class="result-value">{result['label']}</div>
                <div style="margin-top: 15px;">
                    <span style="font-size: 14px; color: #666;">Nivel de confianza: {result['confidence']*100:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Barra de progreso sutil para la confianza
            st.progress(result['confidence'])

        except Exception as e:
            st.error(f"Error en la conexión: {e}")

# Pie de página
st.markdown("<br><p style='text-align: center; color: #999; font-size: 12px;'>Sistema de Gestión de Imágenes 2026</p>", unsafe_allow_html=True)