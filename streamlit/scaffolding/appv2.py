import streamlit as st
import requests
import os
from PIL import Image
from streamlit_option_menu import option_menu

# --- Configuración Pro ---
st.set_page_config(page_title="Real Estate AI Vision", layout="wide", page_icon="🏢")

# --- Estilos CSS Personalizados ---
st.markdown("""
<style>
    .stProgress .st-bo { background-color: #2ecc71; }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    /* Estilo para el Navbar superior */
    .nav-container { margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- NAVBAR SUPERIOR ---
# Puedes cambiar "top" por "sidebar" si prefieres que vuelva al lateral pero con este estilo pro
selected = option_menu(
    menu_title=None, 
    options=["Introducción", "Predicción Individual", "Carga por Lotes", "Configuración", "Métricas"], 
    icons=["house", "camera", "images", "gear", "bar-chart"], 
    menu_icon="cast", 
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#444", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#2b2b2b"},
    }
)

# --- LÓGICA DE SECCIONES ---

if selected == "Introducción":
    st.title("🏠 Real Estate Image Classifier")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### Inteligencia Artificial de Vanguardia
        Este sistema utiliza redes neuronales convolucionales (**EfficientNet-B3**) para el análisis automático de activos inmobiliarios.
        
        #### Capacidades:
        - **Detección de estancias:** Identifica cocinas, baños, salones, etc.
        - **Análisis de entorno:** Clasifica vistas urbanas, industriales o naturales.
        - **Optimización de flujos:** Ahorra cientos de horas en el etiquetado manual de catálogos.
        """)
    with col2:
        st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80")

elif selected == "Predicción Individual":
    st.header("📸 Análisis de Imagen Única")
    uploaded_file = st.file_uploader("Sube una imagen para analizar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        c1, c2 = st.columns(2)
        with c1: st.image(uploaded_file, use_column_width=True)
        with c2:
            if st.button("Ejecutar IA"):
                res = requests.post("http://localhost:8000/predict", files={"file": uploaded_file.getvalue()}).json()
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style='color: #2ecc71;'>{res['label']}</h2>
                    <p>Confianza del modelo: <b>{res['confidence']*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(res['confidence'])

elif selected == "Carga por Lotes":
    st.header("📦 Procesamiento Masivo")
    files = st.file_uploader("Sube múltiples archivos", accept_multiple_files=True)
    if files and st.button("Clasificar Lote"):
        cols = st.columns(4)
        for i, f in enumerate(files):
            res = requests.post("http://localhost:8000/predict", files={"file": f.getvalue()}).json()
            with cols[i % 4]:
                st.image(f, caption=f"{res['label']} ({res['confidence']:.2f})", use_column_width=True)

elif selected == "Configuración":
    st.header("⚙️ Gestión de Modelos")
    
    # Consultar qué modelo tiene la API ahora mismo
    try:
        current_api_model = requests.get("http://localhost:8000/current_model").json()['current_model']
        st.write(f"Modelo activo en el servidor: `{current_api_model}`")
    except:
        st.error("No se pudo conectar con la API")

    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if model_files:
        target_model = st.selectbox("Selecciona un nuevo modelo para cargar:", model_files)
        
        if st.button("🚀 Aplicar cambio en el Servidor"):
            with st.spinner("Cambiando modelo..."):
                # Llamada al nuevo endpoint de FastAPI
                response = requests.post(f"http://localhost:8000/update_model?model_filename={target_model}")
                
                if response.status_code == 200:
                    st.success(f"¡Hecho! El servidor ahora usa: {target_model}")
                else:
                    st.error("Error al actualizar el modelo.")

elif selected == "Métricas":
    st.header("📊 Dashboard de Rendimiento")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Mejor Precisión", "91.27%", "Top Run")
    col_b.metric("Latencia Media", "105ms", "-12ms")
    col_c.metric("Clases", "15", "Estables")
    
    st.divider()
    st.subheader("Análisis por Categoría")
    st.write("Las categorías de 'Exteriores' (Forest, Mountain) muestran un 95% de acierto, mientras que las de 'Interior' tienen un ligero solapamiento entre 'Living Room' y 'Bedroom'.")