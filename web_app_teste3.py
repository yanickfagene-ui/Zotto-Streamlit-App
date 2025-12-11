import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import threading
import winsound

# ---------- Configuração da página ----------
st.set_page_config(page_title="ZOTTO - Detecção de EPI", layout="wide")
st.markdown("""
    <style>
    /* Fundo estilizado */
    .stApp {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
        font-family: 'Helvetica', sans-serif;
    }
    /* Cabeçalho ZOTTO */
    .stHeader {
        font-size:50px;
        font-weight:bold;
        color:#00FFFF;
        text-align:left;
    }
    /* Rodapé */
    .footer {
        position: fixed;
        bottom: 10px;
        right: 20px;
        font-size:12px;
        color: #ffffff99;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='stHeader'>ZOTTO</h1>", unsafe_allow_html=True)
st.write("Sistema avançado de detecção de pessoas com EPI (capacete, colete, botas)")

# ---------- Carregar modelo ----------
model = YOLO("C:/Users/User/Desktop/Zotto/best.pt")  # substitua pelo caminho correto

# ---------- Função para alerta sonoro ----------
def sirene_alert():
    winsound.Beep(2000, 500)  # frequência 2000Hz, duração 500ms

# ---------- Upload de imagem ou vídeo ----------
st.sidebar.header("Envie imagem ou vídeo")
uploaded_file = st.sidebar.file_uploader("Escolha imagem ou vídeo", type=["jpg", "png", "mp4", "mov"])
use_webcam = st.sidebar.checkbox("Usar Webcam ao vivo")

# ---------- Painel lateral ----------
epi_count_placeholder = st.sidebar.empty()
no_epi_count_placeholder = st.sidebar.empty()

# ---------- Processamento de vídeo ou imagem ----------
if uploaded_file or use_webcam:
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        epi_count = 0
        no_epi_count = 0
        
        for det in results[0].boxes:
            cls = int(det.cls[0])
            # IDs corretos do seu modelo para pessoas/EPI (ajuste se necessário)
            if cls in [0,1,2]:  
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                label = results[0].names[cls]
                confidence = det.conf[0]
                
                # Determina cor e status
                if "epi" in label.lower():
                    color = (0,255,0)  # verde
                    status = "TEM EPI"
                    epi_count += 1
                else:
                    color = (0,0,255)  # vermelho
                    status = "NÃO TEM EPI"
                    no_epi_count += 1
                    threading.Thread(target=sirene_alert).start()
                
                # Retângulo e texto
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{status}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Contadores no painel lateral
        epi_count_placeholder.markdown(f"✅ Pessoas com EPI: {epi_count}")
        no_epi_count_placeholder.markdown(f"❌ Pessoas sem EPI: {no_epi_count}")
        
        # Mostrar vídeo
        stframe.image(frame[:,:,::-1], channels="RGB")
    
    cap.release()

# ---------- Rodapé ----------
st.markdown("<div class='footer'>Developed by Yanick Fagene</div>", unsafe_allow_html=True)
