import streamlit as st
import numpy as np
import pandas as pd
import io
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq

# 1. Config Halaman
st.set_page_config(page_title="ANKAYU FREQ Pro", layout="wide")

# 2. Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0F0F0F; color: white; }
    .pod-card {
        background: #1A1A1A;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #FF0000;
        margin-bottom: 20px;
    }
    div[data-baseweb="input"] input {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
    .active-red {
        background: linear-gradient(90deg, #FF0000 0%, #B30000 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 25px rgba(255, 0, 0, 0.4);
    }
    .inactive-dark {
        background: #1E1E1E;
        color: #666;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    }
    .title-red { font-family: 'Arial Black', sans-serif; font-size: 50px; color: white; margin-bottom: -10px; }
    .accent-red { color: #FF0000; }
    hr { border: 0; height: 1px; background: #333; margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

# 3. Header
st.markdown('<h1 class="title-red">ANKAYU<span class="accent-red"> FREQ</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #888;">#MultimediaProject | Universitas Khairun | Mahasiswa: A\'an , Kautsar, & Bayu</p>', unsafe_allow_html=True)

# --- FITUR: PRESENTASI 4 KATEGORI ---
st.write("🎯 **Presentation Mode:**")
p1, p2, p3, p4 = st.columns(4)

if 'current_f' not in st.session_state:
    st.session_state.current_f = 440.0

if p1.button("01. Infrasound (10 Hz)"): st.session_state.current_f = 10.0
if p2.button("02. Hearing (440 Hz)"): st.session_state.current_f = 440.0
if p3.button("03. Ultrasound (40 kHz)"): st.session_state.current_f = 40000.0
if p4.button("04. Hypersound (2 GHz)"): st.session_state.current_f = 2000000000.0

# 4. Input Area
col_ctrl1, col_ctrl2 = st.columns(2)

with col_ctrl1:
    st.markdown('<div class="pod-card"><h3 style="margin:0; color:white;">🎚️ Master Control</h3>', unsafe_allow_html=True)
    freq_input = st.number_input("Freq (Hz):", min_value=0.0, max_value=10**13.0, value=st.session_state.current_f, format="%.2f")
    wave_type = st.radio("Waveform", ["Sine", "Square", "Sawtooth"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_ctrl2:
    st.markdown('<div class="pod-card"><h3 style="margin:0; color:white;">🎤 Live Mic Detector</h3>', unsafe_allow_html=True)
    audio_file = st.audio_input("Record")
    db_level = 0
    if audio_file:
        audio_file.seek(0)
        data_mic = np.frombuffer(audio_file.read(), dtype=np.int16)
        if len(data_mic) > 0:
            rms = np.sqrt(np.mean(data_mic.astype(float)**2))
            db_level = 20 * np.log10(rms) if rms > 0 else 0
    st.write(f"Volume: {db_level:.1f} dB")
    st.progress(min(max(db_level/100, 0.0), 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

# 5. Logika Analisis (FFT/Spectrum)
final_freq = freq_input
spectrum_data = None
if audio_file:
    audio_file.seek(0)
    data_raw = np.frombuffer(audio_file.read(), dtype=np.int16)
    if len(data_raw) > 0:
        sample_rate = 44100
        N = len(data_raw)
        yf = fft(data_raw)
        xf = fftfreq(N, 1 / sample_rate)
        spectrum_data = np.abs(yf[:N//2])
        idx = np.argmax(spectrum_data)
        final_freq = abs(xf[idx])

st.markdown("<hr>", unsafe_allow_html=True)

# 6. Status Kategori
st.markdown(f"### Monitoring: <span style='color:#FF0000'>{final_freq:,.2f} Hz</span>", unsafe_allow_html=True)
cols = st.columns(4)
categories = [
    {"Nama": "Infrasound", "Min": 0, "Max": 20, "Num": "01"},
    {"Nama": "Hearing", "Min": 20, "Max": 20000, "Num": "02"},
    {"Nama": "Ultrasound", "Min": 20000, "Max": 10**9, "Num": "03"},
    {"Nama": "Hypersound", "Min": 10**9, "Max": 10**13, "Num": "04"},
]
for i, cat in enumerate(categories):
    is_active = cat["Min"] <= final_freq < cat["Max"]
    with cols[i]:
        class_name = "active-red" if is_active else "inactive-dark"
        st.markdown(f'<div class="{class_name}"><h4 style="margin:0;">{cat["Nama"]}</h4><p>{"● ACTIVE" if is_active else "○ STANDBY"}</p></div>', unsafe_allow_html=True)

# 7. Visualisasi (Waveform, Spectrum, & Audio)
st.write("")
st.markdown('<div class="pod-card">', unsafe_allow_html=True)
v_col1, v_col2 = st.columns(2)

# Persiapan data audio & grafik
sample_rate = 44100
duration = 1.0
t_audio = np.linspace(0, duration, sample_rate, endpoint=False)

if wave_type == "Sine": ya = np.sin(2 * np.pi * final_freq * t_audio)
elif wave_type == "Square": ya = np.sign(np.sin(2 * np.pi * final_freq * t_audio))
else: ya = 2 * (final_freq * t_audio - np.floor(0.5 + final_freq * t_audio))

with v_col1:
    st.subheader("📈 Waveform Monitor")
    points = 1000
    t_plot = np.linspace(0, 5/final_freq if final_freq > 0 else 0.1, points)
    if wave_type == "Sine": y_plot = np.sin(2 * np.pi * final_freq * t_plot)
    elif wave_type == "Square": y_plot = np.sign(np.sin(2 * np.pi * final_freq * t_plot))
    else: y_plot = 2 * (final_freq * t_plot - np.floor(0.5 + final_freq * t_plot))
    st.line_chart(pd.DataFrame({"Amp": y_plot}), color="#FF0000")
    
    # Export Data
    csv = pd.DataFrame({"Amp": y_plot}).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export CSV", data=csv, file_name="ankayu_data.csv", mime="text/csv")

with v_col2:
    # --- GRAFIK FFT / SPECTRUM ANALYZER ---
    st.subheader("📊 Spectrum Analyzer (FFT)")
    if spectrum_data is not None:
        st.area_chart(spectrum_data[:2000], color="#FF0000")
    else:
        # Simulasi grafik jika pakai Master Control
        sim_spectrum = np.zeros(200)
        peak_pos = int(np.clip((final_freq / 20000) * 200, 0, 199)) if final_freq < 20000 else 199
        sim_spectrum[peak_pos] = 1.0
        st.area_chart(sim_spectrum, color="#FF0000")
    
    # --- AUDIO OUTPUT ---
    st.write("---")
    st.subheader("🔊 Audio Output")
    if final_freq > 0:
        audio_data = (ya * 32767).astype(np.int16)
        byte_io = io.BytesIO()
        write(byte_io, sample_rate, audio_data)
        st.audio(byte_io.read(), format="audio/wav")
    else:
        st.write("Input frekuensi.")

st.markdown('</div>', unsafe_allow_html=True)