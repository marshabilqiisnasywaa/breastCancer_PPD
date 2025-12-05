import requests
import streamlit as st
import os

API_URL = os.environ.get("BREAST_CANCER_API", "http://127.0.0.1:8000")

# Konfigurasi halaman
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .positive-result {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
    }
    .negative-result {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    .feature-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background: #f0f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0ea5e9;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown('<h1 class="main-header">🩺 Breast Cancer Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem prediksi kanker payudara berbasis machine learning</p>', unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def get_feature_columns(api_base: str):
    try:
        r = requests.get(f"{api_base}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("features", [])
        raise RuntimeError(f"Gagal mengambil fitur dari API: {r.text}")
    except Exception as e:
        st.error(f"Koneksi API gagal: {str(e)}")
        return []

# Sidebar untuk input
with st.sidebar:
    st.markdown("## 📊 Input Fitur")
    st.markdown("Masukkan nilai untuk setiap parameter di bawah:")
    
    # Mendapatkan daftar fitur
    with st.spinner("Memuat parameter..."):
        features = get_feature_columns(API_URL)
    
    if not features:
        st.warning("⚠️ Tidak dapat memuat daftar parameter. Pastikan API berjalan.")
        st.stop()
    
    # Input untuk setiap fitur di sidebar
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    inputs = {}
    for feat in features:
        inputs[feat] = st.number_input(
            label=f"**{feat}**",
            value=0.0,
            step=0.1,
            format="%.4f",
            key=feat,
            help=f"Masukkan nilai untuk {feat}"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tombol prediksi
    predict_button = st.button(
        "🚀 **Jalankan Prediksi**",
        type="primary",
        use_container_width=True
    )
    
    # Informasi tambahan di sidebar
    with st.expander("ℹ️ Tentang Nilai Input"):
        st.write("""
        **Format Input:**
        - Semua nilai dalam format bilangan desimal
        - Gunakan titik (.) sebagai pemisah desimal
        - Contoh: 0.1234, 1.5678, 12.3456
        
        **Sumber Data:**
        - Data dikumpulkan dari hasil pemeriksaan medis
        - Nilai telah dinormalisasi untuk analisis
        """)

# Main section - hanya informasi dan hasil prediksi
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📝 Informasi Aplikasi")
    
    # st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Aplikasi Prediksi Kanker Payudara** membantu dalam:
    
    🔍 **Analisis Awal**: Memberikan prediksi probabilistik berdasarkan data input
    
    ⚕️ **Screening Dini**: Mendeteksi kemungkinan adanya indikasi kanker
    
    📊 **Dukungan Keputusan**: Sebagai alat bantu untuk tenaga medis
    
    ---
    
    **Cara Penggunaan:**
    1. Masukkan nilai parameter di sidebar kiri
    2. Klik tombol **"Jalankan Prediksi"**
    3. Lihat hasil dan interpretasi di panel kanan
    
    **Catatan Penting:**
    - Hasil prediksi bersifat probabilistik
    - Tidak menggantikan diagnosis dari dokter
    - Konsultasikan dengan tenaga medis untuk diagnosis yang akurat
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 🔍 Hasil Prediksi")
    
    if predict_button:
        with st.spinner("🔬 Sedang menganalisis data..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=inputs, timeout=10)
                
                if r.status_code == 200:
                    data = r.json()
                    proba = data.get("probability", 0.0)
                    label = data.get("label", 0)
                    
                    # Konversi probabilitas ke persentase
                    proba_percent = proba * 100
                    
                    # Tampilkan hasil
                    result_class = "positive-result" if label == 1 else "negative-result"
                    
                    st.markdown(f'<div class="prediction-box {result_class}">', unsafe_allow_html=True)
                    
                    # Progress bar untuk probabilitas
                    st.markdown(f"**Tingkat Probabilitas:** {proba_percent:.2f}%")
                    st.progress(proba)
                    
                    # Tampilkan hasil dengan emoji dan warna
                    if label == 1:
                        st.markdown("""
                        ## ⚠️ **HASIL: POSITIF**
                        
                        **Interpretasi:**
                        - Terdapat indikasi kemungkinan kanker payudara
                        - Probabilitas: **Tinggi**
                        
                        **Rekomendasi:**
                        1. Konsultasi segera dengan dokter spesialis
                        2. Lakukan pemeriksaan lanjutan (mamografi/USG)
                        3. Diskusikan hasil dengan tenaga medis profesional
                        """)
                    else:
                        st.markdown("""
                        ## ✅ **HASIL: NEGATIF**
                        
                        **Interpretasi:**
                        - Kemungkinan tidak terdapat kanker payudara
                        - Probabilitas: **Rendah**
                        
                        **Rekomendasi:**
                        1. Tetap lakukan pemeriksaan rutin
                        2. Pertahankan pola hidup sehat
                        3. Waspada terhadap perubahan pada payudara
                        """)
                    
            except requests.exceptions.Timeout:
                st.error("⏰ Waktu pemrosesan habis. Silakan coba lagi.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Tidak dapat terhubung ke server prediksi.")
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {str(e)}")
    else:
        # Placeholder sebelum prediksi
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: #f8fafc; border-radius: 10px;'>
            <h3 style='color: #64748b;'>⏳ Menunggu Prediksi</h3>
            <p style='color: #94a3b8;'>
                Masukkan nilai parameter di sidebar<br>
                dan klik tombol <strong>"Jalankan Prediksi"</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        

# Footer
st.markdown("---")
st.caption("""
**Disclaimer:** Aplikasi ini merupakan alat bantu prediksi awal dan tidak menggantikan diagnosis dari tenaga medis profesional. 
Selalu konsultasikan dengan dokter untuk diagnosis yang akurat.
""")