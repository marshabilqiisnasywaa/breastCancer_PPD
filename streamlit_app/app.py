import requests
import streamlit as st
import os

# --- KONFIGURASI HALAMAN (Wajib Paling Atas) ---
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KONFIGURASI AKUN DOKTER ---
# Kamu bisa ganti username/password ini sesuai keinginan
USERNAME_DOKTER = "dokter"
PASSWORD_DOKTER = "medis123"

API_URL = os.environ.get("BREAST_CANCER_API", "http://localhost:8000")

# --- FUNGSI LOGIN KHUSUS DOKTER ---
def check_login():
    """Fungsi untuk membatasi akses hanya untuk dokter"""
    # Inisialisasi status login jika belum ada
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False

    # Jika belum login, tampilkan form login dokter
    if not st.session_state.is_logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            # Tampilan Login Dokter
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #f0f9ff; border-radius: 10px; border: 1px solid #bae6fd;'>
                <h1 style='color: #0369a1;'> Portal Dokter</h1>
                <p style='color: #475569;'>Silakan login untuk mengakses alat prediksi klinis.</p>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            # Kotak Input
            username_input = st.text_input("ID Dokter / Username")
            password_input = st.text_input("Kata Sandi", type="password")
            
            login_btn = st.button("Masuk ke Sistem", use_container_width=True, type="primary")

            if login_btn:
                if username_input == USERNAME_DOKTER and password_input == PASSWORD_DOKTER:
                    st.session_state.is_logged_in = True
                    st.success("Login berhasil! Mengalihkan...")
                    st.rerun() # Refresh halaman untuk masuk
                else:
                    st.error(" Akses Ditolak: ID atau Kata Sandi salah.")
        return False # Artinya belum boleh masuk
    
    return True # Artinya sudah login

# --- MAIN PROGRAM ---
# Cek login dulu. Kalau belum login, stop eksekusi di sini.
if not check_login():
    st.stop()

# --- BAGIAN BAWAH INI HANYA MUNCUL KALAU SUDAH LOGIN ---

# CSS kustom untuk styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    .prediction-box { padding: 2rem; border-radius: 10px; margin-top: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .positive-result { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 2px solid #ef4444; }
    .negative-result { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border: 2px solid #22c55e; }
    .feature-section { background: #f8fafc; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown('<h1 class="main-header"> Breast Cancer Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Pendukung Keputusan Klinis (CDSS)</p>', unsafe_allow_html=True)

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

# Sidebar
with st.sidebar:
    # Tombol Logout Dokter
    st.write(f"Login sebagai: **{USERNAME_DOKTER}**")
    if st.button(" Logout Dokter", use_container_width=True):
        st.session_state.is_logged_in = False
        st.rerun()
        
    st.markdown("---")
    st.markdown("##  Input Parameter Klinis")
    st.markdown("Masukkan data pasien:")
    
    # Mendapatkan daftar fitur
    with st.spinner("Memuat parameter..."):
        features = get_feature_columns(API_URL)
    
    if not features:
        st.warning(" Tidak dapat memuat daftar parameter. Pastikan API berjalan.")
        st.stop()
    
    # Input fitur
    inputs = {}
    for feat in features:
        inputs[feat] = st.number_input(
            label=f"**{feat}**",
            value=0.0,
            step=0.1,
            format="%.4f",
            key=feat
        )
    
    st.markdown("---")
    
    # Tombol prediksi
    predict_button = st.button(
        " **Analisis Pasien**",
        type="primary",
        use_container_width=True
    )

# Layout Hasil
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## Catatan Medis")
    st.info("""
    **Panduan Penggunaan:**
    1. Pastikan data input sesuai dengan hasil lab pasien.
    2. Hasil prediksi adalah probabilitas statistik berdasarkan model Machine Learning.
    3. **Keputusan akhir tetap di tangan Dokter.**
    """)

with col2:
    st.markdown("## Hasil Analisis")
    
    if predict_button:
        with st.spinner(" Sedang memproses data pasien..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=inputs, timeout=10)
                
                if r.status_code == 200:
                    data = r.json()
                    proba = data.get("probability", 0.0)
                    label = data.get("label", 0)
                    proba_percent = proba * 100
                    result_class = "positive-result" if label == 1 else "negative-result"
                    
                    st.markdown(f'<div class="prediction-box {result_class}">', unsafe_allow_html=True)
                    
                    st.markdown(f"**Probabilitas Keganasan:** {proba_percent:.2f}%")
                    st.progress(proba)
                    
                    if label == 1:
                        st.markdown("""
                        ###  HASIL: TERINDIKASI GANAS (Malignant)
                        **Saran Tindakan:**
                        - Jadwalkan pemeriksaan Biopsi.
                        - Lakukan USG lanjutan.
                        """)
                    else:
                        st.markdown("""
                        ###  HASIL: JINAK (Benign)
                        **Saran Tindakan:**
                        - Observasi rutin.
                        - Jadwalkan kontrol ulang 6 bulan lagi.
                        """)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error sistem: {str(e)}")