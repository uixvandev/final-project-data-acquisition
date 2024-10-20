import streamlit as st
import pandas as pd
import plotly.express as px

# Fitur 1: Input Dataset ke Sistem
st.title('Aplikasi Tren Inflasi Global')

uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
if uploaded_file is not None:
    # Membaca data
    data = pd.read_csv(uploaded_file)
    
    # Fitur 2: Preprocess Data
    st.subheader("Preprocessing Data")
    
    # Menghapus kolom yang tidak diperlukan
    data = data.drop(columns=["indicator_name"])
    
    # Pisahkan kolom non-numerik dan numerik
    non_numeric_columns = ['country_name']
    numeric_data = data.drop(columns=non_numeric_columns).apply(pd.to_numeric, errors='coerce')
    
    # Interpolasi pada data numerik
    numeric_data = numeric_data.interpolate(method='linear', axis=1)
    
    # Gabungkan kembali dengan kolom non-numerik
    data[non_numeric_columns] = data[non_numeric_columns]
    data.update(numeric_data)
    
    # Tampilkan 10 baris pertama secara default
    st.subheader("Data yang Diunggah (10 Baris Pertama)")
    st.write(data.head(10))
    
    # Tombol untuk menampilkan semua baris
    if st.button("Tampilkan Semua Baris"):
        st.write(data)
    
    # Fitur 3: Analisis Data
    st.subheader("Analisis Data")
    
    # Pilih negara untuk analisis
    countries = data['country_name'].unique()
    selected_country = st.selectbox("Pilih negara", countries)
    
    # Analisis data negara yang dipilih
    country_data = data[data['country_name'] == selected_country].set_index('country_name').T
    country_data.columns = ['Inflation Rate']
    
    # Statistik sederhana
    st.write(f"Rata-rata inflasi: {country_data.mean()[0]:.2f}")
    st.write(f"Inflasi tertinggi: {country_data.max()[0]:.2f}")
    st.write(f"Inflasi terendah: {country_data.min()[0]:.2f}")
    
    # Fitur 4: Visualisasi Data dengan UI Indah Menggunakan Plotly
    st.subheader("Visualisasi Tren Inflasi")

    # Menggunakan Plotly untuk membuat visualisasi interaktif
    country_data.reset_index(inplace=True)  # Ubah indeks kembali menjadi kolom untuk plotly
    country_data.rename(columns={'index': 'Tahun'}, inplace=True)
    
    # Membuat grafik interaktif menggunakan plotly
    fig = px.line(
        country_data, 
        x='Tahun', 
        y='Inflation Rate', 
        title=f"Tren Inflasi di {selected_country}",
        markers=True,
        template="plotly_white"
    )
    
    # Menambahkan elemen estetis
    fig.update_layout(
        title_font=dict(size=24),
        xaxis_title="Tahun",
        yaxis_title="Tingkat Inflasi (%)",
        font=dict(size=14),
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',  # Background putih yang bersih
        hovermode="x unified",  # Tampilan hover yang jelas
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Menampilkan chart interaktif
    st.plotly_chart(fig, use_container_width=True)
