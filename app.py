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


    # Fitur 3: Analisis Perbandingan Negara
    st.subheader("Analisis Perbandingan Negara")

    # Pilih 2-4 negara untuk dibandingkan
    countries = data['country_name'].unique()
    selected_countries = st.multiselect("Pilih 2 hingga 4 negara untuk dibandingkan", countries, default=countries[:2])
    
    if len(selected_countries) < 2:
        st.error("Pilih minimal 2 negara untuk perbandingan.")
    elif len(selected_countries) > 4:
        st.error("Pilih maksimal 4 negara.")
    else:
        # Memfilter data untuk negara yang dipilih
        filtered_data = data[data['country_name'].isin(selected_countries)]
        
        # Memutar data agar lebih mudah diproses untuk plotly
        melted_data = filtered_data.melt(id_vars=['country_name'], var_name='Tahun', value_name='Inflation Rate')
        
        # Chart 1: Perbandingan Inflasi dengan Line Chart
        st.subheader("Perbandingan Tren Inflasi antara Negara")
        fig1 = px.line(
            melted_data, 
            x='Tahun', 
            y='Inflation Rate', 
            color='country_name', 
            title=f"Perbandingan Inflasi antara {', '.join(selected_countries)}",
            markers=True,
            template="plotly_white"
        )
        fig1.update_layout(
            title_font=dict(size=24),
            xaxis_title="Tahun",
            yaxis_title="Tingkat Inflasi (%)",
            font=dict(size=14),
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',  # Background putih yang bersih
            hovermode="x unified",  # Tampilan hover yang jelas
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Tampilkan grafik Line Chart
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Rata-Rata Inflasi dengan Bar Chart
        avg_inflation = filtered_data.set_index('country_name').mean(axis=1).reset_index()
        avg_inflation.columns = ['country_name', 'Average Inflation']

        # Ambil warna dari Line Chart (plotly express biasanya mengikuti urutan warna secara default)
        color_map = {trace['name']: trace['line']['color'] for trace in fig1['data']}
        
        # Buat Bar Chart dengan warna yang sama seperti pada Line Chart
        fig2 = px.bar(
            avg_inflation,
            x='country_name',
            y='Average Inflation',
            title="Rata-Rata Inflasi Negara yang Dipilih",
            labels={'country_name': 'Negara', 'Average Inflation': 'Rata-Rata Inflasi (%)'},
            template="plotly_white",
            color='country_name',  # Gunakan kategori negara untuk memberi warna
            color_discrete_map=color_map  # Terapkan skema warna yang sama
        )
        
        fig2.update_layout(
            title_font=dict(size=24),
            xaxis_title="Negara",
            yaxis_title="Rata-Rata Inflasi (%)",
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Tampilkan grafik Bar Chart
        st.plotly_chart(fig2, use_container_width=True)
