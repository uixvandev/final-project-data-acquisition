import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.title("Analisis Data Inflasi Global")

# Data Upload
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Data Preprocessing
st.header("2. Data Preprocessing")
st.write("Handling nilai yang hilang")
if st.checkbox("Isi nilai yang hilang"):
    fill_method = st.radio("Pilih metode:", ["Mean", "Median", "Hapus baris"])
    if fill_method == "Mean":
        data = data.fillna(data.mean(numeric_only=True))
    elif fill_method == "Median":
        data = data.fillna(data.median(numeric_only=True))
    elif fill_method == "Hapus baris":
        data = data.dropna()
    st.write("Data setelah preprocessing:")
    st.dataframe(data)

# Select Year for Analysis
st.header("3. Analisis Data")
year_columns = [col for col in data.columns if col.isdigit()]
selected_years = st.multiselect("Pilih tahun untuk analisis:", year_columns)
if not selected_years:
    st.warning("Pilih setidaknya 2 tahun!")
    st.stop()

# Filter data by selected years
analysis_data = data[["country_name"] + selected_years].dropna()
st.write("Data yang difilter untuk tahun yang dipilih")
st.dataframe(analysis_data)

# Clustering Analysis
st.subheader("Analisis Clustering")
n_clusters = st.slider("Jumlah Cluster:", 2, 10, 3)

if st.button("Jalankan Clustering"):
    if len(selected_years) < 2:
        st.error("Pilih setidaknya 2 tahun untuk analisis pengelompokan!")
    else:
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering_model.fit_predict(analysis_data[selected_years])
        analysis_data["Cluster"] = cluster_labels

        # Display clustered data
        st.write("Clustered Data:")
        st.dataframe(analysis_data)

        # Visualize Clusters with Plotly
        st.write("Cluster Visualization")
        fig = px.scatter(
            analysis_data,
            x=selected_years[0],
            y=selected_years[1],
            color="Cluster",
            hover_data={"country_name": True, "Cluster": True},
            title="Clusters of Countries",
            template="plotly"
        )
        st.plotly_chart(fig)

# Elbow Method
if st.checkbox("Tampilkan Elbow Method"):
    st.write("Menghitung jumlah cluster optimal dengan Elbow Method...")
    
    # Gunakan nilai n_clusters dari slider sebagai batas atas perhitungan Elbow Method
    K = range(1, n_clusters + 1)  
    distortions = []  # Untuk menyimpan inertia

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(analysis_data[selected_years])
        distortions.append(kmeans.inertia_)

    # Plotting the Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel("Jumlah Cluster (k)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method untuk Menentukan Cluster Optimal", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)

# Visualization
st.header("4. Data Visualization")
if st.checkbox("Tampilkan Heatmap"):
    st.write("Heatmap of Inflation Rates")
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=0.8)  # Mengatur skala font global
    
    heatmap_data = analysis_data[selected_years].set_index(analysis_data["country_name"])
    
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        annot=True,             # Jika terlalu ramai, Anda dapat ubah ke False
        fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"shrink": 0.5}
    )
    plt.title("Heatmap of Inflation Rates by Country", fontsize=16, pad=20)
    plt.xlabel("Years", fontsize=12, labelpad=10)
    plt.ylabel("Country Name", fontsize=12, labelpad=10)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    st.pyplot(plt)

if st.checkbox("Tampilkan Line Chart"):
    st.write("Line Chart Tingkat Inflasi dari Waktu ke Waktu")
    # Pilih negara yang ingin ditampilkan
    countries = analysis_data["country_name"].unique().tolist()
    selected_countries = st.multiselect("Pilih negara:", countries, default=countries[:5])  # Sebagai contoh default 5 negara pertama

    # Filter data berdasarkan negara terpilih
    filtered_data = analysis_data[analysis_data["country_name"].isin(selected_countries)]

    plt.figure(figsize=(12, 6))
    for country in filtered_data["country_name"].unique():
        country_data = filtered_data[filtered_data["country_name"] == country][["country_name"] + selected_years].dropna()
        plt.plot(
            selected_years,
            country_data[selected_years].values.flatten(),
            label=country,
            linewidth=1.5,
            alpha=0.8
        )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10, title="Countries")
    plt.title("Tren Tingkat Inflasi Berdasarkan Negara (Berdasarkan Tahun Terpilih)", fontsize=16, pad=15)
    plt.xlabel("Tahun", fontsize=12)
    plt.ylabel("Tingkat Inflasi (%)", fontsize=12)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout() 
    st.pyplot(plt)
