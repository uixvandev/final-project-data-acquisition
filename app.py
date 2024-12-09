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

        # Save clustered data as downloadable CSV
        st.write("Download hasil clustering:")
        csv = analysis_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv',
        )

        # Visualize Clusters with Plotly
        st.write("Cluster Visualization")
        fig = px.scatter(
            analysis_data,
            x=selected_years[0],
            y=selected_years[1],
            color="Cluster",
            hover_data={"country_name": True, "Cluster": True},
            title="Clusters of Countries",
            labels={selected_years[0]: f"Inflation in {selected_years[0]}", selected_years[1]: f"Inflation in {selected_years[1]}"},
            template="plotly"
        )
        st.plotly_chart(fig)

# Elbow Method
if st.checkbox("Tampilkan Elbow Method"):
    st.write("Menghitung jumlah cluster optimal dengan Elbow Method...")
    
    # Define range for number of clusters
    K = range(1, 11)  # Test clusters from 1 to 10
    distortions = []  # To store the sum of squared distances (inertia)
    
    # Compute KMeans for each k
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
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines
    st.pyplot(plt)

# Visualization
st.header("4. Data Visualization")
if st.checkbox("Tampilkan Heatmap"):
    st.write("Heatmap of Inflation Rates")
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        analysis_data[selected_years].set_index(analysis_data["country_name"]),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},  # Set font size for annotations
        linewidths=0.5,        # Add grid lines between cells
        linecolor='gray',
    )
    plt.title("Heatmap of Inflation Rates by Country", fontsize=14, pad=15)
    plt.xlabel("Years", fontsize=12)
    plt.ylabel("Country Name", fontsize=12)
    plt.xticks(fontsize=10, rotation=45)  # Rotate x-axis labels for readability
    plt.yticks(fontsize=10, rotation=0)
    st.pyplot(plt)

if st.checkbox("Tampilkan Line Chart"):
    st.write("Line Chart Tingkat Inflasi dari Waktu ke Waktu")
    for country in analysis_data["country_name"].unique():
        country_data = data[data["country_name"] == country]
        plt.plot(year_columns, country_data[year_columns].values.flatten(), label=country,
        linewidth=1.5,  # Adjust line width
            alpha=0.8 
        )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10, title="Countries")
    plt.title("Tren Tingkat Inflasi Berdasarkan Negara", fontsize=16, pad=15)
    plt.xlabel("Tahun", fontsize=12)
    plt.ylabel("Tingkat Inflasi (%)", fontsize=12)
    plt.xticks(fontsize=10, rotation=45)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)  # Add grid lines
    plt.tight_layout() 
    st.pyplot(plt)
