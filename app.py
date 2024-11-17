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
            labels={selected_years[0]: f"Inflation in {selected_years[0]}", selected_years[1]: f"Inflation in {selected_years[1]}"},
            template="plotly"
        )
        st.plotly_chart(fig)

# Visualization
st.header("4. Data Visualization")
if st.checkbox("Tampilkan Heatmap"):
    st.write("Heatmap of Inflation Rates")
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        analysis_data[selected_years].set_index(analysis_data["country_name"]),
        cmap="coolwarm",
        annot=True,
        fmt=".2f"
    )
    st.pyplot(plt)

if st.checkbox("Tampilkan Line Chart"):
    st.write("Line Chart Tingkat Inflasi dari Waktu ke Waktu")
    for country in analysis_data["country_name"].unique():
        country_data = data[data["country_name"] == country]
        plt.plot(year_columns, country_data[year_columns].values.flatten(), label=country)
    plt.legend()
    plt.title("Inflation Trends")
    plt.xlabel("Year")
    plt.ylabel("Inflation Rate")
    st.pyplot(plt)
