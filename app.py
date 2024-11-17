import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.title("Global Inflation Data Analysis")

# Data Upload
st.header("1. Upload or View Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Data Preprocessing
st.header("2. Data Preprocessing")
st.write("Handling Missing Values")
if st.checkbox("Fill Missing Values"):
    fill_method = st.radio("Choose fill method:", ["Mean", "Median", "Drop rows"])
    if fill_method == "Mean":
        data = data.fillna(data.mean(numeric_only=True))
    elif fill_method == "Median":
        data = data.fillna(data.median(numeric_only=True))
    elif fill_method == "Drop rows":
        data = data.dropna()
    st.write("Data after preprocessing:")
    st.dataframe(data)

# Select Year for Analysis
st.header("3. Analyze Data")
year_columns = [col for col in data.columns if col.isdigit()]
selected_years = st.multiselect("Select year(s) for analysis:", year_columns)
if not selected_years:
    st.warning("Please select at least one year.")
    st.stop()

# Filter data by selected years
analysis_data = data[["country_name"] + selected_years].dropna()
st.write("Filtered data for selected years:")
st.dataframe(analysis_data)

# Clustering Analysis
st.subheader("Clustering Analysis")
n_clusters = st.slider("Number of clusters:", 2, 10, 3)

if st.button("Run Clustering"):
    if len(selected_years) < 2:
        st.error("Please select at least two years for clustering analysis.")
    else:
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering_model.fit_predict(analysis_data[selected_years])
        analysis_data["Cluster"] = cluster_labels

        st.write("Clustered Data:")
        st.dataframe(analysis_data)

        # Visualize Clusters with Plotly
        st.write("Cluster Visualization (Interactive)")
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
if st.checkbox("Show Heatmap"):
    st.write("Heatmap of Inflation Rates")
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        analysis_data[selected_years].set_index(analysis_data["country_name"]),
        cmap="coolwarm",
        annot=True,
        fmt=".2f"
    )
    st.pyplot(plt)

if st.checkbox("Show Line Chart"):
    st.write("Line Chart of Inflation Rates Over Time")
    for country in analysis_data["country_name"].unique():
        country_data = data[data["country_name"] == country]
        plt.plot(year_columns, country_data[year_columns].values.flatten(), label=country)
    plt.legend()
    plt.title("Inflation Trends")
    plt.xlabel("Year")
    plt.ylabel("Inflation Rate")
    st.pyplot(plt)
