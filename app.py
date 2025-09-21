import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Cargar dataset
df = pd.read_csv("data/netflix_titles.csv")

st.title("🎬 Análisis Estadístico de Netflix")
st.write("Exploración descriptiva de películas y series en Netflix.")

# Vista previa
st.subheader("Vista rápida del dataset")
st.dataframe(df.head())

# 1. Distribución de duración de películas
st.subheader("Distribución de duración de películas")
movies = df[df['type'] == 'Movie'].copy()
movies['duration_int'] = movies['duration'].str.replace(" min","").astype(float)

fig, ax = plt.subplots()
sns.histplot(movies['duration_int'], bins=30, kde=True, ax=ax, color="red")
st.pyplot(fig)

# 2. Boxplot de duración
st.subheader("Boxplot de duración de películas")
fig, ax = plt.subplots()
sns.boxplot(x=movies['duration_int'], color="orange", ax=ax)
st.pyplot(fig)

# 3. Conteo de shows por país


# 4. Conteo por rating
st.subheader("Distribución de clasificaciones (rating)")
fig = px.histogram(df, x='rating', color='rating')
st.plotly_chart(fig)

# 5. Heatmap de correlación
st.subheader("Correlación entre variables numéricas")
corr = movies[['duration_int']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
