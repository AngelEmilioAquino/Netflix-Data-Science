import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="Estadística Descriptiva Netflix", layout="wide")

st.title("Unidad 2: Estadística Descriptiva en la Ciencia de Datos")
st.subheader("Grupo #2 - Dataset Netflix")

# =========================
# Cargar Dataset
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/netflix_titles.csv")
    df_movies = df[df['type'] == 'Movie'].copy()
    df_movies['duration_int'] = df_movies['duration'].str.replace(" min", "", regex=False)
    df_movies['duration_int'] = pd.to_numeric(df_movies['duration_int'], errors="coerce")
    df_movies = df_movies.dropna(subset=['duration_int'])
    return df_movies

movies = load_data()

# =========================
# 2.4.1 La Media
# =========================
st.header("2.4.1 La Media")

# Histograma con media
fig, ax = plt.subplots()
sns.histplot(movies['duration_int'], bins=30, kde=True, ax=ax, color="skyblue")
ax.axvline(movies['duration_int'].mean(), color="red", linestyle="--", label="Media")
ax.legend()
st.pyplot(fig)

st.markdown("👉 La línea roja marca la media (promedio) de las duraciones.")

# Media muestral vs poblacional
media_poblacional = movies['duration_int'].mean()
media_muestral = movies['duration_int'].sample(100, random_state=42).mean()

st.subheader("2.4.1.1 Media muestral vs. poblacional")
st.info(f"Media poblacional = {media_poblacional:.2f} | Media muestral (n=100) = {media_muestral:.2f}")

# Propiedades de la media (visual con outliers)
st.subheader("2.4.1.2 Propiedades de la media")
outliers = movies['duration_int'].copy()
outliers.iloc[0] = 1000  # forzar un valor extremo
fig, ax = plt.subplots()
sns.boxplot(x=outliers, color="orange", ax=ax)
st.pyplot(fig)
st.markdown("👉 La media es **sensible a valores extremos**, como se ve en el boxplot.")

# Media ponderada (por rating)
st.subheader("2.4.1.3 Media ponderada")
ratings = movies.groupby("rating")['duration_int'].mean()
weights = movies['rating'].value_counts(normalize=True)
media_ponderada = (ratings * weights).sum()
fig = px.bar(ratings, title="Media de duración por Rating")
st.plotly_chart(fig)
st.success(f"Media ponderada = {media_ponderada:.2f} min")

# Media truncada
st.subheader("2.4.1.4 Media truncada")
media_truncada = stats.trim_mean(movies['duration_int'], 0.1)
fig, ax = plt.subplots()
sns.histplot(movies['duration_int'], bins=30, kde=True, color="purple", ax=ax)
ax.axvline(media_poblacional, color="red", linestyle="--", label="Media")
ax.axvline(media_truncada, color="green", linestyle="--", label="Media Truncada")
ax.legend()
st.pyplot(fig)
st.success(f"Media truncada = {media_truncada:.2f} min")

# Error estándar
st.subheader("2.4.1.5 Error estándar de la media")
boot_means = [movies['duration_int'].sample(100, replace=True).mean() for _ in range(500)]
fig, ax = plt.subplots()
sns.histplot(boot_means, kde=True, ax=ax, color="teal")
st.pyplot(fig)
st.info(f"Error estándar ≈ {np.std(boot_means):.2f}")
st.markdown("👉 El error estándar mide la variabilidad de las medias muestrales.")

# =========================
# 2.4.2 La Varianza
# =========================
st.header("2.4.2 La Varianza")

# Histograma con dispersión
fig, ax = plt.subplots()
sns.histplot(movies['duration_int'], bins=30, kde=True, ax=ax, color="blue")
st.pyplot(fig)

var_poblacional = movies['duration_int'].var(ddof=0)
var_muestral = movies['duration_int'].var(ddof=1)
desv = movies['duration_int'].std()

st.subheader("2.4.2.1 Varianza muestral vs. poblacional")
st.write(f"Varianza poblacional = {var_poblacional:.2f}")
st.write(f"Varianza muestral = {var_muestral:.2f}")

st.subheader("2.4.2.2 Desviación estándar e interpretación")
fig, ax = plt.subplots()
sns.boxplot(x=movies['duration_int'], ax=ax, color="lightgreen")
st.pyplot(fig)
st.info(f"Desviación estándar = {desv:.2f} min")

st.subheader("2.4.2.3 Coeficiente de variación")
cv = desv / media_poblacional
st.success(f"CV = {cv:.2%}")

st.subheader("2.4.2.4 Error estándar de la varianza")
n = len(movies)
error_var = np.sqrt(2 * (var_muestral**2) / (n-1))
st.info(f"Error estándar ≈ {error_var:.2f}")

# =========================
# 2.4.3 Medidas de tendencia central
# =========================
st.header("2.4.3 Otras medidas de tendencia central")

mediana = movies['duration_int'].median()
moda = movies['duration_int'].mode()[0]

fig, ax = plt.subplots()
sns.histplot(movies['duration_int'], bins=30, kde=True, ax=ax, color="gray")
ax.axvline(media_poblacional, color="red", linestyle="--", label="Media")
ax.axvline(mediana, color="green", linestyle="--", label="Mediana")
ax.axvline(moda, color="blue", linestyle="--", label="Moda")
ax.legend()
st.pyplot(fig)

st.info(f"Mediana = {mediana} min | Moda = {moda} min")

# =========================
# 2.4.4 Medidas de dispersión
# =========================
st.header("2.4.4 Otras medidas de dispersión")

rango = movies['duration_int'].max() - movies['duration_int'].min()
iqr = stats.iqr(movies['duration_int'])

fig, ax = plt.subplots()
sns.boxplot(x=movies['duration_int'], ax=ax, color="orange")
st.pyplot(fig)

st.info(f"Rango = {rango} min | IQR = {iqr:.2f} min")

# =========================
# 2.4.5 Covarianza y Correlación
# =========================
st.header("2.4.5 Covarianza y Correlación")

fig = px.scatter(movies, x="release_year", y="duration_int",
                 opacity=0.5, trendline="ols",
                 labels={"release_year":"Año", "duration_int":"Duración (min)"})
st.plotly_chart(fig)

cov = np.cov(movies['release_year'], movies['duration_int'])[0,1]
pearson = stats.pearsonr(movies['release_year'], movies['duration_int'])

st.info(f"Covarianza = {cov:.2f} | Pearson = {pearson[0]:.2f}")

fig, ax = plt.subplots()
sns.heatmap(movies[['release_year','duration_int']].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("👉 También existen correlaciones no lineales (Spearman, Kendall).")
st.warning("⚠️ Correlación no implica causalidad.")

