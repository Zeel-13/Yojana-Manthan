import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
nltk.download('punkt_tab')
import nltk
import os
from nltk.corpus import stopwords
from nltk.data import find

from fpdf import FPDF
import qrcode
import tempfile

# --- Safe downloads: check if already downloaded ---
def download_nltk_resource(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

# Download needed resources safely
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')
download_nltk_resource('corpora/wordnet')

# Now use them
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()


# --- TEXT PREPROCESSING ---
def enhanced_preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    bigrams = list(nltk.bigrams(tokens))
    bigram_phrases = [f"{b[0]}_{b[1]}" for b in bigrams]
    final_tokens = tokens + bigram_phrases
    return " ".join(final_tokens)




# --- LOAD & PROCESS DATA ---
@st.cache_resource
def load_and_process_data():
    df = pd.read_csv("Gov schemes.csv")

    def combine_features(row):
        return f"{row['Scheme Name']} {row['Details']} {row['State']} {row['Category']}"

    df["features"] = df.apply(combine_features, axis=1)
    df["features"] = df["features"].apply(enhanced_preprocess_text)

    vectorizer = TfidfVectorizer(min_df=2, max_df=0.85, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["features"])

    n_components = min(100, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components)
    latent_matrix = svd.fit_transform(tfidf_matrix)
    latent_sim = cosine_similarity(latent_matrix, latent_matrix)

    return df, vectorizer, svd, latent_matrix, latent_sim

# --- RECOMMENDER FUNCTION ---
def recommend_schemes_improved(df, vectorizer, svd, latent_matrix, latent_sim, query, top_n=5, method="hybrid"):
    method = method.lower()
    
    if method == "exact" and query in df["Scheme Name"].values:
        idx = df[df["Scheme Name"] == query].index[0]
        similarity_scores = list(enumerate(latent_sim[idx]))

    elif method in ["content", "hybrid"]:
        processed_query = enhanced_preprocess_text(query)
        query_vec = vectorizer.transform([processed_query])
        query_latent = svd.transform(query_vec)
        similarity_scores = cosine_similarity(query_latent, latent_matrix).flatten()
        similarity_scores = list(enumerate(similarity_scores))

        if method == "hybrid" and query in df["Scheme Name"].values:
            idx = df[df["Scheme Name"] == query].index[0]
            exact_scores = list(enumerate(latent_sim[idx]))
            similarity_scores = [
                (i, 0.7 * similarity_scores[i][1] + 0.3 * exact_scores[i][1])
                for i in range(len(similarity_scores))
            ]
    else:
        return "Invalid method. Use 'exact', 'content', or 'hybrid'."

    # Remove the query itself if it's an exact match
    if query in df["Scheme Name"].values:
        idx = df[df["Scheme Name"] == query].index[0]
        similarity_scores = [score for score in similarity_scores if score[0] != idx]

    # Sort scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Filter top N with distinct scheme names
    recommendations = []
    seen_names = set()

    for idx, score in similarity_scores:
        scheme = df.iloc[idx]
        name = scheme["Scheme Name"]
        if name not in seen_names:
            seen_names.add(name)
            recommendations.append({
                "Scheme Name": name,
                "Details": scheme["Details"],
                "Similarity Score": round(score, 3),
                "State": scheme["State"],
                "Category": scheme["Category"],
                "Apply Link": scheme["Apply Link"] if "Apply Link" in df.columns else "N/A"
            })
        if len(recommendations) >= top_n:
            break

    if not recommendations:
        return "No matching schemes found. Try adjusting the query or method."

    return recommendations




# --- STREAMLIT UI ---
st.set_page_config(page_title="Gov Scheme Recommender", layout="centered")
st.title("🌐 Government Scheme Recommender System")
with st.expander("ℹ️ How does this work?"):
    st.write("""
    This system recommends government schemes based on your profile and interests. It uses Natural Language Processing (NLP) to match your inputs with schemes from a database.

    - **Hybrid Method**: Blends exact and content similarity.
    - **Exact Match**: Finds schemes with names matching your input.
    - **Content-Based**: Matches schemes with similar descriptions.

    You can view the results, check apply links, scan QR codes, and download a PDF report.
    """)

df, vectorizer, svd, latent_matrix, latent_sim = load_and_process_data()

# --- Sidebar: View All Schemes ---
with st.sidebar:
    st.markdown("---")
    if st.checkbox("📚 Show all available schemes"):
        st.subheader("Available Schemes")
        st.dataframe(df[["Scheme Name", "Category", "State", "Apply Link"]], use_container_width=True)



st.header("📝 Enter Your Details")


gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 18, 100, 25)
income = st.selectbox("Monthly Income Range", ["< ₹10,000", "₹10,000 - ₹30,000", "₹30,001 - ₹50,000", "> ₹50,000"])
employment = st.selectbox("Employment Status", ["Unemployed", "Self-employed", "Salaried", "Student"])
education = st.selectbox("Education Level", ["No formal education", "Primary", "Secondary", "Graduate", "Postgraduate"])
location = st.text_input("State/Region")
keywords = st.text_area("Enter Interests or Keywords (e.g., agriculture, women empowerment, education)", height=100)
method = st.radio("Recommendation Method", ["Hybrid", "Exact Match", "Content-Based"], horizontal=True)

# ✅ Method mapping fix
method_map = {
    "Hybrid": "hybrid",
    "Exact Match": "exact",
    "Content-Based": "content"
}
method_code = method_map[method]

# --- Recommend ---
if st.button("🔍 Recommend Schemes"):
    with st.spinner("Finding best matching schemes..."):
        results = recommend_schemes_improved(df, vectorizer, svd, latent_matrix, latent_sim, keywords, method=method_code)

        if isinstance(results, str):
            st.error(results)
        elif results:
            st.success("✅ Top Scheme Recommendations")
            for r in results:
                st.subheader(r["Scheme Name"])
                st.markdown(f"**Category**: {r['Category']}")
                st.markdown(f"**State**: {r['State']}")
                st.markdown(f"**Details**: {r['Details']}")
                st.markdown(f"**Apply Link**: [Click here]({r['Apply Link']})")
                st.markdown("---")

            # --- Enhanced Sidebar Plot ---
            with st.sidebar:
                st.subheader("📊 Similarity Scores")

                chart_df = pd.DataFrame({
                    "Full Scheme Name": [r["Scheme Name"] for r in results],
                    "Scheme": [r["Scheme Name"][:30] + "..." if len(r["Scheme Name"]) > 30 else r["Scheme Name"] for r in results],
                    "Similarity Score": [r["Similarity Score"] for r in results]
                })

                max_score = max(chart_df["Similarity Score"])
                chart_df["Color"] = chart_df["Similarity Score"].apply(
                    lambda x: "crimson" if x == max_score else "steelblue"
                )

                fig = px.bar(
                    chart_df,
                    x="Similarity Score",
                    y="Scheme",
                    orientation="h",
                    color="Color",
                    color_discrete_map="identity",
                    hover_data={"Full Scheme Name": True, "Similarity Score": True, "Scheme": False, "Color": False},
                    height=300 + 30 * len(chart_df)
                )
                fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

                



        else:
            st.warning("No schemes matched your query.")

            


st.markdown("---")
st.caption("🔍 Powered by an NLP-enhanced hybrid recommendation engine.")
