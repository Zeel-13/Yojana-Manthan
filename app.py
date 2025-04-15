import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.data import find
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from fpdf import FPDF
import tempfile

from fpdf import FPDF
import tempfile
import os
import qrcode
from PIL import Image

# --- Safe downloads: check if already downloaded ---
def download_nltk_resource(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')
download_nltk_resource('corpora/wordnet')

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
    
    if method == "exact":
        matched_rows = df[df["Scheme Name"].str.lower() == query.lower()]
        if matched_rows.empty:
            return "No exact match found for the given scheme name."
        
        recommendations = []
        for _, scheme in matched_rows.iterrows():
            recommendations.append({
                "Scheme Name": scheme["Scheme Name"],
                "Details": scheme["Details"],
                "Similarity Score": 1.0,
                "State": scheme["State"],
                "Category": scheme["Category"],
                "Apply Link": scheme["Apply Link"] if "Apply Link" in df.columns else "N/A"
            })
        return recommendations

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

    if query in df["Scheme Name"].values and method != "exact":
        idx = df[df["Scheme Name"] == query].index[0]
        similarity_scores = [score for score in similarity_scores if score[0] != idx]

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

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


# --- PDF GENERATION FUNCTION ---
# --- PDF GENERATION FUNCTION ---
from fpdf import FPDF
import tempfile
import os

class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('Noto', '', 'fonts/NotoSansDevanagari-Regular.ttf', uni=True)
        self.set_font('Noto', '', 12)

def generate_pdf(recommendations, user_profile):
    def clean_text(text):
        return str(text).replace("₹", "Rs.")

    pdf = UnicodePDF()
    pdf.add_page()
    pdf.set_font("Noto", size=16)
    pdf.cell(0, 10, "Government Scheme Recommendations", ln=True, align="C")

    pdf.set_font("Noto", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"""User Profile:
Gender: {user_profile['gender']}
Age: {user_profile['age']}
Income: {clean_text(user_profile['income'])}
Employment: {user_profile['employment']}
Education: {user_profile['education']}
Location: {user_profile['location']}
Keywords: {user_profile['keywords']}
""")

    pdf.ln(5)
    for idx, r in enumerate(recommendations, 1):
        pdf.set_font("Noto", size=14)
        pdf.cell(0, 10, f"{idx}. {clean_text(r['Scheme Name'])}", ln=True)

        pdf.set_font("Noto", size=11)
        pdf.multi_cell(0, 8, f"""Category: {clean_text(r['Category'])}
State: {clean_text(r['State'])}
Details: {clean_text(r['Details'])}
Apply Link: {clean_text(r['Apply Link'])}
""")
        pdf.ln(3)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name


# --- STREAMLIT UI ---
st.set_page_config(page_title="Yojana Manthan", layout="centered")
st.title("🌐 Government Scheme Recommender System")
with st.expander("ℹ️ How does this work?"):
    st.write("""
    This system recommends government schemes based on your profile and interests. It uses Natural Language Processing (NLP) to match your inputs with schemes from a database.

    - **Hybrid Method**: Blends exact and content similarity.
    - **Exact Match**: Finds schemes with names matching your input.
    - **Content-Based**: Matches schemes with similar descriptions.

    You can view the results, check apply links, and download a PDF report of your eligible schemes.
    """)

df, vectorizer, svd, latent_matrix, latent_sim = load_and_process_data()

with st.sidebar:
    st.markdown("---")
    if st.checkbox("📚 Show all available schemes"):
        st.subheader("Available Schemes")
        st.dataframe(df[["Scheme Name", "Category", "State", "Apply Link"]], use_container_width=True)

st.header("📝 Fill Your Profile for Scheme Recommendations")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female", "Other"])
    income = st.selectbox("💰 Monthly Income", ["< ₹10,000", "₹10,000 - ₹30,000", "₹30,001 - ₹50,000", "> ₹50,000"])
    education = st.selectbox("🎓 Education Level", ["No formal education", "Primary", "Secondary", "Graduate", "Postgraduate"])

with col2:
    age = st.slider("🎂 Age", 18, 100, 25)
    employment = st.selectbox("💼 Employment Status", ["Unemployed", "Self-employed", "Salaried", "Student"])
    location = st.text_input("📍 State/Region")

keywords = st.text_area("🧠 Your Interests or Keywords", placeholder="e.g., agriculture, women empowerment, education", height=100)

method = st.radio("🔎 Choose Recommendation Method", ["Hybrid", "Exact Match", "Content-Based"], horizontal=True)


method_map = {
    "Hybrid": "hybrid",
    "Exact Match": "exact",
    "Content-Based": "content"
}
method_code = method_map[method]

if st.button("🚀 Recommend Schemes"):
    if not keywords.strip():
        st.warning("❗ Please enter at least one keyword.")
    else:
        with st.spinner("Analyzing your profile and matching with schemes..."):
            method_map = {
                "Hybrid": "hybrid",
                "Exact Match": "exact",
                "Content-Based": "content"
            }
            method_code = method_map[method]

            user_profile = {
                "gender": gender,
                "age": age,
                "income": income,
                "employment": employment,
                "education": education,
                "location": location,
                "keywords": keywords
            }

            results = recommend_schemes_improved(df, vectorizer, svd, latent_matrix, latent_sim, keywords, method=method_code)

        

            if isinstance(results, str):
                st.error(results)
            elif results:
                st.success("✅ Top Government Scheme Recommendations Based on Your Profile")

                for idx, r in enumerate(results, 1):
                    with st.expander(f"📌 {idx}. {r['Scheme Name']}"):
                        st.markdown(f"**🗂️ Category**: `{r['Category']}`")
                        st.markdown(f"**📍 State**: `{r['State']}`")
                        st.markdown(f"**ℹ️ Details**:\n{r['Details']}")
                        st.markdown(f"**🔗 Apply Link**: [Click here]({r['Apply Link']})")

                # PDF Download
                pdf_path = generate_pdf(results, user_profile)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=f,
                        file_name="scheme_recommendations.pdf",
                        mime="application/pdf"
                    )

                # Chart in sidebar
                with st.sidebar:
                    st.markdown("### 📊 Similarity Score Chart")
                    chart_df = pd.DataFrame({
                        "Scheme": [r["Scheme Name"][:30] + "..." if len(r["Scheme Name"]) > 30 else r["Scheme Name"] for r in results],
                        "Full Name": [r["Scheme Name"] for r in results],
                        "Similarity Score": [r["Similarity Score"] for r in results]
                    })
                    max_score = max(chart_df["Similarity Score"])
                    chart_df["Color"] = chart_df["Similarity Score"].apply(
                        lambda x: "crimson" if x == max_score else "royalblue"
                    )
                    fig = px.bar(
                        chart_df,
                        x="Similarity Score",
                        y="Scheme",
                        orientation="h",
                        color="Color",
                        color_discrete_map="identity",
                        hover_data={"Full Name": True, "Similarity Score": True, "Color": False},
                        height=300 + 30 * len(chart_df)
                    )
                    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No matching schemes found.")


st.markdown("---")
st.caption("🔍 Powered by an NLP-enhanced hybrid recommendation engine.")
