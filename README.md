# Yojana Manthan

**Yojana Manthan** is a user-friendly platform designed to help individuals discover and understand various government schemes. By leveraging natural language processing (NLP) techniques, the application matches user queries to relevant schemes, making it easier for users to find information tailored to their needs.

## 🌐 Live Demo

Experience the application live: [yojana-manthan.streamlit.app](https://yojana-manthan.streamlit.app)

## 📂 Project Structure

- **`app.py`**: Main application script built using Streamlit.
- **`Gov schemes.csv`**: Dataset containing details of various government schemes.
- **`tfidf_matrix.pkl` & `vectorizer (3).pkl`**: Pre-trained models for text vectorization and similarity computation.
- **`requirements.txt`**: Lists all Python dependencies.
- **`setup.sh`**: Shell script for setting up the environment.
- **`.devcontainer/`**: Configuration files for development container setup.
- **`fonts/`**: Directory containing font files used in the application.

## 🚀 Getting Started

To run the application locally:

1. Clone the repo:
```bash
  git clone https://github.com/Yash-Bhatnagar-02/Yojana-Manthan.git
```

2. Install requirements in the terminal:
```bash
  pip install -r requirements
```

3. Run the application with the following command:
```bash
  streamlit run app.py
```
```
## 🛠️ Features

- **Natural Language Querying**: Users can input queries in plain language to find relevant government schemes.
- **Efficient Matching**: Utilizes TF-IDF vectorization and cosine similarity to match user queries with scheme descriptions.
- **Hybrid Recommendation Logic**: Combines TF-IDF, bigrams, lemmatization, and dimensionality reduction (SVD) for more accurate matches.
- **Interactive UI**: Built using Streamlit, offering a clean and responsive user interface.
- **PDF Report Generation**: Generates downloadable PDF reports of the recommended schemes.
- **Easy Deployment**: Simple to deploy on Streamlit Cloud or any Python-compatible hosting.

## 📊 Data Source

The system uses a curated CSV dataset (`Gov schemes.csv`) that contains comprehensive information about various Indian government schemes. Each record in the dataset includes:

- **Scheme Name**
- **Description**
- **Eligibility Criteria**
- **Target Audience**
- **Sector**
- **Link to Apply / Official Website**

This data is used to match user queries using natural language processing techniques.

## 📄 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software as long as the original license and copyright
notice are included in all copies or substantial portions of the software.

For more details, refer to the [LICENSE](LICENSE) file.
