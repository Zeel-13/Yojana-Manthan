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

1. **Clone the repository:**
   ```bash
     git clone https://github.com/Yash-Bhatnagar-02/Yojana-Manthan.git
     cd Yojana-Manthan
   ```



2. **Install requirements in the terminal:**
```bash
  pip install -r requirements
```

3. **Run the application with the following command:**
```bash
  streamlit run app.py
```
