# Latent Semantic Analysis on Book Titles

This project demonstrates the application of **Latent Semantic Analysis (LSA)** to analyze semantic relationships among book titles. Using a dataset of diverse book titles, we explore how LSA can reduce dimensionality in text data, cluster semantically similar items, and uncover latent topics within the corpus.

## Project Structure

- **Dataset**: `all_book_titles.txt` - Contains a large collection of book titles.
- **Notebook**: `LSA.ipynb` - Implements LSA on the dataset to analyze semantic relationships among titles.

## Overview

Latent Semantic Analysis is a natural language processing (NLP) technique used to uncover underlying structures in text data by reducing the dimensionality of the data. By applying LSA, we can capture the hidden (latent) topics in our corpus of book titles, which enables us to understand thematic clusters, related titles, and potential subject areas.

### Key Features
- **Preprocessing of Text Data**: Tokenizes, normalizes, and vectorizes the text for analysis.
- **Singular Value Decomposition (SVD)**: Uses SVD to reduce dimensionality and extract meaningful semantic structures.
- **Clustering**: Groups similar book titles based on extracted latent factors.
- **Topic Analysis**: Identifies core topics or themes across the dataset of book titles.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/lsa-book-titles.git
   cd lsa-book-titles
   ```

2. **Install dependencies**:
   The project requires Python 3 and the following packages:
   ```bash
   pip install numpy pandas matplotlib scikit-learn nltk
   ```

3. **Download NLTK resources**:
   In your Python environment, run:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage

1. **Load the Dataset**:
   The dataset of book titles, `all_book_titles.txt`, should be in the root directory. The notebook `LSA.ipynb` will read and preprocess this file.

2. **Run the Notebook**:
   Open `LSA.ipynb` and execute the cells step-by-step. The notebook walks you through:
   - Text preprocessing
   - Applying TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - Performing SVD for dimensionality reduction
   - Visualizing clusters and identifying latent topics

3. **Analyze Results**:
   The notebook includes visualization of clusters and SVD components, which represent latent topics. You can explore:
   - **Top terms** associated with each topic.
   - **Clusters** of book titles that share thematic content.

## Code Overview

### Data Preprocessing
- **Tokenization and Stopword Removal**: Tokenizes book titles and removes common English stopwords to focus on key terms.
- **TF-IDF Vectorization**: Converts text to numerical format, emphasizing unique terms in each title.
  
### Latent Semantic Analysis
- **SVD**: Decomposes the TF-IDF matrix to identify latent topics and reduce data dimensionality.
- **Explained Variance**: Evaluates how much of the original data’s information is captured in reduced dimensions.

### Clustering and Visualization
- **K-Means Clustering**: Groups titles into clusters based on similarity in latent topic space.
- **2D Visualization**: Projects clusters to 2D space for visualization, using principal component analysis (PCA) for clarity.

## Results

The LSA approach reveals distinct clusters among the book titles, providing insights into thematic content across the dataset. The topics extracted highlight major subject areas within the corpus, with each topic represented by a set of terms that frequently appear together.

## Example

Here’s a brief example showing how to extract topics using the `scikit-learn` library for SVD:

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Load data
with open('all_book_titles.txt', 'r') as f:
    book_titles = f.readlines()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(book_titles)

# SVD for dimensionality reduction
svd = TruncatedSVD(n_components=10, random_state=42)
X_reduced = svd.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_reduced)

# Display topics
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
    print(f"Topic {i}:")
    print(" ".join([t[0] for t in sorted_terms]))
```

## Future Enhancements
- **Hyperparameter Tuning**: Optimize the number of dimensions in SVD and the number of clusters.
- **Additional Preprocessing**: Apply stemming or lemmatization to further normalize text.
- **Advanced Topic Modeling**: Consider using LDA (Latent Dirichlet Allocation) for probabilistic topic modeling.

## Contributing

We welcome contributions! If you’d like to contribute, please:
1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests if necessary
4. Submit a pull request

Please ensure your contributions are well-documented.
