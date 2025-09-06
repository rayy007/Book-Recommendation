# ==========================================================
# Streamlit Book Recommender (Hybrid + CF + Popularity)
# ==========================================================
import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
   df = pd.read_csv("generalized_books_dataset.csv")
   df.to_parquet("generalized_books_dataset.parquet", engine="pyarrow", index=False)

    df = df.dropna(subset=["user_id", "book_id", "rating"])
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)
    df["rating"] = df["rating"].astype(float)

    return df

df = load_data()

# ------------------------------
# Build Sparse User–Item Matrix (for CF + Hybrid)
# ------------------------------
@st.cache_resource
def build_cf(df):
    u_codes = df["user_id"].astype("category")
    i_codes = df["book_id"].astype("category")

    user_ids = u_codes.cat.categories.tolist()
    item_ids = i_codes.cat.categories.tolist()

    row = u_codes.cat.codes.to_numpy()
    col = i_codes.cat.codes.to_numpy()
    val = df["rating"].astype(np.float32).to_numpy()

    n_users = len(user_ids)
    n_items = len(item_ids)

    UI = csr_matrix((val, (row, col)), shape=(n_users, n_items))

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    # Try SVD
    user_factors, item_factors, svd_model = None, None, None
    min_dim = min(n_users, n_items)
    if min_dim >= 3:
        k = max(2, min(50, min_dim - 1))
        svd_model = TruncatedSVD(n_components=k, random_state=42)
        user_factors = svd_model.fit_transform(UI)     # (n_users, k)
        item_factors = svd_model.components_.T         # (n_items, k)

    return UI, user_ids, item_ids, user_to_idx, item_to_idx, user_factors, item_factors

UI, user_ids, item_ids, user_to_idx, item_to_idx, user_factors, item_factors = build_cf(df)

# ------------------------------
# Content-based Setup (for Hybrid)
# ------------------------------
@st.cache_resource
def build_tfidf():
    meta_books = (
        df.drop_duplicates("book_id")
        .set_index("book_id")
        .loc[item_ids]   # align to CF items
        .reset_index()
    )

    if "description" in meta_books.columns:
        text_data = meta_books["description"].fillna("")
    else:
        text_data = (
            meta_books[["title", "author", "publisher"]]
            .astype(str)
            .fillna("")
            .agg(" ".join, axis=1)
        )

    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = tfidf.fit_transform(text_data)

    return tfidf_matrix, meta_books

tfidf_matrix, meta_books = build_tfidf()

# ------------------------------
# Helper: Map indices → books
# ------------------------------
def books_from_ids(bids, top_n):
    out = (
        df.loc[df["book_id"].isin(bids), ["book_id", "title", "author", "publisher"]]
          .drop_duplicates("book_id")
    )
    order = {bid: pos for pos, bid in enumerate(bids)}
    out["__ord__"] = out["book_id"].map(order)
    return out.sort_values("__ord__").drop(columns="__ord__").head(top_n)

# ------------------------------
# Recommendation Functions
# ------------------------------
def recommend_cf_svd(user_id, top_n=10):
    uidx = user_to_idx.get(user_id)
    if uidx is None:
        return pd.DataFrame({"message": [f"user_id {user_id} not found"]})

    if user_factors is None or item_factors is None:
        return pd.DataFrame({"message": ["SVD model not available"]})

    uvec = user_factors[uidx]
    scores = item_factors @ uvec

    rated_mask = UI[uidx].toarray().ravel() > 0
    scores[rated_mask] = -np.inf

    top_idx = np.argpartition(-scores, range(min(top_n, len(scores)-1)))[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    chosen_ids = [item_ids[i] for i in top_idx]

    return books_from_ids(chosen_ids, top_n)

def recommend_popularity(top_n=10):
    pop_df = (
        df.groupby("book_id")
        .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
        .reset_index()
    )
    pop_df = pop_df.sort_values(
        by=["num_ratings", "avg_rating"], ascending=[False, False]
    ).head(top_n)

    return books_from_ids(pop_df["book_id"].tolist(), top_n)

def recommend_hybrid(user_id, top_n=10, alpha=0.5):
    uidx = user_to_idx.get(user_id)
    if uidx is None:
        return pd.DataFrame({"message": [f"user_id {user_id} not found"]})

    # CF scores
    cf_scores = np.zeros(len(item_ids), dtype=np.float32)
    if user_factors is not None:
        cf_scores = item_factors @ user_factors[uidx]

    # Content scores
    cont_scores = np.zeros(len(item_ids), dtype=np.float32)
    rated_items = UI[uidx].indices
    if len(rated_items) > 0:
        sim = cosine_similarity(tfidf_matrix[rated_items], tfidf_matrix).mean(axis=0)
        cont_scores = sim.ravel()

    scores = alpha * cf_scores + (1 - alpha) * cont_scores
    scores[rated_items] = -np.inf

    top_idx = np.argpartition(-scores, range(min(top_n, len(scores)-1)))[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    chosen_ids = [item_ids[i] for i in top_idx]

    return books_from_ids(chosen_ids, top_n)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📚 Book Recommendation System")

# User selection (only needed for CF & Hybrid)
user_id = st.selectbox("Select User ID:", user_ids[:500])  # limit for demo
model_choice = st.radio("Choose Recommendation Model:", ["Collaborative (SVD)", "Popularity Baseline", "Hybrid (CF + Content)"])

top_n = st.slider("Number of Recommendations", 5, 20, 10)

if model_choice == "Hybrid":
    alpha = st.slider("Weight for Collaborative Filtering (alpha)", 0.0, 1.0, 0.5)
else:
    alpha = 0.5

if st.button("Recommend"):
    if model_choice == "Collaborative (SVD)":
        recs = recommend_cf_svd(user_id, top_n)
    elif model_choice == "Popularity Baseline":
        recs = recommend_popularity(top_n)
    else:
        recs = recommend_hybrid(user_id, top_n, alpha)

    st.subheader("Recommended Books:")
    st.dataframe(recs)

