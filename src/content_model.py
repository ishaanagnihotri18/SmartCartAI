import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/smartcart.db")

class ContentBasedFilter:
    def __init__(self):
        self.products_df  = None
        self.tfidf_matrix = None
        self.vectorizer   = None
        self.product_ids  = None

    # ── Load products ─────────────────────────────────────────────────────────
    def load_products(self):
        conn  = sqlite3.connect(DB_PATH)
        query = """
            SELECT p.product_id, p.name, p.description,
                   p.price, p.avg_rating, p.review_count,
                   c.name AS category,
                   GROUP_CONCAT(pt.tag, ' ') AS tags
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN product_tags pt ON p.product_id = pt.product_id
            GROUP BY p.product_id
        """
        self.products_df = pd.read_sql_query(query, conn)
        conn.close()
        self.products_df["tags"]        = self.products_df["tags"].fillna("")
        self.products_df["description"] = self.products_df["description"].fillna("")
        print(f"✓ Loaded {len(self.products_df)} products")
        return self.products_df

    # ── Build TF-IDF ──────────────────────────────────────────────────────────
    def build_tfidf(self):
        # Combine text fields into one corpus
        self.products_df["corpus"] = (
            self.products_df["name"]        + " " +
            self.products_df["category"]    + " " +
            self.products_df["tags"]        + " " +
            self.products_df["description"]
        )
        self.vectorizer  = TfidfVectorizer(
            stop_words="english",
            max_features=500,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.products_df["corpus"])
        self.product_ids  = self.products_df["product_id"].tolist()
        print(f"✓ TF-IDF matrix: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix

    # ── Get similar products ──────────────────────────────────────────────────
    def get_similar_products(self, product_id, top_n=10):
        if product_id not in self.product_ids:
            return []
        idx    = self.product_ids.index(product_id)
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores[idx] = 0  # exclude self
        top_idxs = np.argsort(scores)[::-1][:top_n]
        return [(self.product_ids[i], round(float(scores[i]), 4)) for i in top_idxs]

    # ── Recommend for user ────────────────────────────────────────────────────
    def recommend_for_user(self, user_id, interaction_df, top_n=10):
        user_items = interaction_df[interaction_df["user_id"] == user_id]
        if user_items.empty:
            return []

        # Get top interacted products by score
        top_items = (user_items.sort_values("score", ascending=False)
                               .head(5)["product_id"].tolist())

        scores = np.zeros(len(self.product_ids))
        seen   = set(interaction_df[interaction_df["user_id"] == user_id]["product_id"])

        for pid in top_items:
            if pid not in self.product_ids:
                continue
            idx = self.product_ids.index(pid)
            sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
            scores += sim

        # Zero out already seen
        for pid in seen:
            if pid in self.product_ids:
                scores[self.product_ids.index(pid)] = 0

        top_idxs = np.argsort(scores)[::-1][:top_n]
        return [(self.product_ids[i], round(float(scores[i]), 4))
                for i in top_idxs if scores[i] > 0]

    # ── Search by query ───────────────────────────────────────────────────────
    def search(self, query, top_n=10):
        query_vec = self.vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idxs  = np.argsort(scores)[::-1][:top_n]
        results   = []
        for i in top_idxs:
            if scores[i] > 0:
                row = self.products_df.iloc[i]
                results.append({
                    "product_id": row["product_id"],
                    "name":       row["name"],
                    "category":   row["category"],
                    "price":      row["price"],
                    "rating":     row["avg_rating"],
                    "score":      round(float(scores[i]), 4)
                })
        return results

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self):
        print("\nSmartCartAI — Content-Based Filter")
        print("=" * 50)
        self.load_products()
        self.build_tfidf()
        os.makedirs("models", exist_ok=True)
        with open("models/content_model.pkl", "wb") as f:
            pickle.dump(self, f)
        print("✓ Model saved to models/content_model.pkl\n")
        return self


if __name__ == "__main__":
    cb = ContentBasedFilter()
    cb.build()

    # Test similar products
    sample_id = cb.product_ids[0]
    print(f"\nProducts similar to product {sample_id}:")
    for pid, score in cb.get_similar_products(sample_id, top_n=5):
        row = cb.products_df[cb.products_df["product_id"] == pid].iloc[0]
        print(f"  {row['name'][:40]:<40} score: {score}")

    # Test search
    print("\nSearch results for 'laptop bluetooth':")
    for r in cb.search("laptop bluetooth", top_n=5):
        print(f"  {r['name'][:40]:<40} score: {r['score']}")
