import os
import pickle
import sqlite3
import numpy as np
import pandas as pd

class HybridRecommender:
    def __init__(self, cf_model, content_model, cf_weight=0.6, content_weight=0.4):
        self.cf      = cf_model
        self.cb      = content_model
        self.cf_w    = cf_weight
        self.cb_w    = content_weight
        self.train_df = pd.read_csv("data/processed/train.csv")

    def recommend(self, user_id, top_n=10):
        # Get CF recommendations
        cf_recs = dict(self.cf.recommend_hybrid(user_id, top_n=top_n*2))

        # Get content recommendations
        cb_recs = dict(self.cb.recommend_for_user(user_id, self.train_df, top_n=top_n*2))

        # If both empty fall back to popular
        if not cf_recs and not cb_recs:
            return self.cf.fallback_popular(top_n)

        # Normalize scores to 0-1
        def normalize(d):
            if not d: return d
            max_v = max(d.values())
            if max_v == 0: return d
            return {k: v/max_v for k, v in d.items()}

        cf_recs = normalize(cf_recs)
        cb_recs = normalize(cb_recs)

        # Blend
        all_items = set(cf_recs) | set(cb_recs)
        hybrid    = {}
        for item in all_items:
            hybrid[item] = round(
                self.cf_w * cf_recs.get(item, 0) +
                self.cb_w * cb_recs.get(item, 0), 4
            )

        sorted_recs = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]

    def get_product_details(self, product_ids):
        conn    = sqlite3.connect(os.path.join("database", "smartcart.db"))
        ids_str = ",".join(str(i) for i in product_ids)
        query   = f"""
            SELECT p.product_id, p.name, p.price, p.avg_rating,
                   p.review_count, c.name AS category
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            WHERE p.product_id IN ({ids_str})
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def recommend_with_details(self, user_id, top_n=10):
        recs        = self.recommend(user_id, top_n)
        product_ids = [pid for pid, _ in recs]
        scores      = {pid: score for pid, score in recs}

        if not product_ids:
            return pd.DataFrame()

        details           = self.get_product_details(product_ids)
        details["score"]  = details["product_id"].map(scores)
        details           = details.sort_values("score", ascending=False)
        return details

    def save(self):
        os.makedirs("models", exist_ok=True)
        with open("models/hybrid_model.pkl", "wb") as f:
            pickle.dump(self, f)
        print("✓ Hybrid model saved to models/hybrid_model.pkl")


if __name__ == "__main__":
    print("Loading CF model...")
    with open("models/cf_model.pkl", "rb") as f:
        cf = pickle.load(f)

    print("Loading content model...")
    with open("models/content_model.pkl", "rb") as f:
        cb = pickle.load(f)

    hybrid = HybridRecommender(cf, cb)
    hybrid.save()

    # Test
    sample_users = list(cf.user2idx.keys())[:3]
    for uid in sample_users:
        print(f"\nTop 5 recommendations for user {uid}:")
        df = hybrid.recommend_with_details(uid, top_n=5)
        if df.empty:
            print("  No recommendations")
        else:
            for _, row in df.iterrows():
                print(f"  {row['name'][:35]:<35} ₹{row['price']:>10.0f}  "
                      f"★{row['avg_rating']}  score:{row['score']:.3f}")
