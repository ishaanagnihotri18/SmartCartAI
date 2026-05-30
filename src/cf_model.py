import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MATRIX_PATH = "data/processed/matrix_data.pkl"

class CollaborativeFilter:
    def __init__(self):
        self.matrix_df   = None
        self.sparse_mat  = None
        self.user2idx    = {}
        self.idx2user    = {}
        self.item2idx    = {}
        self.idx2item    = {}
        self.user_sim    = None
        self.item_sim    = None

    # ── Load matrix ───────────────────────────────────────────────────────────
    def load(self, path=MATRIX_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.sparse_mat = data["sparse"]
        self.user2idx   = data["user2idx"]
        self.idx2user   = data["idx2user"]
        self.item2idx   = data["item2idx"]
        self.idx2item   = data["idx2item"]
        self.matrix_df  = pd.read_csv("data/processed/user_item_matrix.csv", index_col=0)
        self.matrix_df.index    = self.matrix_df.index.astype(int)
        self.matrix_df.columns  = self.matrix_df.columns.astype(int)
        print(f"✓ Loaded matrix: {self.sparse_mat.shape}")

    # ── Compute similarities ──────────────────────────────────────────────────
    def compute_user_similarity(self):
        dense         = self.sparse_mat.toarray()
        self.user_sim = cosine_similarity(dense)
        print(f"✓ User similarity matrix: {self.user_sim.shape}")
        return self.user_sim

    def compute_item_similarity(self):
        dense         = self.sparse_mat.toarray().T
        self.item_sim = cosine_similarity(dense)
        print(f"✓ Item similarity matrix: {self.item_sim.shape}")
        return self.item_sim

    # ── User-based CF ─────────────────────────────────────────────────────────
    def get_similar_users(self, user_id, top_n=10):
        if user_id not in self.user2idx:
            return []
        idx      = self.user2idx[user_id]
        scores   = self.user_sim[idx]
        top_idxs = np.argsort(scores)[::-1][1:top_n+1]
        return [(self.idx2user[i], round(float(scores[i]), 4)) for i in top_idxs]

    def recommend_user_based(self, user_id, top_n=10):
        if user_id not in self.user2idx:
            return self.fallback_popular(top_n)

        idx          = self.user2idx[user_id]
        user_scores  = self.user_sim[idx]
        matrix_array = self.sparse_mat.toarray()
        user_vector  = matrix_array[idx]

        # Weighted sum of similar users' interactions
        weighted     = user_scores @ matrix_array
        # Zero out items the user already interacted with
        weighted[user_vector > 0] = 0

        top_idxs = np.argsort(weighted)[::-1][:top_n]
        return [(self.idx2item[i], round(float(weighted[i]), 4))
                for i in top_idxs if weighted[i] > 0]

    # ── Item-based CF ─────────────────────────────────────────────────────────
    def get_similar_items(self, product_id, top_n=10):
        if product_id not in self.item2idx:
            return []
        idx      = self.item2idx[product_id]
        scores   = self.item_sim[idx]
        top_idxs = np.argsort(scores)[::-1][1:top_n+1]
        return [(self.idx2item[i], round(float(scores[i]), 4)) for i in top_idxs]

    def recommend_item_based(self, user_id, top_n=10):
        if user_id not in self.user2idx:
            return self.fallback_popular(top_n)

        idx          = self.user2idx[user_id]
        user_vector  = self.sparse_mat.toarray()[idx]
        interacted   = np.where(user_vector > 0)[0]

        if len(interacted) == 0:
            return self.fallback_popular(top_n)

        # Average similarity across all items user interacted with
        scores = np.zeros(self.item_sim.shape[0])
        for item_idx in interacted:
            scores += self.item_sim[item_idx] * user_vector[item_idx]

        scores[interacted] = 0  # remove already seen
        top_idxs = np.argsort(scores)[::-1][:top_n]
        return [(self.idx2item[i], round(float(scores[i]), 4))
                for i in top_idxs if scores[i] > 0]

    # ── Hybrid ────────────────────────────────────────────────────────────────
    def recommend_hybrid(self, user_id, top_n=10, user_weight=0.5, item_weight=0.5):
        user_recs = dict(self.recommend_user_based(user_id, top_n=top_n*2))
        item_recs = dict(self.recommend_item_based(user_id, top_n=top_n*2))

        all_items = set(user_recs) | set(item_recs)
        hybrid    = {}
        for item in all_items:
            u_score      = user_recs.get(item, 0)
            i_score      = item_recs.get(item, 0)
            hybrid[item] = round(user_weight * u_score + item_weight * i_score, 4)

        sorted_recs = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]

    # ── Fallback ──────────────────────────────────────────────────────────────
    def fallback_popular(self, top_n=10):
        col_sums = self.sparse_mat.toarray().sum(axis=0)
        top_idxs = np.argsort(col_sums)[::-1][:top_n]
        return [(self.idx2item[i], round(float(col_sums[i]), 4)) for i in top_idxs]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(self, test_path="data/processed/test.csv", k=10):
        test_df   = pd.read_csv(test_path)
        test_dict = test_df.groupby("user_id")["product_id"].apply(set).to_dict()

        precisions, recalls, hits = [], [], []

        for user_id, actual_items in test_dict.items():
            if user_id not in self.user2idx:
                continue
            recs        = [pid for pid, _ in self.recommend_hybrid(user_id, top_n=k)]
            hits_count  = len(set(recs) & actual_items)
            precisions.append(hits_count / k)
            recalls.append(hits_count / len(actual_items) if actual_items else 0)
            hits.append(1 if hits_count > 0 else 0)

        print("\n── Evaluation Results ──────────────────")
        print(f"  Users evaluated : {len(precisions)}")
        print(f"  Precision@{k}    : {np.mean(precisions):.4f}")
        print(f"  Recall@{k}       : {np.mean(recalls):.4f}")
        print(f"  Hit Rate@{k}     : {np.mean(hits):.4f}")
        print("────────────────────────────────────────")
        return {
            "precision": np.mean(precisions),
            "recall":    np.mean(recalls),
            "hit_rate":  np.mean(hits)
        }

    # ── Build everything ──────────────────────────────────────────────────────
    def build(self):
        print("\nSmartCartAI — Phase 2: Collaborative Filtering")
        print("=" * 50)
        self.load()
        self.compute_user_similarity()
        self.compute_item_similarity()

        # Save model
        os.makedirs("models", exist_ok=True)
        with open("models/cf_model.pkl", "wb") as f:
            pickle.dump(self, f)
        print("✓ Model saved to models/cf_model.pkl")

        # Evaluate
        self.evaluate()
        print("\n✓ Phase 2 Complete!\n")
        return self


if __name__ == "__main__":
    cf = CollaborativeFilter()
    cf.build()

    # Test recommendations
    sample_users = list(cf.user2idx.keys())[:3]
    for user_id in sample_users:
        print(f"\nTop 5 recommendations for user {user_id}:")
        recs = cf.recommend_hybrid(user_id, top_n=5)
        for rank, (pid, score) in enumerate(recs, 1):
            print(f"  {rank}. Product {pid}  (score: {score})")
