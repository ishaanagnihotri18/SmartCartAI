import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class MatrixBuilder:
    def __init__(self, train_path="data/processed/train.csv"):
        self.train_path = train_path
        self.matrix_df  = None
        self.sparse_mat = None
        self.user2idx   = {}
        self.idx2user   = {}
        self.item2idx   = {}
        self.idx2item   = {}

    def load(self):
        df = pd.read_csv(self.train_path, parse_dates=["event_time"])
        print(f"  Loaded {len(df):,} training interactions")
        return df

    def build_dense(self, df):
        matrix = df.pivot_table(
            index="user_id", columns="product_id",
            values="score", aggfunc="max"
        ).fillna(0)
        print(f"  Matrix shape: {matrix.shape[0]} users x {matrix.shape[1]} products")
        sparsity = 1 - (df.shape[0] / (matrix.shape[0] * matrix.shape[1]))
        print(f"  Sparsity: {sparsity:.1%}")
        return matrix

    def build_sparse(self, df):
        users    = df["user_id"].unique()
        products = df["product_id"].unique()
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {p: i for i, p in enumerate(products)}
        self.idx2item = {i: p for p, i in self.item2idx.items()}
        rows   = df["user_id"].map(self.user2idx).values
        cols   = df["product_id"].map(self.item2idx).values
        scores = df["score"].values
        sparse = csr_matrix((scores, (rows, cols)),
                            shape=(len(users), len(products)))
        print(f"  Sparse matrix: {sparse.shape}, {sparse.nnz} stored values")
        return sparse

    def print_stats(self, df):
        print("\n── Interaction Stats ───────────────")
        print(f"  Unique users:    {df['user_id'].nunique():>6}")
        print(f"  Unique products: {df['product_id'].nunique():>6}")
        print(f"  Event breakdown:")
        for evt, count in df["event_type"].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {evt:<15} {count:>5}  {pct:.1f}%")
        print("────────────────────────────────────")

    def save(self, out_dir="data/processed"):
        os.makedirs(out_dir, exist_ok=True)
        self.matrix_df.to_csv(f"{out_dir}/user_item_matrix.csv")
        payload = {
            "sparse":   self.sparse_mat,
            "user2idx": self.user2idx,
            "idx2user": self.idx2user,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
        }
        with open(f"{out_dir}/matrix_data.pkl", "wb") as f:
            pickle.dump(payload, f)
        print(f"  Saved to {out_dir}/")

    def build(self, save=True):
        print("\nSmartCartAI — Matrix Builder")
        print("=" * 40)
        df              = self.load()
        self.print_stats(df)
        self.matrix_df  = self.build_dense(df)
        self.sparse_mat = self.build_sparse(df)
        if save:
            self.save()
        mappings = {
            "user2idx": self.user2idx, "idx2user": self.idx2user,
            "item2idx": self.item2idx, "idx2item": self.idx2item,
        }
        print("\n✓ Matrix build complete\n")
        return self.matrix_df, self.sparse_mat, mappings

if __name__ == "__main__":
    builder = MatrixBuilder()
    matrix, sparse, mappings = builder.build()
    print("Matrix preview (5 users x 8 products):")
    print(matrix.iloc[:5, :8].to_string())