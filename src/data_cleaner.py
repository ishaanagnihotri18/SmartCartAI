import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/smartcart.db")

EVENT_WEIGHTS = {
    "view":        1.0,
    "click":       2.0,
    "add_to_cart": 3.0,
    "purchase":    5.0,
    "rating":      4.0,
}

class DataCleaner:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path)
        print(f"✓ Connected to database")

    def load_interactions(self):
        query = """
            SELECT i.interaction_id, i.user_id, i.product_id,
                   i.event_type, i.rating, i.session_id, i.event_time,
                   p.category_id, p.avg_rating AS product_avg_rating
            FROM interactions i
            JOIN products p ON i.product_id = p.product_id
        """
        df = pd.read_sql_query(query, self.conn, parse_dates=["event_time"])
        print(f"  Loaded {len(df):,} raw interactions")
        return df

    def remove_duplicates(self, df):
        weight_order = {"view":1,"click":2,"add_to_cart":3,"rating":4,"purchase":5}
        df["_weight"] = df["event_type"].map(weight_order)
        df = (df.sort_values("_weight", ascending=False)
                .drop_duplicates(subset=["user_id","product_id"], keep="first")
                .drop(columns=["_weight"]))
        print(f"  After dedup: {len(df):,} interactions")
        return df

    def filter_low_activity(self, df, min_user=3, min_product=3):
        uc = df["user_id"].value_counts()
        pc = df["product_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= min_user].index) &
                df["product_id"].isin(pc[pc >= min_product].index)]
        print(f"  After filter: {len(df):,} interactions, "
              f"{df['user_id'].nunique()} users, {df['product_id'].nunique()} products")
        return df

    def assign_scores(self, df):
        def score_row(row):
            if row["event_type"] == "rating" and pd.notna(row["rating"]) and row["rating"] > 0:
                return float(row["rating"])
            return EVENT_WEIGHTS.get(row["event_type"], 1.0)
        df["score"]      = df.apply(score_row, axis=1)
        df["score_norm"] = df["score"] / 5.0
        print(f"  Score range: [{df['score'].min():.1f}, {df['score'].max():.1f}]")
        return df

    def handle_missing(self, df):
        df["rating"]     = df["rating"].fillna(0.0)
        df["session_id"] = df["session_id"].fillna(-1).astype(int)
        return df

    def train_test_split(self, df, test_ratio=0.2):
        df        = df.sort_values("event_time")
        split_idx = int(len(df) * (1 - test_ratio))
        train     = df.iloc[:split_idx].copy()
        test      = df.iloc[split_idx:].copy()
        train_users    = set(train["user_id"])
        train_products = set(train["product_id"])
        test = test[test["user_id"].isin(train_users) &
                    test["product_id"].isin(train_products)]
        print(f"  Train: {len(train):,} | Test: {len(test):,}")
        return train, test

    def run(self, save_csv=True):
        print("\nSmartCartAI — Data Cleaning")
        print("=" * 40)
        df    = self.load_interactions()
        df    = self.remove_duplicates(df)
        df    = self.filter_low_activity(df)
        df    = self.assign_scores(df)
        df    = self.handle_missing(df)
        train, test = self.train_test_split(df)
        if save_csv:
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv("data/processed/interactions_clean.csv", index=False)
            train.to_csv("data/processed/train.csv", index=False)
            test.to_csv("data/processed/test.csv", index=False)
            print("  Saved to data/processed/")
        print("\n✓ Data cleaning complete\n")
        return df, train, test

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    cleaner = DataCleaner()
    df, train, test = cleaner.run()
    print(df[["user_id","product_id","event_type","score"]].head(10).to_string())
    cleaner.close()