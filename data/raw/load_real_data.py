"""
SmartCartAI — Real Data Integration
-------------------------------------
Replaces synthetic products with real Flipkart products.
Keeps existing users and regenerates interactions + models.

Run: python load_real_data.py
"""

import os
import sys
import sqlite3
import random
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH  = "database/smartcart.db"
CSV_PATH = "data/raw/flipkart_com-ecommerce_sample.csv"

random.seed(42)
np.random.seed(42)

def random_date(days_back=180):
    return datetime.now() - timedelta(days=random.randint(0, days_back))

# ── Step 1: Load and clean Flipkart CSV ───────────────────────────────────────
def load_flipkart_data():
    print("Loading Flipkart dataset...")
    df = pd.read_csv(CSV_PATH)

    # Clean product name
    df["product_name"] = df["product_name"].fillna("Unknown Product").str.strip()

    # Clean price — use discounted price, fallback to retail
    df["discounted_price"] = pd.to_numeric(
        df["discounted_price"].astype(str).str.replace(",","").str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )
    df["retail_price"] = pd.to_numeric(
        df["retail_price"].astype(str).str.replace(",","").str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )
    df["price"] = df["discounted_price"].fillna(df["retail_price"]).fillna(999)

    # Clean rating
    df["product_rating"] = pd.to_numeric(df["product_rating"], errors="coerce").fillna(3.5)
    df["product_rating"] = df["product_rating"].clip(1, 5)

    # Clean description
    df["description"] = df["description"].fillna("").str.strip()
    df["description"] = df["description"].str[:500]  # limit length

    # Extract main category from category tree
    def extract_category(tree):
        try:
            parts = str(tree).replace('"','').replace('[','').replace(']','').split('>>')
            return parts[0].strip() if parts else "General"
        except:
            return "General"

    df["main_category"] = df["product_category_tree"].apply(extract_category)

    # Extract sub category
    def extract_subcategory(tree):
        try:
            parts = str(tree).replace('"','').replace('[','').replace(']','').split('>>')
            return parts[1].strip() if len(parts) > 1 else parts[0].strip()
        except:
            return "General"

    df["sub_category"] = df["product_category_tree"].apply(extract_subcategory)

    # Clean brand
    df["brand"] = df["brand"].fillna("Unknown").str.strip()

    # Filter valid products
    df = df[df["price"] > 0]
    df = df[df["product_name"].str.len() > 3]
    df = df.drop_duplicates(subset=["product_name"])

    print(f"  Loaded {len(df):,} valid products")
    print(f"  Categories: {df['main_category'].nunique()}")
    print(f"  Price range: ₹{df['price'].min():.0f} — ₹{df['price'].max():.0f}")
    return df.reset_index(drop=True)


# ── Step 2: Get top categories and sample products ────────────────────────────
def select_products(df, n=200):
    print(f"\nSelecting top {n} products...")

    # Get top categories by count
    top_cats = df["main_category"].value_counts().head(10).index.tolist()
    df_filtered = df[df["main_category"].isin(top_cats)]

    # Sample evenly across categories
    per_cat = n // len(top_cats)
    sampled = []
    for cat in top_cats:
        cat_df = df_filtered[df_filtered["main_category"] == cat]
        sample_size = min(per_cat, len(cat_df))
        sampled.append(cat_df.sample(sample_size, random_state=42))

    result = pd.concat(sampled).head(n).reset_index(drop=True)
    print(f"  Selected {len(result)} products from {result['main_category'].nunique()} categories")
    for cat, count in result["main_category"].value_counts().items():
        print(f"    {cat:<40} {count:>3} products")
    return result


# ── Step 3: Rebuild database with real products ───────────────────────────────
def rebuild_database(products_df):
    print("\nRebuilding database...")

    # Remove old DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("  Removed old database")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    # Read and execute schema
    with open("database/schema.sql", "r") as f:
        conn.executescript(f.read())
    print("  Schema created")

    cur = conn.cursor()

    # ── Insert categories ──────────────────────────────────────────────────────
    categories = products_df["main_category"].unique().tolist()
    cat_ids = {}
    for cat in categories:
        cur.execute("INSERT INTO categories (name) VALUES (?)", (cat,))
        cat_ids[cat] = cur.lastrowid

    # Add sub categories
    sub_cats = products_df["sub_category"].unique().tolist()
    for sub in sub_cats:
        parent = products_df[products_df["sub_category"] == sub]["main_category"].iloc[0]
        if sub not in cat_ids and sub != parent:
            cur.execute("INSERT INTO categories (name, parent_id) VALUES (?,?)",
                       (sub, cat_ids[parent]))
            cat_ids[sub] = cur.lastrowid

    conn.commit()
    print(f"  Inserted {len(cat_ids)} categories")

    # ── Insert products ────────────────────────────────────────────────────────
    product_ids = []
    for _, row in products_df.iterrows():
        cat_id = cat_ids.get(row["main_category"], 1)
        cur.execute("""
            INSERT INTO products
                (name, description, price, category_id, avg_rating, review_count, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (
            str(row["product_name"])[:200],
            str(row["description"])[:500],
            float(row["price"]),
            cat_id,
            float(row["product_rating"]),
            random.randint(10, 5000),
            random_date(400).isoformat()
        ))
        pid = cur.lastrowid
        product_ids.append(pid)

        # Tags from brand + category
        tags = set()
        if str(row["brand"]) not in ["Unknown", "nan"]:
            tags.add(str(row["brand"]).lower()[:50])
        tags.add(str(row["main_category"]).lower()[:50])
        tags.add(str(row["sub_category"]).lower()[:50])

        for tag in tags:
            if tag and len(tag) > 1:
                cur.execute("INSERT INTO product_tags (product_id, tag) VALUES (?,?)",
                           (pid, tag))

    conn.commit()
    print(f"  Inserted {len(product_ids)} products")

    # ── Insert users ───────────────────────────────────────────────────────────
    FIRST_NAMES = ["Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Ayaan",
                    "Krishna","Ishaan","Priya","Ananya","Pooja","Neha","Sneha","Riya",
                    "Divya","Meera","Kavya","Shruti","Rahul","Rohit","Amit","Suresh",
                    "Rajesh","Vijay","Prakash","Manoj","Deepak","Sunil","Ankita","Preeti",
                    "Sunita","Geeta","Rekha","Vinod","Ramesh","Naresh","Mukesh","Dinesh"]
    LAST_NAMES  = ["Sharma","Verma","Patel","Singh","Kumar","Gupta","Joshi","Mehta",
                    "Shah","Rao","Nair","Iyer","Reddy","Pillai","Mishra","Tiwari",
                    "Pandey","Dubey","Srivastava","Chauhan","Yadav","Malhotra","Kapoor",
                    "Bose","Das","Ghosh","Banerjee","Mukherjee","Chatterjee","Sengupta"]
    AGE_GROUPS  = ["18-24","25-34","35-44","45-54","55+"]
    GENDERS     = ["M","F","Other"]
    CITIES      = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai",
                    "Kolkata","Pune","Ahmedabad","Jaipur","Lucknow"]

    user_ids = []
    count    = 0
    for fn in FIRST_NAMES:
        for ln in LAST_NAMES:
            if count >= 500: break
            for num in range(1, 5):
                if count >= 500: break
                username = f"{fn.lower()}{ln.lower()}{num}"
                email    = f"{fn.lower()}.{ln.lower()}{num}@gmail.com"
                created  = random_date(730)
                last_act = created + timedelta(days=random.randint(0, 365))
                cur.execute("""
                    INSERT INTO users
                        (username, email, age_group, gender, location, created_at, last_active)
                    VALUES (?,?,?,?,?,?,?)
                """, (username, email,
                      random.choice(AGE_GROUPS),
                      random.choice(GENDERS),
                      random.choice(CITIES),
                      created.isoformat(),
                      last_act.isoformat()))
                user_ids.append(cur.lastrowid)
                count += 1

    conn.commit()
    print(f"  Inserted {len(user_ids)} users")

    # ── Insert interactions ────────────────────────────────────────────────────
    EVENT_TYPES   = ["view","click","add_to_cart","purchase","rating"]
    EVENT_WEIGHTS = [0.45,  0.25,  0.15,          0.10,      0.05]

    session_counter = 1
    for _ in range(8000):
        user_id    = random.choice(user_ids)
        product_id = random.choice(product_ids)
        event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
        rating     = round(random.uniform(1, 5), 1) if event_type == "rating" else None
        cur.execute("""
            INSERT INTO interactions
                (user_id, product_id, event_type, rating, session_id, event_time)
            VALUES (?,?,?,?,?,?)
        """, (user_id, product_id, event_type, rating,
              session_counter, random_date(180).isoformat()))
        if random.random() < 0.05:
            session_counter += 1

    conn.commit()
    print(f"  Inserted 8000 interactions")

    # Verify
    print("\n── Row counts ──────────────────────")
    for t in ["categories","users","products","product_tags","interactions"]:
        c = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<20} {c:>6}")
    print("────────────────────────────────────")

    conn.close()
    return user_ids, product_ids


# ── Step 4: Rebuild ML models ─────────────────────────────────────────────────
def rebuild_models():
    print("\nRebuilding ML models...")

    # Clean data
    from src.data_cleaner import DataCleaner
    cleaner = DataCleaner()
    df, train, test = cleaner.run(save_csv=True)
    cleaner.close()

    # Build matrix
    from src.matrix_builder import MatrixBuilder
    builder = MatrixBuilder()
    matrix, sparse, maps = builder.build(save=True)

    # Build CF model
    from src.cf_model import CollaborativeFilter
    cf = CollaborativeFilter()
    cf.build()

    # Build content model
    from src.content_model import ContentBasedFilter
    cb = ContentBasedFilter()
    cb.build()

    # Build hybrid
    from src.hybrid_recommender import HybridRecommender
    hybrid = HybridRecommender(cf, cb)
    hybrid.save()

    print("\n✓ All models rebuilt with real data!")
    return cf, cb, hybrid


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SmartCartAI — Real Flipkart Data Integration")
    print("="*55)

    # Load and process CSV
    df          = load_flipkart_data()
    products_df = select_products(df, n=200)

    # Rebuild database
    user_ids, product_ids = rebuild_database(products_df)

    # Rebuild models
    cf, cb, hybrid = rebuild_models()

    print("\n" + "="*55)
    print("  Integration Complete!")
    print("="*55)
    print("\n  Real Flipkart products loaded!")
    print("  Amazon & Flipkart links now search real products!")
    print("\n  Test recommendations:")

    sample_users = list(cf.user2idx.keys())[:2]
    for uid in sample_users:
        recs = hybrid.recommend(uid, top_n=3)
        print(f"\n  User {uid}:")
        conn = sqlite3.connect(DB_PATH)
        for pid, score in recs:
            row = conn.execute(
                "SELECT name, price FROM products WHERE product_id=?", (pid,)
            ).fetchone()
            if row:
                print(f"    ₹{row[1]:>8.0f}  {row[0][:50]}")
        conn.close()

    print("\n  Run: streamlit run app.py")
    print("  Then push to GitHub to update live app!\n")
