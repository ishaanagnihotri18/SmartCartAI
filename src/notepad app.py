import sys
import os
import pickle
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCartAI",
    page_icon="🛒",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/cf_model.pkl", "rb") as f:
        cf = pickle.load(f)
    with open("models/content_model.pkl", "rb") as f:
        cb = pickle.load(f)
    with open("models/hybrid_model.pkl", "rb") as f:
        hybrid = pickle.load(f)
    return cf, cb, hybrid

@st.cache_data
def load_data():
    conn = sqlite3.connect("database/smartcart.db")
    users    = pd.read_sql_query("SELECT * FROM users", conn)
    products = pd.read_sql_query("""
        SELECT p.*, c.name as category_name
        FROM products p
        JOIN categories c ON p.category_id = c.category_id
    """, conn)
    interactions = pd.read_sql_query("SELECT * FROM interactions", conn)
    conn.close()
    return users, products, interactions

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_product_info(product_id, products_df):
    row = products_df[products_df["product_id"] == product_id]
    if row.empty:
        return None
    return row.iloc[0]

def render_product_card(product, score=None):
    with st.container():
        st.markdown(f"""
        <div style="background:#1e1e1e;border-radius:10px;padding:15px;
                    border:1px solid #333;margin-bottom:10px">
            <div style="font-size:13px;color:#888">{product['category_name']}</div>
            <div style="font-weight:600;font-size:15px;margin:4px 0">{product['name'][:45]}</div>
            <div style="color:#1DB954;font-size:18px;font-weight:700">₹{product['price']:,.0f}</div>
            <div style="color:#FFD700">{'★' * int(product['avg_rating'])}
                <span style="color:#888;font-size:12px"> {product['avg_rating']}</span>
            </div>
            {f'<div style="color:#4CAF50;font-size:12px">Score: {score:.3f}</div>' if score else ''}
        </div>
        """, unsafe_allow_html=True)

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <h1 style="text-align:center;color:#1DB954">
        🛒 SmartCartAI
    </h1>
    <p style="text-align:center;color:#888;margin-top:-10px">
        Personalized Product Recommendation Engine
    </p>
    """, unsafe_allow_html=True)

    # Load everything
    with st.spinner("Loading models..."):
        cf, cb, hybrid = load_models()
        users, products, interactions = load_data()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("Controls")
    page = st.sidebar.radio("Navigate", [
        "Recommendations",
        "Similar Products",
        "Search",
        "Analytics"
    ])

    # ── Page 1: Recommendations ───────────────────────────────────────────────
    if page == "Recommendations":
        st.subheader("Personalized Recommendations")

        user_ids = list(cf.user2idx.keys())
        selected_user = st.selectbox(
            "Select User",
            user_ids,
            format_func=lambda x: f"User {x}"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.slider("Number of recommendations", 3, 20, 10)
            algo  = st.radio("Algorithm", ["Hybrid", "Collaborative Filter", "Content Based"])

        with col2:
            user_info = users[users["user_id"] == selected_user]
            if not user_info.empty:
                u = user_info.iloc[0]
                st.markdown(f"""
                **User:** {u['username']} &nbsp;|&nbsp;
                **Age:** {u['age_group']} &nbsp;|&nbsp;
                **Location:** {u['location']}
                """)

        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                if algo == "Hybrid":
                    recs = hybrid.recommend(selected_user, top_n=top_n)
                elif algo == "Collaborative Filter":
                    recs = cf.recommend_hybrid(selected_user, top_n=top_n)
                else:
                    train_df = pd.read_csv("data/processed/train.csv")
                    recs = cb.recommend_for_user(selected_user, train_df, top_n=top_n)

            if not recs:
                st.warning("No recommendations found for this user.")
            else:
                st.success(f"Top {len(recs)} recommendations for User {selected_user}")
                cols = st.columns(3)
                for i, (pid, score) in enumerate(recs):
                    product = get_product_info(pid, products)
                    if product is not None:
                        with cols[i % 3]:
                            render_product_card(product, score)

    # ── Page 2: Similar Products ──────────────────────────────────────────────
    elif page == "Similar Products":
        st.subheader("Find Similar Products")

        product_options = products[["product_id","name"]].values.tolist()
        selected = st.selectbox(
            "Select a product",
            product_options,
            format_func=lambda x: x[1]
        )
        selected_pid = selected[0]

        top_n = st.slider("Number of similar products", 3, 15, 6)

        if st.button("Find Similar", type="primary"):
            product = get_product_info(selected_pid, products)
            if product is not None:
                st.markdown(f"**Selected:** {product['name']} — ₹{product['price']:,.0f}")

            similar = cb.get_similar_products(selected_pid, top_n=top_n)

            if not similar:
                st.warning("No similar products found.")
            else:
                st.success(f"Top {len(similar)} similar products")
                cols = st.columns(3)
                for i, (pid, score) in enumerate(similar):
                    product = get_product_info(pid, products)
                    if product is not None:
                        with cols[i % 3]:
                            render_product_card(product, score)

    # ── Page 3: Search ────────────────────────────────────────────────────────
    elif page == "Search":
        st.subheader("Search Products")

        query = st.text_input("Search for products", placeholder="e.g. bluetooth laptop, yoga mat...")
        top_n = st.slider("Results", 3, 15, 6)

        if st.button("Search", type="primary") and query:
            results = cb.search(query, top_n=top_n)
            if not results:
                st.warning("No results found.")
            else:
                st.success(f"{len(results)} results for '{query}'")
                cols = st.columns(3)
                for i, r in enumerate(results):
                    product = get_product_info(r["product_id"], products)
                    if product is not None:
                        with cols[i % 3]:
                            render_product_card(product, r["score"])

    # ── Page 4: Analytics ─────────────────────────────────────────────────────
    elif page == "Analytics":
        st.subheader("Dataset Analytics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users",    f"{len(users):,}")
        col2.metric("Total Products", f"{len(products):,}")
        col3.metric("Interactions",   f"{len(interactions):,}")
        col4.metric("Categories",     f"{products['category_name'].nunique()}")

        col1, col2 = st.columns(2)

        with col1:
            event_counts = interactions["event_type"].value_counts().reset_index()
            event_counts.columns = ["event_type", "count"]
            fig = px.bar(event_counts, x="event_type", y="count",
                         title="Interactions by Event Type",
                         color="event_type")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cat_counts = products["category_name"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.pie(cat_counts, names="category", values="count",
                         title="Products by Category")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(products, x="avg_rating",
                               title="Product Rating Distribution",
                               nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(products, x="price",
                               title="Product Price Distribution",
                               nbins=30)
            st.plotly_chart(fig, use_container_width=True)

        # Evaluation metrics
        st.subheader("Model Evaluation")
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating..."):
                cf.evaluate()
            st.success("Evaluation complete! Check terminal for results.")

if __name__ == "__main__":
    main()