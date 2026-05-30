import sys
import os
import pickle
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="SmartCartAI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0a0f;
    color: #e8e6e0;
}
#MainMenu, footer, header { visibility: hidden; }

.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid #1e1e2e;
    min-width: 250px !important;
}

button[data-testid="collapsedControl"] {
    background: #1D9E75 !important;
    border-radius: 0 8px 8px 0 !important;
    color: white !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 2px; }

div[data-testid="stSelectbox"] > div > div {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
    color: #e8e6e0 !important;
}
div[data-testid="stSlider"] > div > div > div { background: #1D9E75 !important; }
div[data-testid="stRadio"] label { color: #e8e6e0 !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1D9E75, #0F6E56) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(29,158,117,0.3) !important;
}

div[data-testid="stTextInput"] input {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
    color: #e8e6e0 !important;
    padding: 10px 14px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #1D9E75 !important;
    box-shadow: 0 0 0 2px rgba(29,158,117,0.15) !important;
}

div[data-testid="stMetric"] {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="stMetric"] label {
    color: #666680 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1D9E75 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Clickable card button style */
div[data-testid="stButton"].card-btn > button {
    background: transparent !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 14px !important;
    padding: 0 !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"].card-btn > button:hover {
    border-color: #1D9E75 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(29,158,117,0.15) !important;
}

.back-btn > button {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    color: #e8e6e0 !important;
    width: auto !important;
    padding: 8px 20px !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────────
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
    tags = pd.read_sql_query("SELECT * FROM product_tags", conn)
    conn.close()
    return users, products, interactions, tags

def get_product(product_id, products_df):
    row = products_df[products_df["product_id"] == product_id]
    return row.iloc[0] if not row.empty else None


# ── Category colors ────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Electronics": "#185FA5", "Clothing": "#993C1D",
    "Books": "#534AB7",       "HomeKitchen": "#854F0B",
    "Sports": "#0F6E56",      "Home Kitchen": "#854F0B",
    "SelfHelp": "#534AB7",    "NonFiction": "#534AB7",
    "Fiction": "#534AB7",     "Science": "#534AB7",
    "Laptops": "#185FA5",     "Smartphones": "#185FA5",
    "Fitness": "#0F6E56",     "Yoga": "#0F6E56",
}

def cat_color(cat):
    return CAT_COLORS.get(str(cat), "#444460")


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:24px 20px 20px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <div style="width:32px;height:32px;background:linear-gradient(135deg,#1D9E75,#0F6E56);
                            border-radius:8px;display:flex;align-items:center;justify-content:center;
                            font-size:16px">🛒</div>
                <span style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                             color:#e8e6e0;letter-spacing:-0.02em">SmartCartAI</span>
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                        letter-spacing:0.1em;text-transform:uppercase;margin-left:42px">
                Recommendation Engine
            </div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,transparent,#1e1e2e,transparent);
                    margin:0 20px 20px"></div>
        <div style="padding:0 12px 8px;font-family:'DM Mono',monospace;font-size:10px;
                    color:#444460;letter-spacing:0.1em;text-transform:uppercase">Navigate</div>
        """, unsafe_allow_html=True)

        page = st.radio("nav", [
            "🎯  Recommendations",
            "🔗  Similar Products",
            "🔍  Search",
            "📊  Analytics"
        ], label_visibility="collapsed")

        st.markdown("""
        <div style="height:1px;background:linear-gradient(90deg,transparent,#1e1e2e,transparent);
                    margin:20px 20px"></div>
        <div style="padding:0 12px 10px;font-family:'DM Mono',monospace;font-size:10px;
                    color:#444460;letter-spacing:0.1em;text-transform:uppercase">System</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background:#13131f;border:1px solid #1e1e2e;border-radius:10px;
                        padding:10px 12px;margin:0 4px">
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                            text-transform:uppercase">Users</div>
                <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                            color:#1D9E75;margin-top:2px">500</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background:#13131f;border:1px solid #1e1e2e;border-radius:10px;
                        padding:10px 12px;margin:0 4px">
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                            text-transform:uppercase">Products</div>
                <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                            color:#1D9E75;margin-top:2px">200</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:16px 16px 0;font-family:'DM Sans',sans-serif;font-size:11px;
                    color:#333350;line-height:1.5">
            Collaborative filtering + TF-IDF content similarity
        </div>""", unsafe_allow_html=True)

    return page


# ── Page header ────────────────────────────────────────────────────────────────
def page_header(title, subtitle):
    st.markdown(f"""
    <div style="padding:32px 32px 0">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px">SmartCartAI</div>
        <h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                   color:#e8e6e0;margin:0 0 6px;letter-spacing:-0.02em">{title}</h1>
        <p style="font-family:'DM Sans',sans-serif;font-size:14px;color:#666680;margin:0 0 24px">{subtitle}</p>
        <div style="height:1px;background:linear-gradient(90deg,#1e1e2e,transparent)"></div>
    </div>
    """, unsafe_allow_html=True)


# ── Product card (clickable) ───────────────────────────────────────────────────
def render_product_card(product, score=None, rank=None, key_prefix="card"):
    color  = cat_color(product.get("category_name",""))
    stars  = "★" * int(float(product.get("avg_rating",0))) + "☆" * (5 - int(float(product.get("avg_rating",0))))
    pid    = int(product.get("product_id",0))
    rank_badge = f'<div style="position:absolute;top:10px;left:10px;width:22px;height:22px;background:#1D9E75;border-radius:6px;display:flex;align-items:center;justify-content:center;font-family:\'DM Mono\',monospace;font-size:10px;color:#fff">{rank}</div>' if rank else ''
    score_tag  = f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#1D9E75;margin-top:6px">score {score:.3f}</div>' if score else ''

    # HTML card
    st.markdown(f"""
    <div style="position:relative;background:#13131f;border:1px solid #1e1e2e;
                border-radius:14px;padding:16px;border-top:2px solid {color}30;
                margin-bottom:4px">
        {rank_badge}
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:{color};
                    letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;
                    padding-left:{('28px' if rank else '0')}">
            {str(product.get('category_name',''))}
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
                    color:#e8e6e0;line-height:1.4;margin-bottom:10px;min-height:36px">
            {str(product.get('name',''))[:45]}
        </div>
        <div style="display:flex;align-items:baseline;justify-content:space-between">
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:#1D9E75">
                ₹{float(product.get('price',0)):,.0f}
            </div>
            <div style="font-size:12px;color:#666680">
                {stars} <span style="font-family:'DM Mono',monospace;font-size:10px">{float(product.get('avg_rating',0))}</span>
            </div>
        </div>
        {score_tag}
    </div>
    """, unsafe_allow_html=True)

    # Separate button below card
    if st.button(f"View Details →", key=f"{key_prefix}_{pid}"):
        st.session_state["view_product"] = int(pid)
        st.rerun()


# ── Product detail page ────────────────────────────────────────────────────────
def page_product_detail(products, tags, cf, cb):
    pid     = st.session_state["view_product"]
    product = get_product(pid, products)

    if product is None:
        st.error("Product not found.")
        return

    color   = cat_color(product.get("category_name",""))
    stars   = "★" * int(float(product.get("avg_rating",0))) + "☆" * (5 - int(float(product.get("avg_rating",0))))
    ptags   = tags[tags["product_id"] == pid]["tag"].tolist()
    amazon_query = str(product["name"]).replace(" ", "+")
    amazon_url   = f"https://www.amazon.in/s?k={amazon_query}"
    flip_url     = f"https://www.flipkart.com/search?q={amazon_query}"

    st.markdown('<div style="padding:24px 32px 0">', unsafe_allow_html=True)

    # Back button
    col_back, _ = st.columns([1, 5])
    with col_back:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back"):
            st.session_state["view_product"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Hero section
    st.markdown(f"""
    <div style="background:#0d0d14;border:1px solid #1e1e2e;border-radius:20px;
                padding:32px;margin:16px 0;border-top:3px solid {color}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:20px">
            <div style="flex:1;min-width:250px">
                <div style="font-family:'DM Mono',monospace;font-size:11px;color:{color};
                            letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px">
                    {product.get('category_name','')}
                </div>
                <h1 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
                           color:#e8e6e0;margin:0 0 12px;line-height:1.2">
                    {product['name']}
                </h1>
                <div style="font-size:13px;color:#666680;line-height:1.6;margin-bottom:16px">
                    {product.get('description','')}
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px">
                    {''.join([f'<span style="background:#1a1a28;border:1px solid #2a2a3e;border-radius:20px;padding:4px 10px;font-family:\'DM Mono\',monospace;font-size:10px;color:#666680">{t}</span>' for t in ptags])}
                </div>
                <div style="display:flex;gap:16px;align-items:center">
                    <a href="{amazon_url}" target="_blank"
                       style="background:#FF9900;color:#000;border-radius:8px;padding:8px 16px;
                              font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
                              text-decoration:none;display:inline-block">
                        Search on Amazon ↗
                    </a>
                    <a href="{flip_url}" target="_blank"
                       style="background:#2874F0;color:#fff;border-radius:8px;padding:8px 16px;
                              font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
                              text-decoration:none;display:inline-block">
                        Search on Flipkart ↗
                    </a>
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:800;
                            color:#1D9E75;line-height:1">
                    ₹{float(product['price']):,.0f}
                </div>
                <div style="font-size:18px;color:#FFD700;margin:8px 0">{stars}</div>
                <div style="font-family:'DM Mono',monospace;font-size:13px;color:#666680">
                    {float(product['avg_rating'])} / 5.0
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:11px;color:#444460;margin-top:4px">
                    {int(product.get('review_count',0)):,} reviews
                </div>
                <div style="margin-top:16px;background:#1a1a28;border-radius:10px;padding:12px 16px">
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                                text-transform:uppercase;margin-bottom:4px">Product ID</div>
                    <div style="font-family:'DM Mono',monospace;font-size:14px;color:#1D9E75">
                        #{pid}
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

   # Similar products
    similar = cb.get_similar_products(pid, top_n=6)
    if similar:
        cols = st.columns(3)
        for i, (spid, score) in enumerate(similar):
            p = get_product(spid, products)
            if p is not None:
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style="background:#13131f;border:1px solid #1e1e2e;
                                border-radius:14px;padding:16px;border-top:2px solid {cat_color(p.get('category_name',''))}30;
                                margin-bottom:4px">
                        <div style="font-family:'DM Mono',monospace;font-size:10px;
                                    color:{cat_color(p.get('category_name',''))};
                                    text-transform:uppercase;margin-bottom:6px">
                            {str(p.get('category_name',''))}
                        </div>
                        <div style="font-size:13px;font-weight:500;color:#e8e6e0;
                                    margin-bottom:10px">{str(p.get('name',''))[:45]}</div>
                        <div style="font-family:'Syne',sans-serif;font-size:20px;
                                    font-weight:700;color:#1D9E75">
                            ₹{float(p.get('price',0)):,.0f}
                        </div>
                        <div style="font-family:'DM Mono',monospace;font-size:10px;
                                    color:#1D9E75;margin-top:6px">score {score:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View Details →", key=f"sim_{pid}_{int(p['product_id'])}"):
                        st.session_state["view_product"] = int(p["product_id"])
                        st.rerun()

    # Frequently bought together
    st.markdown("""
    <div style="margin:32px 0 12px">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">
            Collaborative Filtering
        </div>
        <h2 style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
                   color:#e8e6e0;margin:0">Frequently Bought Together</h2>
    </div>
    """, unsafe_allow_html=True)

    cf_similar = cf.fallback_popular(top_n=3)
    cols = st.columns(3)
    for i, (spid, score) in enumerate(cf_similar):
        p = get_product(spid, products)
        if p is not None:
            with cols[i]:
                st.markdown(f"""
                <div style="background:#13131f;border:1px solid #1e1e2e;
                            border-radius:14px;padding:16px;border-top:2px solid {cat_color(p.get('category_name',''))}30;
                            margin-bottom:4px">
                    <div style="font-family:'DM Mono',monospace;font-size:10px;
                                color:{cat_color(p.get('category_name',''))};
                                text-transform:uppercase;margin-bottom:6px">
                        {str(p.get('category_name',''))}
                    </div>
                    <div style="font-size:13px;font-weight:500;color:#e8e6e0;
                                margin-bottom:10px">{str(p.get('name',''))[:45]}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:20px;
                                font-weight:700;color:#1D9E75">
                        ₹{float(p.get('price',0)):,.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View Details →", key=f"cf_{pid}_{int(p['product_id'])}"):
                    st.session_state["view_product"] = int(p["product_id"])
                    st.rerun()


# ── Recommendations page ───────────────────────────────────────────────────────
def page_recommendations(cf, cb, hybrid, users, products):
    page_header("Personalized Recommendations",
                "AI-powered product suggestions tailored to each user's behavior")

    st.markdown("""
    <div style="padding:24px 32px 0">
    <div style="background:#0d0d14;border:1px solid #1e1e2e;border-radius:16px;
                padding:24px;margin-bottom:24px">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px">
            Control Panel
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.markdown('<div style="font-size:12px;color:#666680;margin-bottom:6px">Select User</div>', unsafe_allow_html=True)
        user_ids      = list(cf.user2idx.keys())
        selected_user = st.selectbox("user", user_ids,
            format_func=lambda x: f"User {x}", label_visibility="collapsed")
    with col2:
        st.markdown('<div style="font-size:12px;color:#666680;margin-bottom:6px">Recommendations</div>', unsafe_allow_html=True)
        top_n = st.slider("topn", 3, 20, 10, label_visibility="collapsed")
    with col3:
        st.markdown('<div style="font-size:12px;color:#666680;margin-bottom:6px">Algorithm</div>', unsafe_allow_html=True)
        algo = st.radio("algo", ["Hybrid","CF Only","Content Only"], label_visibility="collapsed")

    user_info = users[users["user_id"] == selected_user]
    if not user_info.empty:
        u = user_info.iloc[0]
        st.markdown(f"""
        <div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap">
            <div style="background:#1a1a28;border-radius:8px;padding:8px 14px;display:flex;align-items:center;gap:8px">
                <div style="width:28px;height:28px;background:#1D9E75;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-family:'Syne',sans-serif;font-size:11px;font-weight:700;color:#fff">
                    {str(u['username'])[0].upper()}
                </div>
                <div>
                    <div style="font-size:13px;font-weight:500;color:#e8e6e0">{u['username']}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460">User #{selected_user}</div>
                </div>
            </div>
            <div style="background:#1a1a28;border-radius:8px;padding:8px 14px">
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;text-transform:uppercase">Age</div>
                <div style="font-size:13px;color:#e8e6e0;margin-top:1px">{u['age_group']}</div>
            </div>
            <div style="background:#1a1a28;border-radius:8px;padding:8px 14px">
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;text-transform:uppercase">Location</div>
                <div style="font-size:13px;color:#e8e6e0;margin-top:1px">{u['location']}</div>
            </div>
            <div style="background:#1a1a28;border-radius:8px;padding:8px 14px">
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;text-transform:uppercase">Gender</div>
                <div style="font-size:13px;color:#e8e6e0;margin-top:1px">{u['gender']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown('<div style="padding:0 32px">', unsafe_allow_html=True)

    if st.button("Generate Recommendations →"):
        with st.spinner("Analyzing preferences..."):
            train_df = pd.read_csv("data/processed/train.csv")
            if algo == "Hybrid":
                recs = hybrid.recommend(selected_user, top_n=top_n)
            elif algo == "CF Only":
                recs = cf.recommend_hybrid(selected_user, top_n=top_n)
            else:
                recs = cb.recommend_for_user(selected_user, train_df, top_n=top_n)
        st.session_state["last_recs"] = recs
        st.session_state["last_user"] = selected_user
        st.session_state["last_algo"] = algo

    recs = st.session_state.get("last_recs", [])
    if recs:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin:20px 0 16px">
            <div style="height:1px;flex:1;background:#1e1e2e"></div>
            <div style="font-family:'DM Mono',monospace;font-size:11px;color:#1D9E75;letter-spacing:0.1em">
                TOP {len(recs)} · {st.session_state.get('last_algo','').upper()}
            </div>
            <div style="height:1px;flex:1;background:#1e1e2e"></div>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (pid, score) in enumerate(recs):
            p = get_product(pid, products)
            if p is not None:
                with cols[i % 3]:
                    render_product_card(p, score, rank=i+1,
                                        key_prefix=f"rec_{st.session_state.get('last_user')}")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Similar products page ──────────────────────────────────────────────────────
def page_similar(cb, products):
    page_header("Similar Products",
                "Find products with similar descriptions, tags, and categories")

    st.markdown("""
    <div style="padding:24px 32px 0">
    <div style="background:#0d0d14;border:1px solid #1e1e2e;border-radius:16px;
                padding:24px;margin-bottom:24px">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px">
            Control Panel
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div style="font-size:12px;color:#666680;margin-bottom:6px">Select Product</div>', unsafe_allow_html=True)
        product_options = products[["product_id","name"]].values.tolist()
        selected = st.selectbox("prod", product_options,
            format_func=lambda x: x[1], label_visibility="collapsed")
    with col2:
        st.markdown('<div style="font-size:12px;color:#666680;margin-bottom:6px">Results</div>', unsafe_allow_html=True)
        top_n = st.slider("simn", 3, 15, 6, label_visibility="collapsed")

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown('<div style="padding:0 32px">', unsafe_allow_html=True)
    if st.button("Find Similar Products →"):
        pid     = selected[0]
        similar = cb.get_similar_products(pid, top_n=top_n)
        st.session_state["last_similar"]     = similar
        st.session_state["last_similar_pid"] = pid

    similar = st.session_state.get("last_similar", [])
    pid     = st.session_state.get("last_similar_pid", None)

    if similar and pid:
        p = get_product(pid, products)
        if p is not None:
            st.markdown(f"""
            <div style="background:#13131f;border:1px solid #1D9E75;border-radius:12px;
                        padding:16px 20px;margin-bottom:20px;display:flex;
                        align-items:center;justify-content:space-between">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#1D9E75;
                                text-transform:uppercase">Selected</div>
                    <div style="font-size:15px;font-weight:500;color:#e8e6e0;margin-top:4px">{p['name']}</div>
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#1D9E75">
                    ₹{float(p['price']):,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (spid, score) in enumerate(similar):
            sp = get_product(spid, products)
            if sp is not None:
                with cols[i % 3]:
                    render_product_card(sp, score, rank=i+1, key_prefix=f"sim_{pid}")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Search page ────────────────────────────────────────────────────────────────
def page_search(cb, products):
    page_header("Product Search",
                "Find products using natural language — powered by TF-IDF similarity")

    st.markdown("""
    <div style="padding:24px 32px 0">
    <div style="background:#0d0d14;border:1px solid #1e1e2e;border-radius:16px;
                padding:24px;margin-bottom:24px">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444460;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px">
            Search Panel
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4,1])
    with col1:
        st.markdown('<div style="font-size:12px;color:#666680;margin-bottom:6px">Search Query</div>', unsafe_allow_html=True)
        query = st.text_input("q",
            placeholder="e.g. bluetooth headphones, yoga mat, sci-fi novel...",
            label_visibility="collapsed")
    with col2:
        st.markdown('<div style="font-size:12px;color:#666680;margin-bottom:6px">Results</div>', unsafe_allow_html=True)
        top_n = st.slider("sn", 3, 15, 6, label_visibility="collapsed")

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown('<div style="padding:0 32px">', unsafe_allow_html=True)
    if st.button("Search →") and query:
        with st.spinner("Searching..."):
            results = cb.search(query, top_n=top_n)
        st.session_state["last_search_results"] = results
        st.session_state["last_query"]          = query

    results = st.session_state.get("last_search_results", [])
    query   = st.session_state.get("last_query", "")
    if results:
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:11px;color:#1D9E75;
                    margin-bottom:16px">{len(results)} results for "{query}"</div>
        """, unsafe_allow_html=True)
        cols = st.columns(3)
        for i, r in enumerate(results):
            p = get_product(r["product_id"], products)
            if p is not None:
                with cols[i % 3]:
                    render_product_card(p, r["score"], rank=i+1,
                                        key_prefix=f"srch_{query[:5]}")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Analytics page ─────────────────────────────────────────────────────────────
def page_analytics(interactions, products, users):
    page_header("Analytics", "Dataset insights and model performance metrics")

    st.markdown('<div style="padding:24px 32px">', unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("Total Users",    f"{len(users):,}")
    with col2: st.metric("Total Products", f"{len(products):,}")
    with col3: st.metric("Interactions",   f"{len(interactions):,}")
    with col4: st.metric("Categories",     f"{products['category_name'].nunique()}")

    st.markdown("<br>", unsafe_allow_html=True)

    bg = "#0d0d14"; gc = "#1e1e2e"; tc = "#666680"

    col1,col2 = st.columns(2)
    with col1:
        ec = interactions["event_type"].value_counts().reset_index()
        ec.columns = ["event_type","count"]
        fig = px.bar(ec, x="event_type", y="count",
                     color="count", color_continuous_scale=["#0F6E56","#1D9E75","#5DCAA5"])
        fig.update_layout(title=dict(text="Interaction Types",
            font=dict(family="Syne",size=14,color="#e8e6e0")),
            paper_bgcolor=bg, plot_bgcolor=bg,
            font=dict(family="DM Sans",color=tc),
            xaxis=dict(gridcolor=gc,color=tc),
            yaxis=dict(gridcolor=gc,color=tc),
            coloraxis_showscale=False,
            margin=dict(t=40,b=20,l=20,r=20), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cc = products["category_name"].value_counts().reset_index()
        cc.columns = ["category","count"]
        fig = px.pie(cc, names="category", values="count",
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(title=dict(text="Products by Category",
            font=dict(family="Syne",size=14,color="#e8e6e0")),
            paper_bgcolor=bg, font=dict(family="DM Sans",color=tc),
            margin=dict(t=40,b=20,l=20,r=20), height=280)
        st.plotly_chart(fig, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        fig = px.histogram(products, x="avg_rating", nbins=20,
                           color_discrete_sequence=["#1D9E75"])
        fig.update_layout(title=dict(text="Rating Distribution",
            font=dict(family="Syne",size=14,color="#e8e6e0")),
            paper_bgcolor=bg, plot_bgcolor=bg,
            font=dict(family="DM Sans",color=tc),
            xaxis=dict(gridcolor=gc,color=tc),
            yaxis=dict(gridcolor=gc,color=tc),
            margin=dict(t=40,b=20,l=20,r=20), height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(products, x="price", nbins=30,
                           color_discrete_sequence=["#5DCAA5"])
        fig.update_layout(title=dict(text="Price Distribution",
            font=dict(family="Syne",size=14,color="#e8e6e0")),
            paper_bgcolor=bg, plot_bgcolor=bg,
            font=dict(family="DM Sans",color=tc),
            xaxis=dict(gridcolor=gc,color=tc),
            yaxis=dict(gridcolor=gc,color=tc),
            margin=dict(t=40,b=20,l=20,r=20), height=260)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if "view_product" not in st.session_state:
        st.session_state["view_product"] = None

    with st.spinner("Loading models..."):
        cf, cb, hybrid = load_models()
        users, products, interactions, tags = load_data()

    # Check product detail FIRST before rendering sidebar
    if st.session_state.get("view_product") is not None:
        page_product_detail(products, tags, cf, cb)
        return

    page = render_sidebar()
    st.session_state["current_page"] = page

    if "Recommendations" in page:
        page_recommendations(cf, cb, hybrid, users, products)
    elif "Similar" in page:
        page_similar(cb, products)
    elif "Search" in page:
        page_search(cb, products)
    elif "Analytics" in page:
        page_analytics(interactions, products, users)

if __name__ == "__main__":
    main()
