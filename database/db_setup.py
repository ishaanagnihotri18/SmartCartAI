import sqlite3
import os
import random
import uuid
from datetime import datetime, timedelta

DB_PATH     = os.path.join(os.path.dirname(__file__), "smartcart.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

NUM_USERS        = 500
NUM_PRODUCTS     = 200
NUM_INTERACTIONS = 8000

random.seed(42)

def random_date(days_back=365):
    return datetime.now() - timedelta(days=random.randint(0, days_back))

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def create_schema(conn):
    with open(SCHEMA_PATH, "r") as f:
        sql = f.read()
    conn.executescript(sql)
    conn.commit()
    print("✓ Schema created")

CATEGORY_TREE = {
    "Electronics":    ["Laptops", "Smartphones", "Headphones", "Cameras"],
    "Clothing":       ["Mens Wear", "Womens Wear", "Footwear", "Accessories"],
    "Books":          ["Fiction", "NonFiction", "Science", "SelfHelp"],
    "Home Kitchen":   ["Cookware", "Furniture", "Decor", "Appliances"],
    "Sports":         ["Fitness", "Outdoor", "TeamSports", "Yoga"],
}

def seed_categories(conn):
    cur = conn.cursor()
    cat_ids = {}
    for parent, children in CATEGORY_TREE.items():
        cur.execute("INSERT INTO categories (name) VALUES (?)", (parent,))
        parent_id = cur.lastrowid
        cat_ids[parent] = parent_id
        for child in children:
            cur.execute("INSERT INTO categories (name, parent_id) VALUES (?,?)", (child, parent_id))
            cat_ids[child] = cur.lastrowid
    conn.commit()
    print(f"✓ Seeded {len(cat_ids)} categories")
    return cat_ids

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS    = ["M", "F", "Other"]
CITIES     = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
               "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

FIRST_NAMES = ["Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Ayaan",
                "Krishna","Ishaan","Priya","Ananya","Pooja","Neha","Sneha","Riya",
                "Divya","Meera","Kavya","Shruti","Rahul","Rohit","Amit","Suresh",
                "Rajesh","Vijay","Prakash","Manoj","Deepak","Sunil"]

LAST_NAMES  = ["Sharma","Verma","Patel","Singh","Kumar","Gupta","Joshi","Mehta",
                "Shah","Rao","Nair","Iyer","Reddy","Pillai","Mishra","Tiwari",
                "Pandey","Dubey","Srivastava","Chauhan"]

def seed_users(conn):
    cur = conn.cursor()
    used_usernames = set()
    used_emails    = set()
    count = 0
    attempts = 0

    while count < NUM_USERS and attempts < NUM_USERS * 10:
        attempts += 1
        fname    = random.choice(FIRST_NAMES)
        lname    = random.choice(LAST_NAMES)
        num      = random.randint(1, 9999)
        username = f"{fname.lower()}{lname.lower()}{num}"
        email    = f"{fname.lower()}.{lname.lower()}{num}@{random.choice(['gmail.com','yahoo.com','outlook.com'])}"

        if username in used_usernames or email in used_emails:
            continue

        used_usernames.add(username)
        used_emails.add(email)

        created  = random_date(730)
        last_act = created + timedelta(days=random.randint(0, 365))

        cur.execute("""
            INSERT INTO users (username, email, age_group, gender, location, created_at, last_active)
            VALUES (?,?,?,?,?,?,?)
        """, (
            username,
            email,
            random.choice(AGE_GROUPS),
            random.choice(GENDERS),
            random.choice(CITIES),
            created.isoformat(),
            last_act.isoformat(),
        ))
        count += 1

    conn.commit()
    print(f"✓ Seeded {count} users")

PRODUCT_TEMPLATES = {
    "Laptops":      ("Laptop",     15000, 120000),
    "Smartphones":  ("Smartphone",  8000,  80000),
    "Headphones":   ("Headphones",   500,  20000),
    "Cameras":      ("Camera",      5000,  80000),
    "Mens Wear":    ("Shirt",        300,   3000),
    "Womens Wear":  ("Dress",        400,   5000),
    "Footwear":     ("Shoes",        500,   8000),
    "Accessories":  ("Watch",        500,  50000),
    "Fiction":      ("Novel",        150,    800),
    "NonFiction":   ("Book",         200,   1200),
    "Science":      ("Textbook",     300,   2000),
    "SelfHelp":     ("Guide",        200,    900),
    "Cookware":     ("Pan",          300,   5000),
    "Furniture":    ("Chair",       2000,  30000),
    "Decor":        ("Vase",         200,   3000),
    "Appliances":   ("Blender",      800,   8000),
    "Fitness":      ("Dumbbell",     500,  15000),
    "Outdoor":      ("Tent",        2000,  20000),
    "TeamSports":   ("Ball",         300,   3000),
    "Yoga":         ("Mat",          400,   3000),
}

ADJECTIVES = ["Pro","Ultra","Smart","Premium","Elite","Classic","Advanced","Compact","Deluxe","Lite"]
BRANDS     = ["TechNova","Zurix","Apex","Lumina","CoreBrand","Nimbus","Stellar","Vantage","Orbit","Zephyr"]
FEATURES   = ["with Bluetooth","with WiFi","Waterproof","Lightweight","Heavy Duty",
               "Eco Friendly","Fast Charging","Ergonomic","Portable","All Weather"]
USES       = ["for home use","for professionals","for students","for outdoor activities",
               "for daily use","for sports enthusiasts","for beginners","for experts"]

def seed_products(conn, cat_ids):
    cur = conn.cursor()
    leaf_cats   = [c for c in cat_ids if c not in CATEGORY_TREE]
    product_ids = []

    for i in range(NUM_PRODUCTS):
        cat_name              = random.choice(leaf_cats)
        keyword, pmin, pmax   = PRODUCT_TEMPLATES[cat_name]
        name   = f"{random.choice(BRANDS)} {random.choice(ADJECTIVES)} {keyword} {i+1}"
        price  = round(random.uniform(pmin, pmax), 2)
        rating = round(random.uniform(2.5, 5.0), 1)
        count  = random.randint(5, 2000)
        desc   = (f"{name} {random.choice(FEATURES)} {random.choice(USES)}. "
                  f"High quality {keyword.lower()} designed for performance and durability.")

        cur.execute("""
            INSERT INTO products (name, description, price, category_id, avg_rating, review_count, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (name, desc, price, cat_ids[cat_name], rating, count, random_date(400).isoformat()))

        pid = cur.lastrowid
        product_ids.append(pid)

        tags = [keyword.lower(), cat_name.lower()] + \
               random.sample(["quality","durable","popular","trending","bestseller","new","sale"], 3)
        for tag in set(tags):
            cur.execute("INSERT INTO product_tags (product_id, tag) VALUES (?,?)", (pid, tag))

    conn.commit()
    print(f"✓ Seeded {NUM_PRODUCTS} products with tags")
    return product_ids

EVENT_TYPES   = ["view","click","add_to_cart","purchase","rating"]
EVENT_WEIGHTS = [0.45,  0.25,  0.15,          0.10,      0.05]

def seed_interactions(conn, user_ids, product_ids):
    cur = conn.cursor()
    session_counter = 1
    for _ in range(NUM_INTERACTIONS):
        user_id    = random.choice(user_ids)
        product_id = random.choice(product_ids)
        event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
        rating     = round(random.uniform(1, 5), 1) if event_type == "rating" else None
        event_time = random_date(180).isoformat()
        cur.execute("""
            INSERT INTO interactions (user_id, product_id, event_type, rating, session_id, event_time)
            VALUES (?,?,?,?,?,?)
        """, (user_id, product_id, event_type, rating, session_counter, event_time))
        if random.random() < 0.05:
            session_counter += 1
    conn.commit()
    print(f"✓ Seeded {NUM_INTERACTIONS} interactions")

def verify(conn):
    tables = ["categories","users","products","product_tags","interactions","recommendations"]
    print("\n── Row counts ──────────────────────")
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<20} {count:>6} rows")
    print("────────────────────────────────────")
    print(f"\n✓ Database ready at: {DB_PATH}\n")

if __name__ == "__main__":
    print("\nSmartCartAI — Database Setup")
    print("=" * 40)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("! Removed existing database")
    conn = get_connection()
    create_schema(conn)
    cat_ids     = seed_categories(conn)
    seed_users(conn)
    user_ids    = [r[0] for r in conn.execute("SELECT user_id FROM users").fetchall()]
    product_ids = seed_products(conn, cat_ids)
    seed_interactions(conn, user_ids, product_ids)
    verify(conn)
    conn.close()