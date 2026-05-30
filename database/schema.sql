CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    parent_id     INTEGER REFERENCES categories(category_id)
);

CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    email         TEXT    NOT NULL UNIQUE,
    age_group     TEXT    CHECK(age_group IN ('18-24','25-34','35-44','45-54','55+')),
    gender        TEXT    CHECK(gender IN ('M','F','Other','Prefer not to say')),
    location      TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    description   TEXT,
    price         REAL    NOT NULL CHECK(price >= 0),
    category_id   INTEGER REFERENCES categories(category_id),
    avg_rating    REAL    DEFAULT 0.0 CHECK(avg_rating BETWEEN 0 AND 5),
    review_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS product_tags (
    tag_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id    INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    tag           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(user_id),
    product_id     INTEGER NOT NULL REFERENCES products(product_id),
    event_type     TEXT    NOT NULL CHECK(event_type IN ('view','click','add_to_cart','purchase','rating')),
    rating         REAL    CHECK(rating BETWEEN 1 AND 5),
    session_id     INTEGER,
    event_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recommendations (
    rec_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(user_id),
    product_id     INTEGER NOT NULL REFERENCES products(product_id),
    algo_used      TEXT    NOT NULL CHECK(algo_used IN ('CF','content','hybrid','popular','cold_start')),
    score          REAL    NOT NULL,
    was_clicked    INTEGER DEFAULT 0 CHECK(was_clicked IN (0,1)),
    generated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_interactions_user    ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_product ON interactions(product_id);
CREATE INDEX IF NOT EXISTS idx_interactions_event   ON interactions(event_type);
CREATE INDEX IF NOT EXISTS idx_products_category    ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_recs_user            ON recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_tags_product         ON product_tags(product_id);