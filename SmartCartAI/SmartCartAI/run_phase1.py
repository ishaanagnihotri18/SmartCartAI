import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database.db_setup import get_connection, create_schema, seed_categories, seed_users, seed_products, seed_interactions, verify
from src.data_cleaner import DataCleaner
from src.matrix_builder import MatrixBuilder
import os
db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "smartcart.db")
if os.path.exists(db): os.remove(db)
conn = get_connection()
create_schema(conn)
cat_ids = seed_categories(conn)
seed_users(conn)
user_ids = [r[0] for r in conn.execute("SELECT user_id FROM users").fetchall()]
product_ids = seed_products(conn, cat_ids)
seed_interactions(conn, user_ids, product_ids)
verify(conn)
conn.close()
cleaner = DataCleaner()
df, train, test = cleaner.run(save_csv=True)
cleaner.close()
builder = MatrixBuilder()
matrix, sparse, maps = builder.build(save=True)
print("Phase 1 Complete!")
