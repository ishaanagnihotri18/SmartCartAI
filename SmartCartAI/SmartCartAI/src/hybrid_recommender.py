import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
class HybridRecommender:
    def __init__(self,cf_model,content_model,cf_weight=0.6,content_weight=0.4):
        self.cf=cf_model;self.cb=content_model;self.cf_w=cf_weight;self.cb_w=content_weight
        self.train_df=pd.read_csv('data/processed/train.csv')
    def recommend(self,user_id,top_n=10):
        cf_recs=dict(self.cf.recommend_hybrid(user_id,top_n=top_n*2))
        cb_recs=dict(self.cb.recommend_for_user(user_id,self.train_df,top_n=top_n*2))
        if not cf_recs and not cb_recs:return self.cf.fallback_popular(top_n)
        def norm(d):
            if not d:return d
            mx=max(d.values());return {k:v/mx for k,v in d.items()} if mx>0 else d
        cf_recs=norm(cf_recs);cb_recs=norm(cb_recs)
        all_items=set(cf_recs)|set(cb_recs)
        h={item:round(self.cf_w*cf_recs.get(item,0)+self.cb_w*cb_recs.get(item,0),4) for item in all_items}
        return sorted(h.items(),key=lambda x:x[1],reverse=True)[:top_n]
    def get_product_details(self,product_ids):
        conn=sqlite3.connect(os.path.join('database','smartcart.db'))
        ids_str=','.join(str(i) for i in product_ids)
        df=pd.read_sql_query(f'SELECT p.product_id,p.name,p.price,p.avg_rating,c.name AS category FROM products p JOIN categories c ON p.category_id=c.category_id WHERE p.product_id IN ({ids_str})',conn)
        conn.close();return df
    def recommend_with_details(self,user_id,top_n=10):
        recs=self.recommend(user_id,top_n)
        if not recs:return pd.DataFrame()
        pids=[pid for pid,_ in recs];scores={pid:s for pid,s in recs}
        details=self.get_product_details(pids)
        details['score']=details['product_id'].map(scores)
        return details.sort_values('score',ascending=False)
    def save(self):
        with open('models/hybrid_model.pkl','wb') as f:pickle.dump(self,f)
        print('Hybrid model saved')
