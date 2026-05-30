import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
DB_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../database/smartcart.db')
class ContentBasedFilter:
    def __init__(self):
        self.products_df=None;self.tfidf_matrix=None;self.vectorizer=None;self.product_ids=None
    def load_products(self):
        conn=sqlite3.connect(DB_PATH)
        q='SELECT p.product_id,p.name,p.description,p.price,p.avg_rating,c.name AS category,GROUP_CONCAT(pt.tag,chr(32)) AS tags FROM products p JOIN categories c ON p.category_id=c.category_id LEFT JOIN product_tags pt ON p.product_id=pt.product_id GROUP BY p.product_id'
        self.products_df=pd.read_sql_query(q,conn);conn.close()
        self.products_df['tags']=self.products_df['tags'].fillna('')
        self.products_df['description']=self.products_df['description'].fillna('')
        print(f'Loaded {len(self.products_df)} products')
    def build_tfidf(self):
        self.products_df['corpus']=(self.products_df['name']+' '+self.products_df['category']+' '+self.products_df['tags']+' '+self.products_df['description'])
        self.vectorizer=TfidfVectorizer(stop_words='english',max_features=500,ngram_range=(1,2))
        self.tfidf_matrix=self.vectorizer.fit_transform(self.products_df['corpus'])
        self.product_ids=self.products_df['product_id'].tolist()
        print(f'TF-IDF: {self.tfidf_matrix.shape}')
    def get_similar_products(self,product_id,top_n=10):
        if product_id not in self.product_ids:return []
        idx=self.product_ids.index(product_id)
        scores=cosine_similarity(self.tfidf_matrix[idx],self.tfidf_matrix).flatten()
        scores[idx]=0;top_idxs=np.argsort(scores)[::-1][:top_n]
        return [(self.product_ids[i],round(float(scores[i]),4)) for i in top_idxs]
    def recommend_for_user(self,user_id,interaction_df,top_n=10):
        user_items=interaction_df[interaction_df['user_id']==user_id]
        if user_items.empty:return []
        top_items=user_items.sort_values('score',ascending=False).head(5)['product_id'].tolist()
        scores=np.zeros(len(self.product_ids));seen=set(user_items['product_id'])
        for pid in top_items:
            if pid not in self.product_ids:continue
            idx=self.product_ids.index(pid)
            scores+=cosine_similarity(self.tfidf_matrix[idx],self.tfidf_matrix).flatten()
        for pid in seen:
            if pid in self.product_ids:scores[self.product_ids.index(pid)]=0
        top_idxs=np.argsort(scores)[::-1][:top_n]
        return [(self.product_ids[i],round(float(scores[i]),4)) for i in top_idxs if scores[i]>0]
    def search(self,query,top_n=10):
        qv=self.vectorizer.transform([query]);scores=cosine_similarity(qv,self.tfidf_matrix).flatten()
        top_idxs=np.argsort(scores)[::-1][:top_n]
        return [{'product_id':self.products_df.iloc[i]['product_id'],'name':self.products_df.iloc[i]['name'],'score':round(float(scores[i]),4)} for i in top_idxs if scores[i]>0]
    def build(self):
        self.load_products();self.build_tfidf()
        with open('models/content_model.pkl','wb') as f:pickle.dump(self,f)
        print('Content model saved')
