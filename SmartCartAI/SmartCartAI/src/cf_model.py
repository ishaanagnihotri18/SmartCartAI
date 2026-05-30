import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
MATRIX_PATH='data/processed/matrix_data.pkl'
class CollaborativeFilter:
    def __init__(self):
        self.sparse_mat=None;self.user2idx={};self.idx2user={};self.item2idx={};self.idx2item={};self.user_sim=None;self.item_sim=None
    def load(self,path=MATRIX_PATH):
        with open(path,'rb') as f:data=pickle.load(f)
        self.sparse_mat=data['sparse'];self.user2idx=data['user2idx'];self.idx2user=data['idx2user'];self.item2idx=data['item2idx'];self.idx2item=data['idx2item']
        print(f'Loaded: {self.sparse_mat.shape}')
    def compute_user_similarity(self):
        self.user_sim=cosine_similarity(self.sparse_mat.toarray());print(f'User sim: {self.user_sim.shape}')
    def compute_item_similarity(self):
        self.item_sim=cosine_similarity(self.sparse_mat.toarray().T);print(f'Item sim: {self.item_sim.shape}')
    def recommend_hybrid(self,user_id,top_n=10,uw=0.5,iw=0.5):
        if user_id not in self.user2idx:return self.fallback_popular(top_n)
        idx=self.user2idx[user_id];user_scores=self.user_sim[idx];matrix_array=self.sparse_mat.toarray();user_vector=matrix_array[idx]
        weighted=user_scores@matrix_array;weighted[user_vector>0]=0
        scores2=np.zeros(self.item_sim.shape[0])
        for i in np.where(user_vector>0)[0]:scores2+=self.item_sim[i]*user_vector[i]
        scores2[user_vector>0]=0
        h={self.idx2item[i]:round(uw*float(weighted[i])+iw*float(scores2[i]),4) for i in range(len(weighted))}
        return sorted(h.items(),key=lambda x:x[1],reverse=True)[:top_n]
    def fallback_popular(self,top_n=10):
        col_sums=self.sparse_mat.toarray().sum(axis=0);top_idxs=np.argsort(col_sums)[::-1][:top_n]
        return [(self.idx2item[i],round(float(col_sums[i]),4)) for i in top_idxs]
    def evaluate(self,test_path='data/processed/test.csv',k=10):
        test_df=pd.read_csv(test_path);test_dict=test_df.groupby('user_id')['product_id'].apply(set).to_dict()
        precisions,hits=[],[]
        for uid,actual in test_dict.items():
            if uid not in self.user2idx:continue
            recs=[pid for pid,_ in self.recommend_hybrid(uid,top_n=k)]
            precisions.append(len(set(recs)&actual)/k);hits.append(1 if len(set(recs)&actual)>0 else 0)
        print(f'Precision@{k}: {np.mean(precisions):.4f} | HitRate@{k}: {np.mean(hits):.4f}')
    def build(self):
        self.load();self.compute_user_similarity();self.compute_item_similarity()
        os.makedirs('models',exist_ok=True)
        with open('models/cf_model.pkl','wb') as f:pickle.dump(self,f)
        print('CF model saved');self.evaluate()
