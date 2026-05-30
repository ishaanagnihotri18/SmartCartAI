import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from src.cf_model import CollaborativeFilter
from src.content_model import ContentBasedFilter
from src.hybrid_recommender import HybridRecommender
print('='*50)
print('SmartCartAI - Phase 2')
print('='*50)
cf=CollaborativeFilter()
cf.build()
cb=ContentBasedFilter()
cb.build()
hybrid=HybridRecommender(cf,cb)
hybrid.save()
print('Phase 2 Complete!')
