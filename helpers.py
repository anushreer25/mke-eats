# %%
# Milwaukee Restaurant Reddit Data Collection

import requests
import requests.auth
import praw
import pandas as pd
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

from datetime import datetime, timedelta
import re
from collections import Counter


# %%
def detect_trending_restaurants(df, time_window='30D', min_mentions=3):#base this on mentions and sentiment

    df_copy = df.copy()
    
    if 'Created_UTC' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['Created_UTC'])
        df_copy['date'] = df_copy['date'].dt.tz_localize(None)
    elif 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        if df_copy['date'].dt.tz is not None:
            df_copy['date'] = df_copy['date'].dt.tz_localize(None)
    
    df_time = df_copy.set_index('date')
    
    recent_cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(time_window)
    recent_df = df_time[df_time.index >= recent_cutoff].copy()
    trending_data = []
    
    if 'Restaurants_Mentioned' in df.columns:
        #exploded = recent_df.explode('Restaurants_Mentioned') dont do thisanymore, bc cols are no longer lists
        exploded = recent_df.copy()

        restaurant_stats = exploded.groupby('Restaurants_Mentioned').agg({
            'sentiment_score': ['mean', 'count'],
            'sentiment': lambda x: (x == 'positive').sum() / len(x) * 100 if len(x) > 0 else 0
        }).round(2)
        
        restaurant_stats.columns = ['avg_sentiment', 'mention_count', 'positive_percentage']
        restaurant_stats = restaurant_stats[restaurant_stats['mention_count'] >= min_mentions]
        
        # weigh the score  by mentions and sentiment
        restaurant_stats['trend_score'] = (
            restaurant_stats['mention_count'] * 
            restaurant_stats['avg_sentiment'] * 
            (1 + restaurant_stats['positive_percentage'] / 100)
        )
        
        trending_data = restaurant_stats.sort_values('trend_score', ascending=False)
    
    elif 'restaurant' in df.columns:
        restaurant_stats = recent_df.groupby('restaurant').agg({
            'sentiment_score': ['mean', 'count'],
            'sentiment': lambda x: (x == 'positive').sum() / len(x) * 100 if len(x) > 0 else 0
        }).round(2)
        
        restaurant_stats.columns = ['avg_sentiment', 'mention_count', 'positive_percentage']
        restaurant_stats = restaurant_stats[restaurant_stats['mention_count'] >= min_mentions]
        
        restaurant_stats['trend_score'] = (
            restaurant_stats['mention_count'] * 
            restaurant_stats['avg_sentiment'] * 
            (1 + restaurant_stats['positive_percentage'] / 100)
        )
        
        trending_data = restaurant_stats.sort_values('trend_score', ascending=False)
    
    return trending_data




# %%
def extract_topics(texts, top_n=10):

    all_text = ' '.join([str(t) for t in texts if isinstance(t, str)])
    
    stop_words = set([
        'the', 'and', 'for', 'that', 'this', 'with', 'was', 'were', 'are',
        'have', 'has', 'had', 'but', 'not', 'they', 'their', 'there',
        'from', 'which', 'like', 'just', 'very', 'really', 'also',
        'about', 'when', 'where', 'what', 'how', 'why', 'then', 'than',
        'been', 'being', 'will', 'would', 'could', 'should', 'might',
        'much', 'many', 'some', 'any', 'more', 'most', 'less', 'least',
        'you', 'your', 'our', 'www', 'com', 'org', 'http', 'https', 
        'case', 'county', 'project', 'charleyproject'
    ])
    
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    word_counts = Counter([w for w in words if w not in stop_words])
    topics = word_counts.most_common(top_n)
    
    return topics

# %%
def calculate_trend_direction(df_reviews, restaurant_name):
    restaurant_reviews = df_reviews[df_reviews['restaurant'] == restaurant_name].copy()
    if len(restaurant_reviews) < 2:
        return "neutral", 0
    
    restaurant_reviews = restaurant_reviews.sort_values('date', ascending=False)
    
    # try w split at recent 60%, rounded up
    import math
    total_reviews = len(restaurant_reviews)
    recent_count = math.ceil(total_reviews * 0.6)
    
    recent_reviews = restaurant_reviews.head(recent_count)
    older_reviews = restaurant_reviews.tail(total_reviews - recent_count)
    
    recent_sentiment = recent_reviews['sentiment_score'].mean()
    older_sentiment=older_reviews['sentiment_score'].mean()
    
    if len(older_reviews) == 0:
        if recent_sentiment > older_sentiment:
            return "up", 0
        elif recent_sentiment < older_sentiment:
            return "down", 0
        else:
            return "neutral", 0
    
    
    sentiment_change = recent_sentiment - older_sentiment
    
    if sentiment_change > 0.02:  #went up
        return "up", sentiment_change * 100
    elif sentiment_change < -0.02:  # down
        return "down", sentiment_change * 100
    else:
        return "neutral", sentiment_change * 100
    

def safe_fix_list(x):
    import ast
    if isinstance(x, list):
        return " ".join(x)
    
    if isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return " ".join(parsed)
        except:
            pass
    
    return str(x)
