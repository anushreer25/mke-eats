
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast 
from datetime import datetime
import re
from collections import Counter
import helpers
import ast
import os
import utils.retriever as retriever
import logging
from utils.agent import create_agent
from utils.retriever import create_retriever
from utils.config import Config
from utils.state import GraphState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




@st.cache_data
def load_data():
    df_posts = pd.read_csv('milwaukee_restaurants_posts.csv')
    df_comments = pd.read_csv('milwaukee_restaurants_comments.csv')
    df_reviews = pd.read_csv('google_reviews_long_format.csv')
    
    df_posts['Restaurants_Mentioned'] = df_posts['Restaurants_Mentioned'].apply(helpers.safe_fix_list)
    df_comments['Restaurants_Mentioned'] = df_comments['Restaurants_Mentioned'].apply(helpers.safe_fix_list)
    
    return df_posts, df_comments, df_reviews

df_posts, df_comments, df_reviews = load_data()


st.title("The Best Milwaukee Restaurant Tool Ever")




# bm25 woohoo
st.header("Ask a question here")

user_query = st.text_input(" ",placeholder="e.g., good dinner places, restaurants with outdoor seating")

if user_query:
    with st.spinner("searching..."):
        
        config = Config()
        retriever = create_retriever()
        agent = create_agent()
        
        initial_state = GraphState(
            question=user_query,
            context=[],
            current_step="",
            final_answer="",
            retriever=retriever,
            web_search_tool=None,
            error=None,
            selected_namespaces=[],
            web_search_results=[]
        )
        
        result = agent.invoke(initial_state)
        
        if result.get("final_answer"):
            st.success("**Answer:**")
            st.write(result["final_answer"])

        if result.get("context"):
            st.subheader("Comments from the internet")
            
            # show 3 and then see more button maybe
            for i, ctx in enumerate(result["context"][:3], 1):
                snippet = ctx[:600] + "..." if len(ctx) > 600 else ctx
                st.markdown(f"> {snippet}")
                st.markdown("---")
            if len(result["context"]) > 5:
                if st.button("See more"):
                    for i, ctx in enumerate(result["context"][3:], 4):
                        snippet = ctx[:600] + "..." if len(ctx) > 600 else ctx
                        st.markdown(f"> {snippet}")
                        st.markdown("---")

        if result.get("error"):
            st.error(f"Error: {result['error']}")


st.header("Trending Restaurants📈 ")

if not df_reviews.empty:
    unique_restaurants = df_reviews['restaurant'].unique()[:20]
    
    # trend direction for each restaurant
    trend_data = []
    for restaurant in unique_restaurants:
        direction, change = helpers.calculate_trend_direction(df_reviews, restaurant)
        if direction == "up":
            emoji = "🟩"
        elif direction == "down":
            emoji = "🟥"
        else:
            emoji = "⬜"
        
        avg_sentiment = df_reviews[df_reviews['restaurant'] == restaurant]['sentiment_score'].mean()#idk if youll need this
        
        trend_data.append({
            'Restaurant': restaurant,
            'Trend': emoji,
            'Sentiment': f"{avg_sentiment:.2f}",
            'Change': f"{change:+.0f}%" if direction != "neutral" else "Stable"
        })
    
    trend_df = pd.DataFrame(trend_data)
    st.dataframe(trend_df, use_container_width=True)
else:
    st.info("No trending restaurants found")

########################################################################

st.header("Explore a Restaurant")

restaurant_list = df_reviews['restaurant'].explode().unique()
selected_restaurant = st.selectbox("Select a restaurant:", sorted(restaurant_list))

if selected_restaurant:
    st.subheader(f"Recent mentions of {selected_restaurant}")
    
    # Reddit POST mentions 
    recent_posts = df_posts[
        df_posts['Restaurants_Mentioned'].astype(str).str.contains(selected_restaurant, na=False)
    ].sort_values('Created_UTC', ascending=False).head(5)
    
    # Reddit COMMENT mentions 
    recent_comments = df_comments[
        df_comments['Restaurants_Mentioned'].astype(str).str.contains(selected_restaurant, na=False)
    ].sort_values('Created_UTC', ascending=False).head(5)
    
    # Google reviews
    recent_reviews = df_reviews[
        df_reviews['restaurant'] == selected_restaurant
    ].sort_values('date', ascending=False).head(5)
    
    for _, post in recent_posts.iterrows():
        body_text = post['Body'] if isinstance(post['Body'], str) else "No content"
        body_preview = body_text[:1000] + "..." if len(body_text) > 1000 else body_text
        
        with st.expander(f"📝 {post['Title']}"):
            st.write(body_preview)
            st.caption(f"Source: Reddit Post | Posted: {post['Created_UTC']}")
    
    for _, comment in recent_comments.iterrows():
        comment_text = comment['Comment_Body'] if isinstance(comment['Comment_Body'], str) else "No content"
        comment_preview = comment_text[:1000] + "..." if len(comment_text) > 1000 else comment_text
        
        with st.expander(f"💬 Comment"):
            st.write(comment_preview)
            st.caption(f"Source: Reddit Comment | Posted: {comment['Created_UTC']}")
    
    for _, review in recent_reviews.iterrows():
        review_text = review['text'] if isinstance(review['text'], str) else "No content"
        review_preview = review_text[:1000] + "..." if len(review_text) > 1000 else review_text
        
        with st.expander(f"⭐ Google Review"):
            st.write(review_preview)
            st.caption(f"Source: Google Review | Posted: {review['date']}")


all_text = []
all_text += (df_reviews["restaurant"] + " | " + df_reviews["text"]).tolist()
all_text += (df_posts["Restaurants_Mentioned"] + " | " + df_posts["Title"] + " | " + df_posts["Body"]).tolist()
all_text += (df_comments["Restaurants_Mentioned"] + " | " + df_comments["Comment_Body"]).tolist()

output_path = os.path.join("data", "combined_corpus.txt")

with open(output_path, "w", encoding="utf-8") as f:
    for line in all_text:
        f.write(str(line).replace("\n", " ") + "\n")

###########################

#main here
 
#nvm