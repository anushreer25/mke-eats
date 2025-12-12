# mke-eats
# Milwaukee Restaurant Discovery Dashboard

This web application helps users discover and explore Milwaukee restaurants. It pulls from a pool of 60 (greater than 3 star Google review) Milwaukee restaurants, each with Google review, Reddit comment, and Reddit post data. 

## Three main components:

1. **AI-Powered Search: Natural Language search** Ask natural language questions about Milwaukee restaurants using BM25 retrieval and local LLM.
2. **Trending Restaurants: Sentiment analysis** User can view top 20 trending restaurants as well as which ones are gaining/losing popularity based on sentiment trends.
3. **Restaurant Explorer** Browse recent mentions of specific restaurants from Reddit and Google Reviews.

## How to Use

### Prerequisites

- Python 3.11+
- Ollama (for LLM functionality)
- Git

### Installation

1. **Clone the repo**
```bash
git clone 
cd mke-eats
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and start Ollama**
- Pull the model from [ollama.ai](https://ollama.ai):
```bash
ollama pull llama3.2:1b
ollama serve
```

4. **Create the env file**
```
PYSERINI_CNAME=combined_corpus
OLLAMA_MODEL=llama3.2:1b
RETRIEVER_K=10
PYSERINI_K1=0.9
PYSERINI_B=0.4
```

5. **Prepare data**

Ensure these CSV files are in your project directory:
- `milwaukee_restaurants_posts.csv`
- `milwaukee_restaurants_comments.csv`
- `google_reviews_long_format.csv`

Ensure you are in a mnt directory within WSL (if using Linux) with pyserini installed. Then run
```bash
conda activate pyserini
```

6. **Build search index**

The index will be built automatically on first run, or manually:
```bash
python -c "from utils.retriever import create_retriever; create_retriever()"
```


### Running the Application
```bash
streamlit run app.py
```
The dashboard will open at `http://localhost:8501`

### Using the Dashboard

**Trending Section**
- View top trending restaurants with sentiment indicators:
  - 🟩 = Sentiment improving
  - 🟥 = Sentiment declining  
  - ⬜ = Stable sentiment
- See trending food topics extracted from recent discussions

**Explore a Restaurant**
- Select a restaurant from the dropdown
- View recent mentions from Reddit posts, comments, and Google reviews
- Each excerpt includes source (Reddit Post/Reddit Comment/Google Review) and date posted

**AI Search**
- Enter natural language queries like:
  - "where to get good soup"
  - "i am looking for vegan food"
  - "best cheese curds and beer"
- The system retrieves relevant documents and generates an AI answer
- View 3 supporting excerpts from the retrieved documents. Click "See More" button for more excerpts

## Implementation

### Architecture
```
app.py (Streamlit UI)
│
├── helpers.py (helper functions for app.py)
└── utils/
    ├── retriever.py (BM25 indexing)
    ├── agent.py (RAG agent with LangGraph)
    ├── llm.py (Ollama LLM wrapper)
    ├── config.py (Configuration management)
    └── state.py (GraphState definition)
```

### Data Collection

**Google Reviews** (Apify API)
- 60 Milwaukee restaurants (rated above 3 stars)
- Pulled up to 8 reviews per restaurant

**Reddit Data** (PRAW: Reddit API)
- Subreddits: r/milwaukee, r/wisconsin
- Posts and comments mentioning at least one of the 60 restaurants

### Sentiment Analysis

- Custom wrote dict of positive/negative words
- Handles negation
  - "not good" means negative
- Handles intensifiers
  - "very good" means positive
- Sentiment score ranges from -1 to +1

### Trend Detection

Compares recent vs. older reviews to detect sentiment changes:

1. Split reviews: Recent 60% vs. older 40% for each restaurant
2. Calculate average sentiment for each period
3. Determine trend:
   - Change > 0.02 means improving
   - Change < -0.02 means declining
   - Else, it's stable

### RAG (Retrieval Augmented Generation)

**BM25 Indexing** (Pyserini)
- Combines all text into a searchable corpus
- Preprocessing: tokenization, JSON document creation
- Index stored locally in `indexes/`

**Query Processing**:
1. User submits natural language query
2. BM25 retrieves top-k relevant documents

**LLM Generation** (Ollama):
- Model: Llama 3.2:1b
- Limited to 150 tokens for speed

**`helpers.py`**
- `simple_sentiment_analysis()`: Lexicon-based sentiment scoring
- `calculate_trend_direction()`: Compare recent vs. older sentiment
- `detect_trending_restaurants()`: Aggregate mentions and sentiment
- `extract_topics()`: TF-IDF keyword extraction

**`agent.py`**
- `retrieve()`: BM25 retrieval with optional query decomposition
- `generate_answer()`: LLM-based answer generation
- `decompose_query()`: Break complex queries into sub-queries
- `fuse_results()`: Merge and rank multi-query results

**`utils/retriever.py`**
- `preprocess_corpus()`: Convert text to JSON documents
- `build_index()`: Create Pyserini BM25 index
- `PyseriniBM25Retriever`: Custom retriever class

**`app.py`**
- Streamlit interface with three main sections
- Data loading with caching (`@st.cache_data`)
- Real-time query processing and visualization


- **Frontend**: Streamlit
- **Retrieval**: Pyserini (BM25)
- **LLM**: Ollama (Llama 3.2)
- **Orchestration**: LangChain, LangGraph
- **Data Processing**: Pandas, PRAW
- **Sentiment Analysis**: Custom lexicon-based approach
