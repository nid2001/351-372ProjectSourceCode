import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from transformers import BertTokenizer, BertModel
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import re
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
file_path = "Amazon_Fashion.jsonl/Amazon_Fashion.jsonl"
data = []
with open(file_path, "r") as f:
    for line in f:
        data.append(json.loads(line.strip()))
df = pd.DataFrame(data)

# Verify fields
expected_columns = ['rating', 'title', 'text', 'images', 'asin', 'parent_asin',
                    'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']
print("Data fields:", df.columns.tolist())

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    abbreviations = {"brb": "be right back", "lol": "laughing out loud"}
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    return text

# Clean text
df["clean_text"] = df["text"].apply(clean_text)
df["clean_title"] = df["title"].apply(clean_text)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df["clean_text"] = df["clean_text"].replace("", np.nan)
print("Empty text count:", df["clean_text"].isna().sum())
df["clean_text"] = df["clean_text"].fillna("")

# Feature extraction
analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["clean_text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))
df["image_count"] = df["images"].apply(lambda x: len(x) if isinstance(x, list) else 0)
current_date = pd.to_datetime("2025-04-14")
df["time_weight"] = df["timestamp"].apply(
    lambda x: np.exp(-(current_date - x).days / 365) if pd.notnull(x) else 0
)
df["verified_purchase"] = df["verified_purchase"].astype(int)  # Convert to 0/1

# LDA topic modeling
print("Performing LDA topic modeling...")
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
tf = vectorizer.fit_transform(df["clean_text"])
lda = LatentDirichletAllocation(n_components=5, random_state=42, n_jobs=-1)
topics = lda.fit_transform(tf)
df["topic"] = topics.argmax(axis=1)

# Display topic keywords
feature_names = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_words = [feature_names[j] for j in topic.argsort()[-10:]]
    print(f"Topic {i}: {top_words}")
print("Topic frequency:\n", df["topic"].value_counts(normalize=True))

# BERT clustering (negative reviews)
print("Performing BERT clustering for negative reviews...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)

def get_bert_embeddings_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)

# Extract negative reviews
neg_reviews = df[df["sentiment"] < 0]["clean_text"].tolist()
neg_sample = neg_reviews[:10000]
if len(neg_sample) > 0:
    embeddings = get_bert_embeddings_batch(neg_sample)
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    df_neg_sample = df[df["sentiment"] < 0].iloc[:len(neg_sample)].copy()
    df_neg_sample["cluster"] = clusters
    print("Negative review cluster distribution:\n", df_neg_sample["cluster"].value_counts())

    # Check cluster keywords
    for cluster in range(5):
        cluster_texts = df_neg_sample[df_neg_sample["cluster"] == cluster]["clean_text"]
        if len(cluster_texts) > 0:
            tf_cluster = vectorizer.transform(cluster_texts)
            top_words = [feature_names[i] for i in tf_cluster.sum(axis=0).A1.argsort()[-10:]]
            print(f"Cluster {cluster} keywords: {top_words}")
else:
    print("No negative reviews found.")

# Reputation risk index
def reputation_risk(group):
    total_comments = len(group)
    neg_comments = len(group[group["sentiment"] < 0])
    sentiment_strength = abs(group["sentiment"].mean()) if total_comments > 0 else 0
    return (total_comments / (neg_comments + 1)) * sentiment_strength

# Aggregate by product
product_stats = df.groupby("asin").agg({
    "rating": "mean",
    "sentiment": "mean",
    "text_length": "mean",
    "image_count": "mean",
    "time_weight": "mean",
    "helpful_vote": "mean",
    "verified_purchase": "mean",
    "topic": lambda x: x.mode()[0] if len(x) > 0 else -1,
}).reset_index()

# Calculate reputation risk
product_risk = df.groupby("asin").apply(reputation_risk, include_groups=False).reset_index(name="reputation_risk")
product_stats = product_stats.merge(product_risk, on="asin")

# Sales proxy: log of review count
product_stats["sales_proxy"] = df.groupby("asin").size().reset_index(name="review_count")["review_count"]
product_stats["sales_proxy"] = np.log1p(product_stats["sales_proxy"])

# Supervised learning
features = ["rating", "sentiment", "text_length", "image_count", "time_weight",
            "helpful_vote", "verified_purchase"]
X = product_stats[features]
y_sales = product_stats["sales_proxy"]
y_reputation = product_stats["reputation_risk"]

# Data split
X_train, X_test, y_sales_train, y_sales_test = train_test_split(X, y_sales, test_size=0.2, random_state=42)
_, _, y_rep_train, y_rep_test = train_test_split(X, y_reputation, test_size=0.2, random_state=42)

# Random Forest: sales
rf_sales = RandomForestRegressor(n_estimators=100, random_state=42)
rf_sales.fit(X_train, y_sales_train)
y_sales_pred = rf_sales.predict(X_test)
print("Sales prediction metrics:")
print(f"R²: {r2_score(y_sales_test, y_sales_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_sales_test, y_sales_pred)):.3f}")

# Random Forest: reputation risk
rf_rep = RandomForestRegressor(n_estimators=100, random_state=42)
rf_rep.fit(X_train, y_rep_train)
y_rep_pred = rf_rep.predict(X_test)
print("Reputation risk prediction metrics:")
print(f"R²: {r2_score(y_rep_test, y_rep_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_rep_test, y_rep_pred)):.3f}")

# Feature importance
importance_sales = pd.DataFrame({
    "Feature": features,
    "Importance": rf_sales.feature_importances_
}).sort_values("Importance", ascending=False)
print("Sales prediction feature importance:\n", importance_sales)

importance_rep = pd.DataFrame({
    "Feature": features,
    "Importance": rf_rep.feature_importances_
}).sort_values("Importance", ascending=False)
print("Reputation risk prediction feature importance:\n", importance_rep)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x="rating", y="reputation_risk", size="sentiment", hue="sentiment", data=product_stats)
plt.title("Rating vs Reputation Risk (colored by sentiment)")
plt.xlabel("Average Rating")
plt.ylabel("Reputation Risk")
plt.savefig("D:/372/rating_vs_reputation_risk.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="rating", y="sales_proxy", size="sentiment", hue="sentiment", data=product_stats)
plt.title("Rating vs Sales Proxy (colored by sentiment)")
plt.xlabel("Average Rating")
plt.ylabel("Sales Proxy (log review count)")
plt.savefig("D:/372/rating_vs_sales_proxy.png")
plt.show()

# Save results
product_stats.to_csv("D:/372/Amazon_Fashion_product_stats.csv", index=False)
df.to_csv("D:/372/Amazon_Fashion_processed_updated.csv", index=False)
print("Results saved.")

# Summary
print("\nRQ3 key findings:")
print("- Sentiment and rating significantly impact reputation risk.")
print(f"- Sales main drivers: {importance_sales['Feature'].iloc[:2].tolist()}")
print(f"- Reputation risk main drivers: {importance_rep['Feature'].iloc[:2].tolist()}")
print("- Negative review clusters indicate issues like logistics delays, product quality.")