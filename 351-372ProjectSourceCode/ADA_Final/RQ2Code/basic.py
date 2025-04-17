import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load and combine datasets

# ============================== Basic Statistics ==============================
df1 = pd.read_csv("/content/drive/MyDrive/Clothing_Shoes_and_Jewelry.csv")
df2 = pd.read_csv("/content/drive/MyDrive/Copy of Electronics.csv")
df3 = pd.read_csv("/content/drive/MyDrive/Copy of Health_and_Personal_Care.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)

# Combine 'title' and 'text' into a single 'review_text' column
df['review_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df = df[df['review_text'].str.len() > 20].copy()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=10, max_features=1000)
X = vectorizer.fit_transform(df['review_text'])

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['topic'] = kmeans.fit_predict(X)

# Set product and user columns dynamically
product_col = 'product_id' if 'product_id' in df.columns else 'asin'
user_col = 'user_id' if 'user_id' in df.columns else 'reviewerID'

# Prepare analysis columns
df['text_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
df['is_misleading'] = (df['predict_stars'] - df['rating']).abs() >= 2

print("📊 Dataset Overview")
print("Total Reviews:", len(df))
print("Total Unique Users:", df[user_col].nunique())
print("Total Unique Products:", df[product_col].nunique())
print("Verified Reviews:", df['verified_purchase'].sum())
print("Unverified Reviews:", len(df) - df['verified_purchase'].sum())
print("Misleading Reviews:", df['is_misleading'].sum())
print("Misleading Review Rate: {:.2f}%".format(df['is_misleading'].mean() * 100))

# Distribution of review lengths
plt.figure(figsize=(10, 5))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title("📝 Distribution of Review Lengths (in words)")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Rating distribution vs predicted stars
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='rating', data=df)
plt.title("⭐ Original Rating Distribution")
plt.subplot(1, 2, 2)
sns.countplot(x='predict_stars', data=df)
plt.title("🤖 Predicted Rating Distribution")
plt.tight_layout()
plt.show()

# Rating distribution vs predicted stars
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='rating', data=df)
plt.title("⭐ Original Rating Distribution")
plt.subplot(1, 2, 2)
sns.countplot(x='predict_stars', data=df)
plt.title("🤖 Predicted Rating Distribution")
plt.tight_layout()
plt.show()
df1 = pd.read_csv("/content/drive/MyDrive/Clothing_Shoes_and_Jewelry.csv")
df2 = pd.read_csv("/content/drive/MyDrive/Copy of Electronics.csv")
df3 = pd.read_csv("/content/drive/MyDrive/Copy of Health_and_Personal_Care.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)

# Combine 'title' and 'text' into a single 'review_text' column
df['review_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df = df[df['review_text'].str.len() > 20].copy()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=10, max_features=1000)
X = vectorizer.fit_transform(df['review_text'])

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['topic'] = kmeans.fit_predict(X)

# Set product and user columns dynamically
product_col = 'product_id' if 'product_id' in df.columns else 'asin'
user_col = 'user_id' if 'user_id' in df.columns else 'reviewerID'

# Prepare analysis columns
df['text_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
df['is_misleading'] = (df['predict_stars'] - df['rating']).abs() >= 2