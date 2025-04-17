import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load and combine datasets

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

# ============================== Product-Level Analysis (by Count) ==============================
product_mismatch = df.groupby(product_col)['is_misleading'].agg(['mean', 'count', 'sum']).reset_index()
product_mismatch.columns = ['Product ID', 'Misleading Rate (%)', 'Review Count', 'Misleading Count']
product_mismatch['Misleading Rate (%)'] *= 100
filtered_products = product_mismatch[product_mismatch['Review Count'] >= 20]
top_mismatch_products = filtered_products.sort_values('Misleading Count', ascending=False).head(10)
print("\n🔍 Top 10 Products with Most Misleading Reviews:")
print(top_mismatch_products[['Product ID', 'Review Count', 'Misleading Count', 'Misleading Rate (%)']])

plt.figure(figsize=(12, 6))
sns.barplot(data=top_mismatch_products, x='Product ID', y='Misleading Count', palette='magma')
plt.title("📦 Top 10 Products by Number of Misleading Reviews")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Misleading Review Count")
plt.xlabel("Product ID")
plt.tight_layout()
plt.show()



# ============================== Time Trend by Verification (Top Products) ==============================
df['review_time'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
top_product_ids = top_mismatch_products['Product ID'].tolist()
df_top_products = df[df[product_col].isin(top_product_ids) & df['is_misleading']]
trend_data = df_top_products.groupby([pd.Grouper(key='review_time', freq='M'), 'verified_purchase'])['is_misleading'].count().reset_index()
trend_data.columns = ['Month', 'Verified', 'Misleading Count']
trend_data['Verified'] = trend_data['Verified'].map({1: 'Verified', 0: 'Unverified'})

plt.figure(figsize=(14, 6))
sns.lineplot(data=trend_data, x='Month', y='Misleading Count', hue='Verified', marker='o')
plt.title("📆 Monthly Misleading Review Count (Verified vs Unverified, Top 10 Products)")
plt.ylabel("Misleading Review Count")
plt.xlabel("Month")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================== Sample Reviews from Top Product ==============================
top_product = top_mismatch_products.iloc[0]['Product ID']
samples = df[(df[product_col] == top_product) & (df['is_misleading'])][['review_text', 'predict_stars', 'rating']].sample(5, random_state=42).reset_index(drop=True)
print(samples)

# ============================== Verified vs Unverified (Product Level) ==============================
if 'verified_purchase' in df.columns:
    vis_data = []
    for pid in top_mismatch_products['Product ID']:
        subset = df[df[product_col] == pid]
        breakdown = pd.crosstab(subset['verified_purchase'], subset['is_misleading'], normalize='index') * 100
        for v in [0, 1]:
            if v in breakdown.index:
                rate = breakdown.loc[v, True] if True in breakdown.columns else 0
                vis_data.append({
                    'Product ID': pid,
                    'Verification': 'Verified' if v == 1 else 'Unverified',
                    'Misleading Rate (%)': rate
                })
    vis_df = pd.DataFrame(vis_data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=vis_df, x='Product ID', y='Misleading Rate (%)', hue='Verification')
    plt.title("🔍 Misleading Rate by Verification Status (Top 10 Products)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.show()
    display(vis_df.pivot(index='Product ID', columns='Verification', values='Misleading Rate (%)').round(2))

# ============================== User-Level Analysis ==============================
user_mismatch = df.groupby(user_col)['is_misleading'].agg(['mean', 'count', 'sum']).reset_index()
user_mismatch.columns = ['User ID', 'Misleading Rate (%)', 'Review Count', 'Misleading Count']
user_mismatch['Misleading Rate (%)'] *= 100
active_users = user_mismatch[user_mismatch['Review Count'] >= 5]
top_users = active_users.sort_values('Misleading Count', ascending=False).head(10)
print("\n🔍 Top 10 Users with Most Misleading Reviews:")
print(top_users)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_users, x='User ID', y='Misleading Count', palette='coolwarm')
plt.title("👤 Top 10 Users by Number of Misleading Reviews")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Misleading Review Count")
plt.xlabel("User ID")
plt.tight_layout()
plt.show()

# ============================== Time Trend for Top Users ==============================
suspicious_ids = top_users['User ID'].tolist()
df_suspicious = df[df[user_col].isin(suspicious_ids)].copy()
monthly_user_trend = df_suspicious.groupby([pd.Grouper(key='review_time', freq='M'), user_col, 'verified_purchase'])['is_misleading'].sum().reset_index()
monthly_user_trend['verified_purchase'] = monthly_user_trend['verified_purchase'].map({1: 'Verified', 0: 'Unverified'})

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_user_trend, x='review_time', y='is_misleading', hue='verified_purchase', style=user_col, markers=True)
plt.title("📆 Monthly Misleading Count of Top Users (by Verification)")
plt.ylabel("Misleading Review Count")
plt.xlabel("Month")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================== Verified vs Unverified (User Level) ==============================
if 'verified_purchase' in df.columns:
    verified_data = []
    for user in suspicious_ids:
        user_df = df[df[user_col] == user]
        breakdown = pd.crosstab(user_df['verified_purchase'], user_df['is_misleading'], normalize='index') * 100
        for v in [0, 1]:
            if v in breakdown.index:
                rate = breakdown.loc[v, True] if True in breakdown.columns else 0
                verified_data.append({
                    'User ID': user,
                    'Verification': 'Verified' if v == 1 else 'Unverified',
                    'Misleading Rate (%)': rate
                })
    verified_df = pd.DataFrame(verified_data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=verified_df, x='User ID', y='Misleading Rate (%)', hue='Verification')
    plt.title("✅❌ Misleading Rate by Verification Status (Top Suspicious Users)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.show()
    display(verified_df.pivot(index='User ID', columns='Verification', values='Misleading Rate (%)').round(2))
