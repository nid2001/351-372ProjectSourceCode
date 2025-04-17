import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/MyDrive"

df = pd.read_csv("/content/drive/MyDrive/Clothing_Shoes_and_Jewelry.csv")

# Step 2: Combine 'title' and 'text' into a single review string
df['review_text'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Step 3: Drop very short or missing reviews
df = df[df['review_text'].str.len() > 20].copy()

# Step 4: TF-IDF vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=10,
    max_features=1000
)
X = vectorizer.fit_transform(df['review_text'])

# Step 5: KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['topic'] = kmeans.fit_predict(X)


# Step 6: Display top keywords for each topic
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("🧠 Top Keywords per Topic:\n")
for i in range(5):
    print(f"Topic {i + 1}: ", ", ".join([terms[ind] for ind in order_centroids[i, :10]]))

# Step 7: Visualize topic distribution
topic_counts = df['topic'].value_counts().sort_index()
sns.barplot(x=topic_counts.index, y=topic_counts.values)
plt.title("Topic Distribution of Reviews")
plt.xlabel("Topic")
plt.ylabel("Number of Reviews")
plt.show()

# Step 8: Compare rating_diff per topic
df['rating_diff'] = df['predict_stars'] - df['rating']
topic_diff = df.groupby('topic')['rating_diff'].agg(['mean', 'count']).reset_index()
topic_diff.columns = ['Topic', 'Avg Rating Diff', 'Review Count']
print("\n📉 Average Rating Difference per Topic:")
print(topic_diff)

# Step 9: Explore review length by topic
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
bins = [0, 20, 50, 100, 200, float('inf')]
labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
df['length_category'] = pd.cut(df['text_length'], bins=bins, labels=labels)

length_topic = pd.crosstab(df['topic'], df['length_category'], normalize='index') * 100
print("\n📝 Text Length Distribution per Topic (%):")
print(length_topic)

# Step 10: Explore verified status by topic
if 'verified_purchase' in df.columns:
    verified_topic = pd.crosstab(df['topic'], df['verified_purchase'], normalize='index') * 100
    verified_topic.columns = ['Not Verified', 'Verified'] if 0 in verified_topic.columns else verified_topic.columns
    print("\n✅ Verified Purchase Distribution per Topic (%):")
    print(verified_topic)

# Optional: Plot rating difference by topic
sns.barplot(data=topic_diff, x='Topic', y='Avg Rating Diff')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Average Rating Difference by Topic")
plt.ylabel("Predicted - Actual Rating")
plt.show()

# Reuse vectorizer and kmeans from above
# Sample for exploration
df_sample = df.sample(n=50000, random_state=42).copy()
df_sample['review_text'] = df_sample['title'].fillna('') + " " + df_sample['text'].fillna('')

# Use the same vectorizer to transform the sample
X_sample = vectorizer.transform(df_sample['review_text'])

# Predict topic using trained KMeans
df_sample['topic'] = kmeans.predict(X_sample)

# Compute absolute rating difference
df_sample['rating_diff'] = (df_sample['predict_stars'] - df_sample['rating']).abs()

# Get Topic 1 samples with large mismatch
topic_1_mismatched = df_sample[(df_sample['topic'] == 1) & ((df_sample['predict_stars'] - df_sample['rating']) >= 2)]

# Show 5 samples
topic_1_mismatched_samples = topic_1_mismatched[['review_text', 'predict_stars', 'rating']].sample(15, random_state=42)
topic_1_mismatched_samples.reset_index(drop=True, inplace=True)
print(topic_1_mismatched_samples)

import seaborn as sns
import matplotlib.pyplot as plt

# Filter Topic 1 reviews from df_sample
topic_1_reviews = df_sample[df_sample['topic'] == 1].copy()

# Add length and rating_diff columns
topic_1_reviews['text_length'] = topic_1_reviews['review_text'].apply(lambda x: len(str(x).split()))
topic_1_reviews['rating_diff'] = topic_1_reviews['predict_stars'] - topic_1_reviews['rating']

# Separate verified vs unverified
verified = topic_1_reviews[topic_1_reviews['verified_purchase'] == 1]
unverified = topic_1_reviews[topic_1_reviews['verified_purchase'] == 0]

# 1. Distribution of text length
plt.figure(figsize=(10, 5))
sns.kdeplot(verified['text_length'], label='Verified', shade=True)
sns.kdeplot(unverified['text_length'], label='Unverified', shade=True)
plt.title("📝 Text Length Distribution in Topic 1")
plt.xlabel("Number of Words")
plt.legend()
plt.show()

# 2. Distribution of rating difference (predict - real)
plt.figure(figsize=(10, 5))
sns.kdeplot(verified['rating_diff'], label='Verified', shade=True)
sns.kdeplot(unverified['rating_diff'], label='Unverified', shade=True)
plt.title("📉 Rating Difference Distribution in Topic 1")
plt.xlabel("Predicted Stars - Actual Rating")
plt.legend()
plt.show()

# 3. Optional: Boxplot comparison
plt.figure(figsize=(8, 5))
sns.boxplot(data=topic_1_reviews, x='verified_purchase', y='rating_diff')
plt.xticks([0, 1], ['Unverified', 'Verified'])
plt.title("Rating Difference by Verified Status")
plt.ylabel("Predicted - Actual Rating")
plt.show()

# Step 1: Define "long" vs "short"
topic_1_reviews['text_length'] = topic_1_reviews['review_text'].apply(lambda x: len(str(x).split()))
topic_1_reviews['length_category'] = topic_1_reviews['text_length'].apply(lambda x: 'Long' if x >= 50 else 'Short')

# Step 2: Define misleading
topic_1_reviews['is_misleading'] = (topic_1_reviews['predict_stars'] - topic_1_reviews['rating']).abs() >= 2

# Step 3: Categorize into 4 groups
topic_1_reviews['group'] = topic_1_reviews.apply(
    lambda row: f"{row['length_category']} + {'Verified' if row['verified_purchase'] == 1 else 'Unverified'}", axis=1
)

# Step 4: Calculate misleading rates
group_stats = topic_1_reviews.groupby('group')['is_misleading'].agg(['mean', 'count']).reset_index()
group_stats.columns = ['Group', 'Misleading Rate (%)', 'Review Count']
group_stats['Misleading Rate (%)'] *= 100

# Display results
print(group_stats)

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=group_stats, x='Group', y='Misleading Rate (%)')
plt.title("Misleading Review Rate by Group (Topic 1)")
plt.xticks(rotation=45)
plt.ylabel("Misleading Rate (%)")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assume 'topic_1_reviews' is already defined and includes 'rating_diff' and 'text_length'
# If not, rerun the block that creates it from df_sample

# Define 'Long' and 'Short' categories
topic_1_reviews['length_category'] = topic_1_reviews['text_length'].apply(lambda x: 'Long' if x >= 50 else 'Short')

# Plot KDE
plt.figure(figsize=(10, 5))
sns.kdeplot(data=topic_1_reviews[topic_1_reviews['length_category'] == 'Long'], 
            x='rating_diff', label='Long Reviews', shade=True)
sns.kdeplot(data=topic_1_reviews[topic_1_reviews['length_category'] == 'Short'], 
            x='rating_diff', label='Short Reviews', shade=True)

plt.title("📝 Rating Difference Distribution in Topic 1 (Long vs. Short Reviews)")
plt.xlabel("Predicted Stars - Actual Rating")
plt.ylabel("Density")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure these fields exist
topic_1_reviews['text_length'] = topic_1_reviews['review_text'].apply(lambda x: len(str(x).split()))
topic_1_reviews['length_category'] = topic_1_reviews['text_length'].apply(lambda x: 'Long' if x >= 50 else 'Short')

# Define group labels
topic_1_reviews['group'] = topic_1_reviews.apply(
    lambda row: f"{'Verified' if row['verified_purchase'] == 1 else 'Unverified'} + {row['length_category']}", axis=1
)

# Plot KDE for each group
plt.figure(figsize=(12, 6))
groups = topic_1_reviews['group'].unique()
colors = ['blue', 'orange', 'green', 'red']

for group, color in zip(groups, colors):
    subset = topic_1_reviews[topic_1_reviews['group'] == group]
    sns.kdeplot(data=subset, x='rating_diff', label=group, shade=True, color=color)

plt.title("📉 Rating Difference Distribution in Topic 1 by Length and Verification")
plt.xlabel("Predicted Stars - Actual Rating")
plt.ylabel("Density")
plt.legend(title="Group")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure these fields exist
topic_1_reviews['text_length'] = topic_1_reviews['review_text'].apply(lambda x: len(str(x).split()))
topic_1_reviews['length_category'] = topic_1_reviews['text_length'].apply(lambda x: 'Long' if x >= 50 else 'Short')

# Define group labels
topic_1_reviews['group'] = topic_1_reviews.apply(
    lambda row: f"{'Verified' if row['verified_purchase'] == 1 else 'Unverified'} + {row['length_category']}", axis=1
)

# Plot KDE for each group
plt.figure(figsize=(12, 6))
groups = topic_1_reviews['group'].unique()
colors = ['blue', 'orange', 'green', 'red']

for group, color in zip(groups, colors):
    subset = topic_1_reviews[topic_1_reviews['group'] == group]
    sns.kdeplot(data=subset, x='rating_diff', label=group, shade=True, color=color)

plt.title("📉 Rating Difference Distribution in Topic 1 by Length and Verification")
plt.xlabel("Predicted Stars - Actual Rating")
plt.ylabel("Density")
plt.legend(title="Group")
plt.show()

# Extract long, unverified, and misleading (abs(predict - rating) >= 2) reviews from Topic 1
long_unverified_misleading = topic_1_reviews[
    (topic_1_reviews['length_category'] == 'Long') &
    (topic_1_reviews['verified_purchase'] == 0) &
    ((topic_1_reviews['predict_stars'] - topic_1_reviews['rating']).abs() >= 2)
]

# Sample 5 for display
sample_reviews = long_unverified_misleading[['review_text', 'predict_stars', 'rating']].sample(5, random_state=42).reset_index(drop=True)

# Truncate text for preview
truncated_samples = sample_reviews.copy()
truncated_samples['review_text'] = truncated_samples['review_text'].apply(lambda x: x[:400] + "..." if len(x) > 400 else x)

truncated_samples

