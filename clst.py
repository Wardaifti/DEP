import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the CSV file
data = pd.read_csv(r'c:\Users\Iftikhar\Downloads\transactions-sample.csv')

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Handle missing values (if necessary)
data['productmeasure'].fillna('Unknown', inplace=True)

# Create RFM Features
current_date = data['date'].max()
rfm = data.groupby('id').agg({
    'date': lambda x: (current_date - x.max()).days,
    'id': 'count',
    'purchaseamount': 'sum'
}).rename(columns={
    'date': 'Recency',
    'id': 'Frequency',
    'purchaseamount': 'Monetary'
}).reset_index()

# Ensure the rfm dataframe is created correctly
print(rfm.head())

# Summary statistics
print(rfm.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(rfm['Recency'], bins=30, kde=True)
plt.title('Recency Distribution')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(rfm['Frequency'], bins=30, kde=True)
plt.title('Frequency Distribution')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(rfm['Monetary'], bins=30, kde=True)
plt.title('Monetary Distribution')
plt.show()
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(rfm.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
plt.close()

# Clustering
# Standardize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
plt.close()

# Fit K-Means with the optimal number of clusters (e.g., 4)
optimal_clusters = 4  # Choose the number of clusters based on the elbow method
kmeans = KMeans(n_clusters=optimal_clusters, random_state=1)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Visualize clusters

# Recency vs. Monetary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm, palette='viridis')
plt.title('Customer Segments (Recency vs Monetary)')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.legend(title='Cluster')
plt.show()
plt.close()

# Frequency vs. Monetary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=rfm, palette='viridis')
plt.title('Customer Segments (Frequency vs Monetary)')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.legend(title='Cluster')
plt.show()
plt.close()

# Recency vs. Frequency
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='Frequency', hue='Cluster', data=rfm, palette='viridis')
plt.title('Customer Segments (Recency vs Frequency)')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.legend(title='Cluster')
plt.show()
plt.close()

# Cluster distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=rfm, palette='viridis')
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()
plt.close()

# Save clustered data
rfm.to_csv('/mnt/data/clustered_customers.csv', index=False)
