# Filter users and books based on minimum ratings
MIN_USER_RATINGS = 200
MIN_BOOK_RATINGS = 100

# Count ratings per user
user_counts = df.groupby('User-ID').size()
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index

# Count ratings per book
book_counts = df.groupby('Book-Title').size()
popular_books = book_counts[book_counts >= MIN_BOOK_RATINGS].index

# Filter dataset
df_filtered = df[df['User-ID'].isin(active_users) & df['Book-Title'].isin(popular_books)]

# Create pivot table (users x books)
pivot_table = df_filtered.pivot(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)

# Convert pivot table to numpy array
X = pivot_table.values

# Import and fit KNN model
from sklearn.neighbors import NearestNeighbors

knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
knn_model.fit(X)

# Function to get recommendations
def get_recommends(book_title):
    if book_title not in pivot_table.index:
        return [book_title, []]

    book_idx = pivot_table.index.get_loc(book_title)
    distances, indices = knn_model.kneighbors([X[book_idx]])
    
    recommendations = []
    for i in range(1, 6):
        idx = indices[0][i]
        dist = distances[0][i]
        recommendations.append([pivot_table.index[idx], float(dist)])
    
    return [book_title, recommendations]

# Example usage:
# get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
