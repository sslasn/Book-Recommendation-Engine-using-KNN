import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

books = pd.read_csv("BX-Books.csv", sep=";", error_bad_lines=False, encoding="latin-1")
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", error_bad_lines=False, encoding="latin-1")
df = pd.merge(ratings, books, on="ISBN")
user_counts = df["User-ID"].value_counts()
book_counts = df["ISBN"].value_counts()
df = df[df["User-ID"].isin(user_counts[user_counts >= 200].index)]
df = df[df["ISBN"].isin(book_counts[book_counts >= 100].index)]
label_encoder = LabelEncoder()
df["Book-ID"] = label_encoder.fit_transform(df["Book-Title"])
book_pivot = df.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating").fillna(0)
knn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute", n_jobs=-1)
knn.fit(book_pivot.values.T)
def get_recommends(book_title):
    encoded_book_title = label_encoder.transform([book_title])[0]
    book_idx = book_pivot.index.get_loc(book_title)
    distances, indices = knn.kneighbors(book_pivot.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=6)
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_books.append([label_encoder.inverse_transform([book_pivot.index[indices.flatten()[i]]])[0], distances.flatten()[i]])

    return [book_title, recommended_books]
recommendations = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(recommendations)
