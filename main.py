import pandas as pd
from recomendar import recomendar_filmes

ratings = pd.read_csv('./ml-100k/u.data',sep='\t',names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('./ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['movie_id', 'title'])
# print(movies.head())
# print(ratings.head())

user_movies_matrix = ratings.pivot(index='user_id',columns='movie_id',values='rating')

# print(user_movies_matrix.head())
user_movies_matrix = user_movies_matrix.fillna(0)
user_movies_matrix = user_movies_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
# print(user_movies_matrix.head())
filme_escolhido = 1200  
recomendados = recomendar_filmes(filme_escolhido, user_movies_matrix, movies)
print(recomendados)