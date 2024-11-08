from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def recomendar_filmes(filme_id,user_movies_matrix,movies,num_recomendacoes = 5):
    
    cosine_sim = cosine_similarity(user_movies_matrix.T.fillna(0))
    
    cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movies_matrix.columns, columns=user_movies_matrix.columns)

    similaridades = cosine_sim_df[filme_id].sort_values(ascending=False)
    
    recomendados = similaridades.drop(filme_id).head(num_recomendacoes)
    
    filmes_recomendados = movies[movies['movie_id'].isin(recomendados.index)]
    return filmes_recomendados[['movie_id', 'title']]