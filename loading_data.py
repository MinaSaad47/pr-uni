import pandas as pd

def __load_data(director_file, actor_file, revenue_file, target_actual, target_new):
    movie_director = pd.read_csv(director_file)
    movie_director_columns = {'name': 'Title', 'director' : 'Director'}
    movie_director.rename(columns=movie_director_columns, inplace=True)
    # movie_director.set_index('Title', verify_integrity=False)

    movie_revenue = pd.read_csv(revenue_file)
    movie_revenue_columns = {'movie_title': 'Title', 'release_date': 'Date', 'genre': 'Genre', 'MPAA_rating': 'MPAA', target_actual: target_new}
    movie_revenue.rename(columns=movie_revenue_columns, inplace=True)
    # movie_revenue.set_index('Title', verify_integrity=False)

    movie_actor = pd.read_csv(actor_file)
    movie_actor_columns = {'movie': 'Title', 'voice-actor': 'Actor', 'character': 'Character'}
    movie_actor.rename(columns=movie_actor_columns, inplace=True)
    # movie_actor.set_index('Title', verify_integrity=False)

    total_data = pd.merge(movie_director, movie_revenue, on=['Title'], how='outer')
    total_data = pd.merge(total_data, movie_actor, on=['Title'], how='outer')

    total_data = total_data[['Title', 'Date', 'Genre', 'MPAA', 'Director', 'Actor', 'Character', target_new]]

    return total_data

def load_classification_data(director_file, actor_file, revenue_file):
    return __load_data(director_file, actor_file, revenue_file, 'MovieSuccessLevel', 'MovieSuccessLevel')

def load_regression_data(director_file, actor_file, revenue_file):
    return __load_data(director_file, actor_file, revenue_file, 'revenue', 'MovieRevenue')