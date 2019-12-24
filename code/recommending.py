from fuzzywuzzy import fuzz
import numpy as np
import pickle
from sklearn.metrics.pairwise import pairwise_distances

with open('pickles/modeled_data','rb') as file:
    movie_to_id = pickle.load(file)
    id_to_movie = pickle.load(file)
    movie_titles = pickle.load(file)
    doc_topic = pickle.load(file)

def recommend_movie(movie_input):
    '''
    Recommends movie(s) based on highest cosine similarity.

    Parameters
    ----------
    movie_input : str
        string of movie titles separated by commas
    n_recs : int
        number of movie recommendations to output

    Returns
    -------
    list
        list of recommended movie titles
    '''
    
    # Empty list of ranks
    ranks = []

    # Retrieve individual movies by splitting at comma
    movie_input = movie_input.split(",")

    # Retrieve the requested number of recommendations
    n_recs = int(movie_input.pop().strip())

    if n_recs < 1:
        return ("You chose to receive 0 recommendations.")

    # Clean up white space for each entry
    for idx in range(len(movie_input)):
        movie_input[idx] = movie_input[idx].strip()

    # For each movie in the list of inputted movies
    for idx, movie in enumerate(movie_input):
        
        # Returns the closest movie title if typo
        movie = spell_check(movie, movie_titles)

        # Edits the entry in the movie input list
        movie_input[idx] = movie

        # Turn movie string into row index for movie
        movie = movie_to_id[movie]
        
        # Cosine distances for the given movie to all others
        dists = [dist[0] for dist in pairwise_distances(doc_topic, doc_topic[movie].reshape(1,-1))]

        # Sort the distances from closest to furthest, excluding the movie itself, and retain movie ids
        rec_movie_ids = np.argsort(dists)[1:]
        
        # Add this movie's ranks to the ranks list
        ranks.append(rec_movie_ids)
     
    # Create a dictionary of "average" ranks per movie
    rank_dict = {}
    
    # Loop through each movie and add the ranks up
    for i in range(len(movie_input)):
        for idx, movie in enumerate(ranks[i]):
            try:
                rank_dict[movie] += idx
            except:
                rank_dict[movie] = idx
    
    # Generate and return movie recommendation(s), and spell checked movie input
    if n_recs == 1:
        movie_recommendation = [id_to_movie[min(rank_dict, key = rank_dict.get)]]
        return movie_recommendation, movie_input
    else:
        movie_recommendations = [id_to_movie[x[0]] for x in sorted(list(rank_dict.items()), key = lambda x: x[1])][:int(n_recs)]
        return movie_recommendations, movie_input

def spell_check(movie_input, movie_titles):
    '''
    Gives you the most likely movie name based on
    what you type in.

    Parameters
    ----------
    movie_input : string
        the movie that is being spell-checked
    movie_titles : list
        list of all movie titles to compare to

    Returns
    -------
    string
        closest movie title in the list
    '''
    
    most_similar = 0
    for movie in movie_titles:
        ratio = fuzz.ratio(movie_input, movie)
        if ratio > most_similar:
            most_similar = ratio
            closest_movie = movie
    return closest_movie

def main():
    print('Search for any number of movies, separated by a commas.\n')
    print('The last entry should be the number of recommendations.\n')
    print('For example: Get Out, The Ring, Step Brothers, 3\n')
    movie_input = input('Enter your movies here: ')
    recommendations, movie_input = recommend_movie(movie_input)

    print('\n==========You searched for: \n')
    for movie in movie_input:
        print(f'- {movie}\n')
    print('==========You should watch: \n')
    for movie in recommendations:
        print(f'- {movie}\n')

main()