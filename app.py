# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:53:39 2020

@author: Harivamshi
"""
from flask import Flask
import flask
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app=Flask(__name__,template_folder='html')

df=pd.read_csv('./data/movies_metadata.csv')
df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.95)
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)

s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = df.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    if len(genre.split())>1:
        genres=genre.split()
        genre=''
        for i in genres:
            genre+=i[0].upper()+i[1:].lower()+' '
        genre=genre.strip()
    else:
        genre=genre[0].upper()+genre[1:].lower()
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified

links_small = pd.read_csv('./data/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

df = df.drop([19730, 29503, 35587])

df['id'] = df['id'].astype('int')

smd = df[df['id'].isin(links_small)]
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'].str.lower())

def get_recommendations(title):
    title=title.lower()
#     indices=ind.reindex(ind.index.str.lower())
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method=='POST':
        return(flask.render_template('rec.html'))
            
            
@app.route('/recommendations', methods=['GET','POST'])

def recommendations():
    if flask.request.method == 'POST':
        movie = flask.request.form['movie_name']
        movie=movie.title()
        if movie in list(titles):
            result=get_recommendations(movie).reset_index(drop=True)

            names=[]
            dates=[]
            for i in range(len(result)):
                names.append(result[i])
            return flask.render_template('recommended_movies.html',movie_names=names,movie_date=dates,search_name=movie)
        else:
            return(flask.render_template('negative.html',name=movie))    

@app.route('/topmovies',methods=['GET','POST'])
def topMovies():
    if flask.request.method=='POST':
        names=[]
        dates=[]
        for i in range(len(qualified)):
            names.append(qualified.iloc[i]['title'])
            dates.append(qualified.iloc[i]['year'])
        return flask.render_template('topmovies.html',movie_names=names,movie_date=dates)

@app.route('/genres',methods=['GET','POST'])
def genres():
    if flask.request.method=='POST':
        genre=flask.request.form['genre_name']
        result=build_chart(genre)
        names=[]
        dates=[]
        for i in range(len(result)):
            names.append(result.iloc[i]['title'])
            dates.append(result.iloc[i]['year'])
        return flask.render_template('topgenres.html',movie_names=names,movie_date=dates,genre_name=genre)