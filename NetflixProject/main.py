import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

#Data import
df= pd.read_csv(r"C:\Users\shree\Desktop\Netflix dashboard\netflix_titles.csv")

#Streamlit Setup -:
st.set_page_config(
    page_title="NETFLIX",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded")

import streamlit as st

st.header('NETFLIX')
st.sidebar.title("Netflix Content")

user_menu = st.sidebar.radio(
    'Select an Option',
    ('Recommendation','Content by Cast','Content by Director','Content by Genres','Content by Ratings')
)


# Data Processing

df['cast'] = df['cast'].str.replace(r'\s+', '')

df['director'] = df['director'].str.replace(r'\s+', '')

df=df.rename(columns={'listed_in': 'genre'})

#Now I am creating new column named tags.

def create_tags(df):
    # replace null values with an empty string
    df = df.fillna('')

    # concatenate columns to create tags column
    df['tags'] = df['type'] + ' ' + df['genre'] + ' ' + df['description'] + ' ' + df['cast'] + ' ' + df['director']

    # return dataframe with tags column
    return df

tags_df=create_tags(df)

new_df=tags_df[['show_id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

#In tags column i see that some words are repeating so i have to handle that.

import nltk

from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

#Text Vectorisation :

from sklearn.feature_extraction.text  import CountVectorizer

cv = CountVectorizer(max_features=5000,stop_words= 'english')

vectors= cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

#sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:11]

#Recommendation System -:

title_list = df['title'].tolist()
title_list = sorted(title_list, key=lambda x: str(x).lower())
title_list.insert(0, "Select a title")

def recommend(movie):
    if movie == "Select a title":
        return df[['type', 'title', 'cast', 'director', 'rating', 'genre', 'description']]
    else:
        movie_index = new_df[new_df["title"] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(df.iloc[i[0]][['type', 'title', 'cast', 'director', 'rating', 'genre', 'description']])
        recommended_df = pd.DataFrame(recommended_movies)
        return recommended_df



if user_menu == 'Recommendation':
    st.sidebar.header('List for Content')
    selected_content = st.sidebar.selectbox("Select Content", title_list)
    st.header('Recommendations for ' +selected_content)
    x = recommend(selected_content)
    st.dataframe(x)


    # Count the number of movies and TV shows
    counts = df['type'].value_counts()

    # Create a bar chart using Altair
    chart = alt.Chart(df).mark_bar().encode(
    x=alt.X('type', title='Type of content'),
    y=alt.Y('count()', title='Number of titles'),
    color=alt.Color('type', legend=None)
    ).properties(width=400, height=300)

    # Display the chart using Streamlit
    st.altair_chart(chart)


#Contetnt By Cast -:

def get_content_by_cast(name):
    if name == 'Select a cast':
        return df[['type', 'title', 'director', 'cast', 'genre', 'description']]
    else:
        filtered_df = df[df['cast'].str.contains(name, na=False)]
        return filtered_df[['type', 'title', 'director', 'cast', 'genre', 'description']]


#cast_list = list(set(df['cast'].str.split(',').explode()))
cast_list = list(set(df['cast'].str.split(',').explode().map(str)))
cast_list = sorted(cast_list, key=lambda x: str(x).lower())
cast_list.insert(0, "Select a cast")

if user_menu == 'Content by Cast':
    st.sidebar.header('List for Cast')
    selected_cast = st.sidebar.selectbox("Select cast", cast_list)
    st.header('Content By Casts '+ selected_cast)
    cast_rec = get_content_by_cast(selected_cast)
    st.dataframe(cast_rec)



#Content By Director -:
director_list = df['director'].dropna().unique().tolist()
director_list = sorted(director_list, key=lambda x: str(x).lower())
director_list.insert(0, "Select a director")
def get_content_by_director(director):
    if director == "Select a director":
        return df[['type', 'title', 'director', 'cast', 'genre', 'description']]
    else:
        filtered_df = df[df['director'].str.contains(director, na=False)]
    return filtered_df[['type', 'title', 'director', 'cast', 'genre', 'description']]

if user_menu == 'Content by Director':
    st.sidebar.header('List for Director')
    selected_director = st.sidebar.selectbox("Select Director", director_list)
    st.header('Content By Director '+ selected_director)
    director_rec = get_content_by_director(selected_director)
    st.dataframe(director_rec)


#Content By Genres -:

genre_list = list(set(df['genre'].str.split(',').explode()))

genre_list = sorted(genre_list, key=lambda x: str(x).lower())

genre_list.insert(0, "Select a Genre")

def get_content_by_genre(genre):
    if genre == "Select a Genre":
        return df[['type', 'title', 'director', 'cast', 'genre', 'description']]
    else:
        filtered_df = df[df['genre'].str.contains(genre, na=False)]
    return filtered_df[['type', 'title', 'director', 'cast', 'genre', 'description']]

if user_menu == 'Content by Genres':
    st.sidebar.header('List for Genres')
    selected_genre = st.sidebar.selectbox("Select Genre", genre_list)
    st.header('Content By Genres '+ selected_genre)
    genre_rec = get_content_by_genre(selected_genre)
    st.dataframe(genre_rec)

    #Chart for genre
    # Group by genre and type and count the number of occurrences
    genre_counts = df.groupby(['genre', 'type']).size().reset_index(name='count')

    # Plot using Altair
    genre_chart = alt.Chart(genre_counts).mark_bar().encode(
        x='genre',
        y='count',
        color='type',
        tooltip=['genre', 'count']
    ).properties(
        width=600,
        height=400
    )

    # Display the chart using Streamlit
    st.altair_chart(genre_chart)

#Content Based on Rating
rating_list = tags_df['rating'].unique().tolist()
rating_list.sort()
rating_list.insert(0, "Select a rating")

def get_columns_by_rating(rating):
    if rating == "Select a rating":
        return df[['type', 'title', 'director', 'cast', 'genre', 'description']]
    else:
        # filter the dataframe by rating
        filtered_df = df[df['rating'] >= rating]

        # return the columns from the filtered dataframe
        return filtered_df[['type', 'title', 'director', 'cast', 'genre', 'description']]


if user_menu == 'Content by Ratings':
    st.sidebar.header('List for Rating')
    selected_rating = st.sidebar.selectbox("Select rating", rating_list)
    st.header('Content for ' + selected_rating)
    rating_rec = get_columns_by_rating(selected_rating)
    st.dataframe(rating_rec)

    # Filter data by type
    movies = df[df['type'] == 'Movie']
    tv_shows = df[df['type'] == 'TV Show']

    # Group data by rating and count the number of movies and TV shows for each rating
    movies_by_rating = movies.groupby('rating').size().reset_index(name='count')
    tv_shows_by_rating = tv_shows.groupby('rating').size().reset_index(name='count')

    # Create line chart for movies
    movies_chart = alt.Chart(movies_by_rating).mark_line(color='red').encode(
        x='rating:N',
        y='count:Q',
        tooltip=['rating:N', 'count:Q'],
    ).properties(
        title='Number of Movies by Rating'
    )

    # Create line chart for TV shows
    tv_shows_chart = alt.Chart(tv_shows_by_rating).mark_line(color='blue').encode(
        x='rating:N',
        y='count:Q',
        tooltip=['rating:N', 'count:Q'],
    ).properties(
        title='Number of TV Shows by Rating'
    )

    # Combine charts and show in streamlit
    st.altair_chart(movies_chart + tv_shows_chart, use_container_width=True)
