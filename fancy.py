#================IMPORTING LIBRARIES================
import streamlit as st
import pandas as pd
import requests
from surprise import Reader, Dataset, SVD
from dotenv import load_dotenv
import os



#===============SETTING PAGE==========================
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.big-font {
    font-size:26px !important;
    font-weight: bold;
    color: #009688;  # Adjust the color to fit your theme
}
.stButton>button {
    color: white;
    background-color: #4CAF50;  # Green background for buttons
    border-radius:20px;
    border:1px solid white;
}
.stTextInput>div>div>input {
    color: #4CAF50;
}
body {
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

#===============LOADING DATA==========================
@st.cache_data  # Updated cache command
def load_movie_data(file_path):
    return pd.read_csv(file_path)

# Ensure these are the correct paths and they are called after `set_page_config`
movies_df = pd.read_csv('movies_links_2.csv')
ratings_data = pd.read_csv('ratings.csv')

#================COMPANY LOGO=========================
st.markdown("""
    <h1 style='text-align: left; color: #009688;'>Welcome to WBS CUEVANA</h1>
    """, unsafe_allow_html=True)

#============== Add GIF after the company logo========
st.markdown("""
    <img src="https://media.giphy.com/media/l0HluKLonblTf8ili/giphy.gif?cid=790b7611im299mxs3q7bvnaoxeyjfsrk2q9sd54bxcqiz1a7&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="300">
    """, unsafe_allow_html=True)

#=================API FUNCTION==========================
@st.cache_data  # Cache the function to prevent repeated API calls
def get_movie_details(title, movies_df):
    filtered_df = movies_df.loc[movies_df['title'] == title, 'tmdbId']
    #load_dotenv()
    if not filtered_df.empty:
        tmdbId = filtered_df.iloc[0]
        api_key = st.secrets['API_KEY'] # Your TMDb API key
        try:
            response = requests.get(f'https://api.themoviedb.org/3/movie/{tmdbId}?api_key={api_key}')
            if response.status_code == 200:
                data = response.json()
                # Extracting the required details
                poster_path = data.get('poster_path')
                description = data.get('overview')
                duration = data.get('runtime')
                
                # Constructing the poster URL
                poster_url = f'https://image.tmdb.org/t/p/w500{poster_path}' if poster_path else None
                
                return {
                    'poster_url': poster_url,
                    'description': description,
                    'duration': duration
                }
            else:
                print("Failed to fetch movie details. Response status code:", response.status_code)
        except Exception as e:
            print(f"Error during API request: {e}")
    else:
        print(f"No TMDb ID found for movie: {title}")
    
    return None


#===========FUNCTION FOR MOST POPULAR MOVIES==========
def top_movies(n, ratings_data, movies_df):
    df = pd.DataFrame(ratings_data.groupby('movieId')['rating'].mean())
    df['rating_count'] = ratings_data.groupby('movieId')['rating'].count()
    df['combined_metric'] = 2 * df['rating'] + df['rating_count']
    df = pd.merge(df, movies_df, on="movieId", how="inner")[["movieId", "title", "rating", "rating_count", "combined_metric"]]
    df = df.sort_values(by='combined_metric', ascending=False).head(n)
    top_n_movies = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    return top_n_movies

#================FIRST BANNER==========================
st.markdown('<p class="big-font">Explore the Blockbusters Everyone\'s Talking About!</p>', unsafe_allow_html=True)

n = 5  # Number of top movies you want
top_n_movies = top_movies(n, ratings_data, movies_df) # Running the function with parameters

movie_titles = top_n_movies['title'].tolist()  # Getting the movie titles

# Create columns for each movie. The number of columns should match the number of movies in 'movie_titles'
cols = st.columns(len(movie_titles))

for i, movie_title in enumerate(movie_titles):
    movie_details = get_movie_details(movie_title, movies_df)  # Fetch movie details
    # Display the movie title and its poster in each column
    with cols[i]:
        st.markdown(f"**{movie_title}**")  # Display the movie title in bold
        if movie_details and movie_details['poster_url']:
            # Using the caption of the image as a makeshift button for expanding details
            if st.image(movie_details['poster_url'], width=150, caption="Click for details"):
                # This is a conceptual placeholder; Streamlit does not support clickable images directly
                pass
            # Use an expander to show movie details
            with st.expander("View Details"):
                st.write(f"**Description:** {movie_details['description']}")
                st.write(f"**Duration:** {movie_details['duration']} minutes")
        else:
            st.write("Poster not available.")

#=================FUNCTION ITEM BASED==============================

def recommend_similar_movies(ratings_data, movies_df, movie_id, top_n_movies, similarity_threshold, no_of_users_threshold):
    
    movie_id = int(movie_id) # Ensure movie_id is an integer, assuming movieId columns are integers
    
    # Generating user-item matrix
    user_movie_matrix = pd.pivot_table(data=ratings_data, values='rating', index='userId', columns='movieId', fill_value=0)
    
    # Getting (Cosine) Similarity matrix
    movie_similarity_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T), columns=user_movie_matrix.columns, index=user_movie_matrix.columns)
    
    # Check if movie_id exists in the matrix
    if movie_id not in movie_similarity_matrix.columns:
        print(f"Movie ID {movie_id} not found in similarity matrix columns.")
        return pd.DataFrame()  # Return an empty DataFrame or handle appropriately
    
    # Getting similarities for selected movie
    movie_similarities = movie_similarity_matrix[movie_id].drop(movie_id).to_frame(name="movie_similarities").sort_values("movie_similarities", ascending=False)
    
    # Filter for number of users who rated both movies
    no_of_users_rated_both_movies = user_movie_matrix.apply(lambda x: sum((x > 0) & (user_movie_matrix[movie_id] > 0)), axis=0)
    movie_similarities["users_who_rated_both_movies"] = no_of_users_rated_both_movies
    
    movie_similarities = movie_similarities[movie_similarities["users_who_rated_both_movies"] > no_of_users_threshold]
    movie_similarities = movie_similarities[movie_similarities["movie_similarities"] > similarity_threshold]
    
    # Merge to get titles and genres
    top_n_similar_movies = movie_similarities.head(top_n_movies).merge(movies_df, left_index=True, right_on="movieId", how="left")
    
    return top_n_similar_movies

#====================SECOND BANNER==================================

st.markdown('<p class="big-font">Liked That? You\'ll Love These!</p>', unsafe_allow_html=True)

# Create a layout with columns: one for the expander and one for the poster
col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed

with col1:
    my_expander = st.expander("Tap to Select a Movie")
    default_selection = ('',) + tuple(movies_df["title"].unique())
    selected_movie = my_expander.selectbox("Choose a movie", default_selection, index=0)

    if selected_movie:  # Only proceed if a movie is selected
        selected_movieId = movies_df.loc[movies_df["title"] == selected_movie, "movieId"].iloc[0]
        
        if my_expander.button("Recommend"):
            top10_similar_movies = recommend_similar_movies(
                ratings_data=ratings_data, 
                movies_df=movies_df, 
                movie_id=selected_movieId, 
                top_n_movies=5, 
                similarity_threshold=0.1, 
                no_of_users_threshold=5
            )

            if not top10_similar_movies.empty:
                st.text("Here are a few Recommendations..")
                columns = st.columns(len(top10_similar_movies))

                for idx, row in enumerate(top10_similar_movies.itertuples()):
                    movie_details = get_movie_details(row.title, movies_df)
                    with columns[idx % len(columns)]:
                        st.markdown(f"**{row.title}**")
                        if movie_details['poster_url']:
                            st.image(movie_details['poster_url'], width=150)
                            with st.expander("View Details"):
                                st.write(f"**Description:** {movie_details['description']}")
                                st.write(f"**Duration:** {movie_details['duration']} minutes")
                        else:
                            st.write("Poster not available.")
            else:
                st.write("No recommendations found.")

# Fetch and display the poster and details in the second column for the selected movie
with col2:
    if selected_movie:
        movie_details = get_movie_details(selected_movie, movies_df)
        if movie_details['poster_url']:
            st.image(movie_details['poster_url'], width=150)
            st.write(f"**Description:** {movie_details['description']}")
            st.write(f"**Duration:** {movie_details['duration']} minutes")
        else:
            st.write("No poster available.")

#==================THIRD FUNCTION=====================

   # Define reader and load data
def recommend_what_others_like(
        ratings_data: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_id: int, 
        top_n: int):
    
    # Define reader and load data
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    # Filter the testset to include only rows for the specified user
    filtered_testset = [row for row in testset if row[0] == user_id]

    # Create and train SVD model
    model = SVD(
        n_factors=150, n_epochs=30, 
        lr_all=0.01, reg_all=0.1, 
        random_state=42)
    model.fit(trainset)

    # Make predictions on the filtered test set
    predictions = model.test(filtered_testset)
    predictions_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    # Return the top n recommended movies
    movies = predictions_df.nlargest(top_n, 'est') # 'est' => estimated rating

    # Get the movie names
    movie_IDs = movies["iid"].tolist()
    movie_names = movies_df.loc[
        movies_df["movieId"].isin(movie_IDs), ["movieId", "title"]]
    # Merge the predictions
    movies_final = movies.merge(
        movie_names, 
        how="left", left_on="iid", right_on="movieId")

    return movies_final["title"]

#===================THIRD BANNER======================
st.markdown('<p class="big-font">Discover Movies Loved by Like-Minded Viewers</p>', unsafe_allow_html=True)

# User input for the user ID
user_id = st.number_input("Enter your user ID", min_value=1, value=1, step=1)
top_n = 5  # You can adjust this value or make it user-configurable

if st.button("Get Recommendations"):
    recommended_movie_titles = recommend_what_others_like(
        ratings_data=ratings_data, 
        movies_df=movies_df, 
        user_id=user_id, 
        top_n=top_n)

    if not recommended_movie_titles.empty:
        # Dynamically create columns based on the number of recommendations
        columns = st.columns(len(recommended_movie_titles))
        
        for i, movie_title in enumerate(recommended_movie_titles):
            with columns[i]:
                movie_details = get_movie_details(movie_title, movies_df)  # Fetch movie details
                st.markdown(f"**{movie_title}**")  # Display the movie title in bold
                
                # Display the movie poster if available
                if movie_details and movie_details['poster_url']:
                    st.image(movie_details['poster_url'], width=150)
                    with st.expander("View Details"):
                        st.write(f"**Description:** {movie_details['description']}")
                        st.write(f"**Duration:** {movie_details['duration']} minutes")
                else:
                    st.write("Poster not available.")
    else:
        st.write("No recommendations found.")
