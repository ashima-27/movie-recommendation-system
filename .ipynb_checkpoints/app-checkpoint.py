import streamlit as st
import json
from Classifier import KNearestNeighbours
from operator import itemgetter
from PIL import Image

# for speech
import speech_recognition as sr
import subprocess
import pywhatkit
import pyttsx3
import webbrowser
from playsound import playsound
from gtts import gTTS

# r= sr.Recognizer()  

# def SpeakText(command):
    

#     engine = pyttsx3.init()
#     engine.say(command)
#     engine.runAndWait()
    
# with sr.Microphone() as source2:
#     r.adjust_for_ambient_noise(source2,duration=0.2)
#     audio2 =r.listen(source2)

#     MyText =r.recognize_google(audio2)
#     MyText =MyText.lower()

#     SpeakText(MyText)

            
      
image=Image.open("image.jpg")
# img = Image.open('https://i0.wp.com/sierraleoneislamicweb.com/wp-content/uploads/2020/03/short-film-icon.png?fit=250%2C250&ssl=1')
st.set_page_config(page_title="Movie Recommender")
st.image(image,  width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        
# Load data and movies list from corresponding JSON files
with open(r'data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open(r'titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)

hide_menu_style ="""
<style>
#MainMenu {visibility: hidden;}

footer {visibility :hidden;}
</style>
"""
st.markdown(hide_menu_style,unsafe_allow_html=True)
def knn(test_point, k):
    # Create dummy target variable for the KNN Classifier
    target = [0 for item in movie_titles]
    # Instantiate object for the Classifier
    model = KNearestNeighbours(data, target, test_point, k=k)
    # Run the algorithm
    model.fit()
    # Distances to most distant movie
    max_dist = sorted(model.distances, key=itemgetter(0))[-1]
    # Print list of 10 recommendations < Change value of k for a different number >
    table = list()
    for i in model.indices:
        # Returns back movie title and imdb link
        table.append([movie_titles[i][0], movie_titles[i][2]])
    return table

if __name__ == '__main__':
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
              'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

    movies = [title[0] for title in movie_titles]
    st.header('Movie Recommendation System') 
    

    apps = ['--Select--', 'Movie based', 'Genres based']   
    app_options = st.selectbox('Select application:', apps)

      
    
    if app_options == 'Movie based' :
        
        movie_select = st.selectbox('Select movie:', ['--Select--'] + movies)
        if movie_select == '--Select--':
            st.write('Select a movie')
        else:
            n = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
            genres = data[movies.index(movie_select)]
            test_point = genres
            table = knn(test_point, n)
            for movie, link  in table:
                # Displays movie title with link to imdb
                st.markdown(f"[{movie}]({link})")
    elif  app_options == apps[2]:
        options = st.multiselect('Select genres:', genres)
        if options:
            imdb_score = st.slider('IMDb score:', 1, 10, 8)
            n = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
            test_point = [1 if genre in options else 0 for genre in genres]
            test_point.append(imdb_score)
            table = knn(test_point, n)
            for movie, link in table:
                # Displays movie title with link to imdb
                st.markdown(f"[{movie}]({link})")

        else:
                st.write("This is a simple Movie Recommender application. "
                        "You can select the genres and change the IMDb score.")

    else:
        st.write('Select option')
     