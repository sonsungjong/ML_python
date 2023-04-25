import pickle
import streamlit as st
from tmdbv3api import Movie, TMDb

# streamlit 웹 실행 방법
# 터미널에
# streamlit run app.py

movie = Movie()
tmdb = TMDb()
# API 키
tmdb.api_key = 'd545ad803dad51e834f85a338b26806d'
tmdb.language = 'ko-KR'                 # tmdb 데이터를 한국어화

movies = pickle.load(open('movies.pickle', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pickle', 'rb'))

def get_recommendations(title):
    idx = movies[movies['title'] == title].index[0]                 # 영화의 index 값 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))                   # 순번을 주고, 리스트로 만듦
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)             # 내림차순 정렬
    sim_scores = sim_scores[1:11]                                               # 자기 자신을 제외하고 10개 슬라이싱
    movie_indices = [i[0] for i in sim_scores]                              # 추천 영화 목록 10개의 인덱스 정보 추출
    # 인덱스 정보를 통해 영화 제목 추출
    images = []
    titles = []
    for i in movie_indices:
        id = movies['id'].iloc[i]
        details = movie.details(id)                  # tmdb의 영화 디테일정보 가져오기

        # 이미지 경로가 null일 경우에 대한 예외처리
        image_path = details['poster_path']
        if image_path:
            image_path = 'https://image.tmdb.org/t/p/w500' + image_path
        else:
            image_path = 'no_image.jpg'

        images.append(image_path)
        titles.append(details['title'])
    return images, titles

# 와이드 화면
st.set_page_config(layout='wide')
st.header('Myflix')

movie_list = movies['title'].values
title = st.selectbox('Choose a movie you like', movie_list)             # 선택한 제목이 대입됨

# 해당 버튼을 누르면
if st.button('Recommend'):
    # 프로그레스 바
    with st.spinner('Please wait...'):
        # 버튼 동작
        images, titles = get_recommendations(title)
        idx = 0
        for i in range(0, 2):
            # 5개의 컬럼 생성
            cols = st.columns(5)
            for col in cols:
                col.image(images[idx])
                col.write(titles[idx])
                idx += 1
