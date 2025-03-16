import streamlit as st
from tensorflow import keras
from streamlit_option_menu import option_menu
import pandas as pd
import requests
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import gdown

file_id2 = "1DieIObGn0_1-bpsy5Lum497vl3EY9K5O"
output2 = "Image_classify.keras"  # ชื่อไฟล์ที่ต้องการบันทึก

# ดาวน์โหลดไฟล์จาก Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id2}", output2, quiet=False, verify=False)

# ใส่ FILE_ID ที่ได้จาก Google Drive
file_id = "1ggTHWuxw2-I5jcInLhGDH7Zy8VIGa5IQ"
output = "movie_data.pkl"  # เปลี่ยนเป็นชื่อไฟล์ที่ต้องการเซฟ

# ดาวน์โหลดไฟล์
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

print(f"Downloaded {output} successfully!")
with open("movie_data.pkl", "rb") as f:
    model = pickle.load(f)

EXAMPLE_NO = 1

def streamlit_menu():
    # ใช้ sidebar menu เท่านั้น
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Machine Learning", "Neural Network", "Demo Machine Learning", "Demo Neural Network"],  # เพิ่มหน้า About
                # icons=["house", "book", "envelope", "info-circle"],  # เพิ่มไอคอน
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

selected = streamlit_menu()

if selected == "Machine Learning":
    st.title("Welcome to the Machine Learning Page")
    
    
    
    
    st.markdown(
    "<h1 style='font-size:24px;'>🔹 วิธีการทำงาน</h1>",
    unsafe_allow_html=True
    )
    st.markdown("""
    1️⃣ โหลดข้อมูลภาพยนตร์<br>
    2️⃣ แปลงข้อมูลให้พร้อมใช้งาน<br>
    3️⃣ ใช้ AI วิเคราะห์ความคล้ายกัน<br>
    4️⃣ แนะนำหนังที่คล้ายกัน 10 เรื่อง
    """,unsafe_allow_html=True)
   
    code = '''import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
        '''
    st.code(code, language="python")
    
    st.write("1.นำเข้าไลบรารี<br>"
    "pandas : ใช้จัดการข้อมูลตาราง<br>"
    "numpy : ใช้จัดการข้อมูลเชิงตัวเลข<br>"
    "ast : ใช้แปลงข้อมูล string ให้เป็น list หรือ dictionary<br>"
    "TfidfVectorizer : ใช้แปลงข้อความเป็นเวกเตอร์แบบ TF-IDF เพื่อคำนวณความคล้ายคลึงกัน"
    ,unsafe_allow_html=True)
    
    code = '''credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')
        '''
    st.code(code, language="python")
    st.write("2. โหลดข้อมูลภาพยนตร์<br>"
    "credits.csv → ข้อมูลเกี่ยวกับ นักแสดงและทีมงาน ของหนัง<br>"
    "movies.csv → ข้อมูลเกี่ยวกับ ชื่อเรื่อง, แนวหนัง, เรื่องย่อ และอื่น ๆ"
    )
    
    code = '''movies = movies.merge(credits, left_on='title', right_on='title')
        '''
    st.code(code, language="python")
    st.write("รวมข้อมูลจากสองไฟล์เข้าด้วยกันโดยใช้ชื่อภาพยนตร์ (title) เป็นคีย์")
    
    code = '''movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        '''
    st.code(code, language="python")
    st.write("เลือกเฉพาะคอลัมน์ที่จำเป็นสำหรับการแนะนำ")
    
    code = '''def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
        '''
    st.code(code, language="python")
    st.markdown("แปลงข้อมูล JSON (ที่อยู่ในรูปแบบ String) ให้เป็น List ของชื่อ<br>"
    "ast.literal_eval() เพื่อแปลง String → List ของ Dictionary"
    ,unsafe_allow_html=True)

    
    code = '''movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
        '''
    st.code(code, language="python")
    st.write("convert() แปลงค่าทุกแถว ในคอลัมน์ genres และ keywords จาก String → List ของชื่อ")
    
    code = '''movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
        '''
    st.code(code, language="python")
    st.write("ดึงเฉพาะ 3 นักแสดงหลัก จากข้อมูล JSON")
    
    code = '''movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        '''
    st.code(code, language="python")
    st.write("ดึงเฉพาะ ชื่อผู้กำกับ จากข้อมูลทีมงาน (crew)")
    
    code = '''movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
        '''
    st.code(code, language="python")
    st.write("รวม แนวภาพยนตร์ (genres), คำสำคัญ (keywords), นักแสดง (cast) และผู้กำกับ (crew) เข้าด้วยกันเป็น tags เพื่อใช้เป็นข้อมูลสำหรับการแนะนำ")
    
    code = '''from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
        '''
    st.code(code, language="python")
    st.write("ใช้ TF-IDF Vectorizer ใช้วิเคราะห์ ความสำคัญของคำ ในแต่ละหนังตัดคำที่พบบ่อยออก เช่น the, and, is  และแปลงข้อความเป็นเวกเตอร์")
    
    code = '''from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        '''
    st.code(code, language="python")
    st.write("คำนวณ Cosine Similarity เพื่อหาความคล้ายคลึงกันระหว่างภาพยนตร์"
    "ค่าจะอยู่ระหว่าง 0 - 1<br>"
    "1.0 = เหมือนกันเป๊ะ<br>"
    "0.0 = ไม่คล้ายกันเลย"
    ,unsafe_allow_html=True)
    
    code = '''def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices
        '''
    st.code(code, language="python")
    st.write("•	 หาว่าหนังที่ผู้ใช้ต้องการอยู่ ดัชนีที่เท่าไร<br>"
    "•	คำนวณค่าความคล้ายกันของหนังทุกเรื่อง<br>"
    "•	เรียงลำดับจากมากไปน้อย<br>"
    "•	แสดงผล 10 หนังที่คล้ายกันมากที่สุด"
    ,unsafe_allow_html=True)
    
    
    
    
    
if selected == "Neural Network":
    st.title("Image fruit veg classify")

    # อัปโหลดไฟล์รูปภาพ
    st.markdown(
    "<h1 style='font-size:24px;'>🔹 วิธีการทำงาน</h1>",
    unsafe_allow_html=True
    )
    st.write("1️⃣ อัปโหลดรูปข้อมูลผักผลไม้<br>"
    "2️⃣ ใช้ AI วิเคราะห์แยกประเภท ผัก ผลไม้และบอกชื่อ<br>"
    "3️⃣ แสดงค่า ค่าความแม่นยำ (accuracy)<br>"
    ,unsafe_allow_html=True)
     
    code = '''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
        '''
    st.code(code, language="python")
    st.write("เป็นการนำเข้าไลบรารีต่าง ๆ ที่จำเป็นสำหรับการสร้างและฝึกโมเดล Deep Learning โดยใช้ TensorFlow และ Keras รวมถึงการแสดงผลข้อมูลด้วย Matplotlib")
    
    code = '''data_train_path = 'Fruits_Vegetables/train
data_val_path = 'Fruits_Vegetables/validation
data_test_path = 'Fruits_Vegetables/test
        '''
    st.code(code, language="python")
    
    st.write("กำหนด path สำหรับชุดข้อมูลที่ใช้ใน training, การตรวจสอบความถูกต้อง, และการทดสอบ Deep Learning")
    code = '''data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

dataset_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=img_size,
    batch_size=batch_size
)

dataset_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    shuffle=False,
    image_size=img_size,
    batch_size=batch_size
)
        '''
    st.code(code, language="python")
    
    st.write("การเพิ่มข้อมูล และการโหลดชุดข้อมูล  สำหรับการฝึกและการตรวจสอบความถูกต้องของ Deep Learning ")
    code = '''base_model = EfficientNetB0(include_top=False, input_shape=(180, 180, 3), weights='imagenet')
base_model.trainable = False  # ล็อคค่าเริ่มต้นของโมเดล

# สร้างโมเดล
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(dataset_train.class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)
        '''
    st.code(code, language="python")
   
    st.write("เป็นการสร้าง Deep Learningโดยใช้ EfficientNetB0 เป็น base model และเพิ่มเลเยอร์เพิ่มเติมเพื่อปรับให้เหมาะสมกับงานจำแนกประเภท")
    code = '''model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ฝึกโมเดล
epochs = 50
history = model.fit(dataset_train, validation_data=dataset_val, epochs=epochs)
        '''
    st.code(code, language="python")
    
    st.markdown(
    "**โค้ดนี้เป็นการคอมไพล์และฝึก Model**<br>"
    "optimizer=keras.optimizers.Adam(learning_rate=0.001):<br> ใช้ตัวปรับค่า Adam พร้อมกับกำหนด learning rate เป็น 0.001<br>"
    "<br>"
    "loss='sparse_categorical_crossentropy':<br> ใช้ loss function แบบ sparse categorical crossentropy<br>"
    "<br>"
    "metrics=['accuracy']: กำหนด accuracy ระหว่างการฝึก<br>"
    "epochs = 50: กำหนดจำนวนรอบการฝึก เป็น 50 รอบ<br>"
    "history = model.fit(dataset_train, validation_data=dataset_val, epochs=epochs):<br> ฝึกโมเดลโดยใช้ชุดข้อมูลฝึก dataset_train และตรวจสอบความถูกต้องด้วยชุดข้อมูลตรวจสอบความถูกต้อง dataset_val",
    unsafe_allow_html=True)

    
    code = '''
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
        '''
    st.code(code, language="python")
    st.write("ใช้ Matplotlib เพื่อสร้างกราฟแสดงค่าความแม่นยำของโมเดลในระหว่างการฝึกและการตรวจสอบความถูกต้อง")
    
    code = '''# ปลดล็อคบางชั้นของ EfficientNetB0 และฝึกต่อ (Fine-tuning)
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 5
history_fine = model.fit(dataset_train, validation_data=dataset_val, epochs=fine_tune_epochs)
        '''
    st.code(code, language="python")
    st.write("เป็นการปรับแต่งโมเดลเพิ่มเติม โดยการปลดล็อคเลเยอร์ของ base model เพื่อให้สามารถฝึกได้ และทำการฝึกโมเดลอีกครั้งด้วยอัตราการเรียนรู้ที่ต่ำกว่า")
    
    code = '''model.save("fruit_veg_classifier.h5")
        '''
    st.code(code, language="python")
    st.write("เป็นการบันทึกโมเดลที่คุณฝึกเสร็จแล้วลงในไฟล์ชื่อ fruit_veg_classifier.h5 โดยใช้ฟังก์ชัน save ของ Keras")
    
    
    
    
if selected == "Demo Machine Learning":
    st.title("Movie Recommendation System")
    # Load the processed data and similarity matrix
    with open('movie_data.pkl', 'rb') as file:
        movies, cosine_sim = pickle.load(file)

    # Function to get movie recommendations
    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get top 10 similar movies
        movie_indices = [i[0] for i in sim_scores]
        return movies[['title', 'movie_id']].iloc[movie_indices]

    # Fetch movie poster from TMDB API
    def fetch_poster(movie_id):
        api_key = '7b995d3c6fd91a2284b4ad8cb390c7b8'  # Replace with your TMDB API key
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
        
        response = requests.get(url)
        data = response.json()
        print(data)
        poster_path = data.get('poster_path', 'default_value')
        
        
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
        else:
            return 'default_poster_path'  # กำหนดค่าเริ่มต้นเมื่อไม่พบ poster_path

    selected_movie = st.selectbox("Select a movie:", movies['title'].values)

    if st.button('Recommend'):
        recommendations = get_recommendations(selected_movie)
        st.write("Top 10 recommended movies:")

        # Create a 2x5 grid layout
        for i in range(0, 10, 5):  # Loop over rows (2 rows, 5 movies each)
            cols = st.columns(5)  # Create 5 columns for each row
            for col, j in zip(cols, range(i, i+5)):
                if j < len(recommendations):
                    movie_title = recommendations.iloc[j]['title']
                    movie_id = recommendations.iloc[j]['movie_id']
                    poster_url = fetch_poster(movie_id)
                    with col:
                        st.image(poster_url, width=130)
                        st.write(movie_title)
if selected == "Demo Neural Network":  
    
    st.title("Demo Neural Network")
    st.header('Image Classification Model')
    



    # โหลดโมเดล Keras
    model = tf.keras.models.load_model(output2)
    data_cat = ['apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon']
    img_height = 180
    img_width = 180

    # ใช้ st.file_uploader เพื่ออัปโหลดไฟล์
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # โหลดและแสดงภาพที่อัปโหลด
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, 0)

        # ทำการพยากรณ์
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        # แสดงผลลัพธ์
        st.image(image_load, width=200)
        st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
        st.write('With accuracy of ' + str(np.max(score) * 100) + '%')





















