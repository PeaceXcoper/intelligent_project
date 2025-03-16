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





# ดาวน์โหลดไฟล์
folder_id = "1ggTHWuxw2-I5jcInLhGDH7Zy8VIGa5IQ"  # ใส่ Folder ID ของคุณ
url = f"https://drive.google.com/drive/folders/{folder_id}"

gdown.download_folder(url, output="downloaded_folder", quiet=False)

print("✅ ดาวน์โหลดโฟลเดอร์เสร็จแล้ว!"
    
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
    st.write("This is the main landing page of our app.")
    image1 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_1.png"
    image2 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_2.png"
    image3 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_3.png"
    image4 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_4.png"
    image5 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_5.png"
    image6 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_6.png"
    image7 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_7.png"
    image8 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_8.png"
    image9 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_9.png"
    image10 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_10.png"
    image11 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_11.png"
    image12 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_12.png"
    image13 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_13.png"
    image14 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/Mac_14.png"
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
    st.image(image1,  use_container_width=True) 
    st.write("1.นำเข้าไลบรารี<br>"
    "pandas : ใช้จัดการข้อมูลตาราง<br>"
    "numpy : ใช้จัดการข้อมูลเชิงตัวเลข<br>"
    "ast : ใช้แปลงข้อมูล string ให้เป็น list หรือ dictionary<br>"
    "TfidfVectorizer : ใช้แปลงข้อความเป็นเวกเตอร์แบบ TF-IDF เพื่อคำนวณความคล้ายคลึงกัน"
    ,unsafe_allow_html=True)
    
    st.image(image2,  use_container_width=True) 
    st.write("2. โหลดข้อมูลภาพยนตร์<br>"
    "credits.csv → ข้อมูลเกี่ยวกับ นักแสดงและทีมงาน ของหนัง<br>"
    "movies.csv → ข้อมูลเกี่ยวกับ ชื่อเรื่อง, แนวหนัง, เรื่องย่อ และอื่น ๆ"
    )
    
    st.image(image3,  use_container_width=True) 
    st.write("รวมข้อมูลจากสองไฟล์เข้าด้วยกันโดยใช้ชื่อภาพยนตร์ (title) เป็นคีย์")
    
    st.image(image4,  use_container_width=True) 
    st.write("เลือกเฉพาะคอลัมน์ที่จำเป็นสำหรับการแนะนำ")
    
    st.image(image5,  use_container_width=True) 
    st.markdown("แปลงข้อมูล JSON (ที่อยู่ในรูปแบบ String) ให้เป็น List ของชื่อ<br>"
    "ast.literal_eval() เพื่อแปลง String → List ของ Dictionary"
    ,unsafe_allow_html=True)

    
    st.image(image6,  use_container_width=True) 
    st.image(image7,  use_container_width=True) 
    st.write("convert() แปลงค่าทุกแถว ในคอลัมน์ genres และ keywords จาก String → List ของชื่อ")
    
    st.image(image8,  use_container_width=True) 
    st.write("ดึงเฉพาะ 3 นักแสดงหลัก จากข้อมูล JSON")
    
    st.image(image9,  use_container_width=True) 
    st.image(image10,  use_container_width=True) 
    st.write("ดึงเฉพาะ ชื่อผู้กำกับ จากข้อมูลทีมงาน (crew)")
    
    st.image(image11,  use_container_width=True) 
    st.write("รวม แนวภาพยนตร์ (genres), คำสำคัญ (keywords), นักแสดง (cast) และผู้กำกับ (crew) เข้าด้วยกันเป็น tags เพื่อใช้เป็นข้อมูลสำหรับการแนะนำ")
    st.image(image12,  use_container_width=True) 
    st.write("ใช้ TF-IDF Vectorizer ใช้วิเคราะห์ ความสำคัญของคำ ในแต่ละหนังตัดคำที่พบบ่อยออก เช่น the, and, is  และแปลงข้อความเป็นเวกเตอร์")
    st.image(image13,  use_container_width=True)
    st.write("คำนวณ Cosine Similarity เพื่อหาความคล้ายคลึงกันระหว่างภาพยนตร์"
    "ค่าจะอยู่ระหว่าง 0 - 1<br>"
    "1.0 = เหมือนกันเป๊ะ<br>"
    "0.0 = ไม่คล้ายกันเลย"
    ,unsafe_allow_html=True)
    st.image(image14,  use_container_width=True)
    st.write("•	 หาว่าหนังที่ผู้ใช้ต้องการอยู่ ดัชนีที่เท่าไร<br>"
    "•	คำนวณค่าความคล้ายกันของหนังทุกเรื่อง<br>"
    "•	เรียงลำดับจากมากไปน้อย<br>"
    "•	แสดงผล 10 หนังที่คล้ายกันมากที่สุด"
    ,unsafe_allow_html=True)
    
if selected == "Neural Network":
    st.title("Image fruit veg classify")
    image_path1 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code.png"
    image_path2 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code2.png"
    image_path3 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code3.png"
    image_path4 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code4.png"
    image_path5 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code5.png"
    image_path6 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code6.png"
    image_path7 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code7.png"
    image_path8 = "C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/pic code/code8.png"
    # อัปโหลดไฟล์รูปภาพ
    st.markdown(
    "<h1 style='font-size:24px;'>🔹 วิธีการทำงาน</h1>",
    unsafe_allow_html=True
    )
    st.write("1️⃣ อัปโหลดรูปข้อมูลผักผลไม้<br>"
    "2️⃣ ใช้ AI วิเคราะห์แยกประเภท ผัก ผลไม้และบอกชื่อ<br>"
    "3️⃣ แสดงค่า ค่าความแม่นยำ (accuracy)<br>"
    ,unsafe_allow_html=True)
    st.image(image_path1,  use_container_width=True) 
    st.write("เป็นการนำเข้าไลบรารีต่าง ๆ ที่จำเป็นสำหรับการสร้างและฝึกโมเดล Deep Learning โดยใช้ TensorFlow และ Keras รวมถึงการแสดงผลข้อมูลด้วย Matplotlib")
    
    st.image(image_path2,  use_container_width=True) 
    st.write("กำหนด path สำหรับชุดข้อมูลที่ใช้ใน training, การตรวจสอบความถูกต้อง, และการทดสอบ Deep Learning")
    
    st.image(image_path3,  use_container_width=True) 
    st.write("การเพิ่มข้อมูล และการโหลดชุดข้อมูล  สำหรับการฝึกและการตรวจสอบความถูกต้องของ Deep Learning ")
    
    st.image(image_path4,  use_container_width=True) 
    st.write("เป็นการสร้าง Deep Learningโดยใช้ EfficientNetB0 เป็น base model และเพิ่มเลเยอร์เพิ่มเติมเพื่อปรับให้เหมาะสมกับงานจำแนกประเภท")
    
    st.image(image_path5,  use_container_width=True) 
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

    
    st.image(image_path6,  use_container_width=True) 
    st.write("ใช้ Matplotlib เพื่อสร้างกราฟแสดงค่าความแม่นยำของโมเดลในระหว่างการฝึกและการตรวจสอบความถูกต้อง")
    
    st.image(image_path7,  use_container_width=True) 
    st.write("เป็นการปรับแต่งโมเดลเพิ่มเติม โดยการปลดล็อคเลเยอร์ของ base model เพื่อให้สามารถฝึกได้ และทำการฝึกโมเดลอีกครั้งด้วยอัตราการเรียนรู้ที่ต่ำกว่า")
    
    st.image(image_path8,  use_container_width=True) 
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
    model = load_model('C:/Users/peace/OneDrive/Desktop/Work/intell/Final-Project-Intelligent-Systems/Image_classify.keras')
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





















