import streamlit as st
import pandas as pd
import joblib

# Set konfigurasi halaman aplikasi
st.set_page_config(page_title="Prediksi Kondisi Lalu Lintas Menggunakan Algoritma Random Forest", layout="centered")
st.title("Sistem Prediksi Kondisi Lalu Lintas")
st.write("Prediksi tingkat kepadatan lalu lintas berdasarkan volume kendaraan")

# Memuat model
@st.cache_resource
def load_artifacts():
    model = joblib.load('traffic_rf_model.pkl')
    encoders = joblib.load('traffic_label_encoders.pkl')
    return model, encoders

try:
    model, encoders = load_artifacts()
    st.header("Masukkan Parameter Lalu Lintas")
    col1, col2 = st.columns(2)

    with col1:
        date = st.number_input("Tanggal (1-31)", min_value=1, max_value=31, value=15)
        day_options = encoders['Day of the week'].classes_
        day_choice = st.selectbox("Hari", options=day_options)
        time_options = encoders['Time'].classes_
        time_choice = st.selectbox("Waktu (Sesuai Data Asli)", options=time_options)

    with col2:
        car = st.number_input("Jumlah Mobil", min_value=0, value=25)
        bike = st.number_input("Jumlah Motor", min_value=0, value=5)
        bus = st.number_input("Jumlah Bus", min_value=0, value=2)
        truck = st.number_input("Jumlah Truk", min_value=0, value=1)


    total_vehicles = car + bike + bus + truck
    st.metric("Total Kendaraan Terhitung", total_vehicles)

    if st.button("Prediksi Kondisi Jalan"):
        encoded_day = encoders['Day of the week'].transform([day_choice])[0]
        encoded_time = encoders['Time'].transform([time_choice])[0]
       
        input_data = pd.DataFrame([{
            'Time': encoded_time,
            'Date': date,
            'Day of the week': encoded_day,
            'CarCount': car,
            'BikeCount': bike,
            'BusCount': bus,
            'TruckCount': truck,
            'Total': total_vehicles
        }])

        # Prediksi hasil
        prediction = model.predict(input_data)[0]
       
        # Mapping hasil balik ke teks label asli
        labels = {0: '🟢 Low (Lancar)', 1: '🔵 Normal (Ramai Lancar)', 2: '🟡 High (Padat)', 3: '🔴 Heavy (Macet Total)'}

        st.success(f"### Hasil Analisis: {labels.get(prediction, 'Tidak Diketahui')}")

except FileNotFoundError:
    st.error("Error: Pastikan file 'traffic_rf_model.pkl' dan 'traffic_label_encoders.pkl' berada di folder yang sama.")

