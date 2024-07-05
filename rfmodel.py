import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64

# Load the model and feature names
model = joblib.load('rf_model.sav')


original_feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

url = "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/energy_efficiency_data.csv"
data = pd.read_csv(url)
column_mapping = {
    'Relative_Compactness': 'X1',
    'Surface_Area': 'X2',
    'Wall_Area': 'X3',
    'Roof_Area': 'X4',
    'Overall_Height': 'X5',
    'Orientation': 'X6',
    'Glazing_Area': 'X7',
    'Glazing_Area_Distribution': 'X8'
}
data = data.rename(columns=column_mapping)


app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])  # two pages

if app_mode == 'Home' : 
    st.header('HL and CL PREDICTION')
    st.image('mypic.jpg')
    st.write("Pada applikasi ini kita bisa memprediksi seberapa besar heating load (beban pemanasan)  dan beban pendingin(cooling load) berdasarkan nilai setiap kolom yang kita berikan. Metode yang digunakan")
    st.markdown('Heating load :')
    st.write(' jumlah energi panas yang dibutuhkan untuk mempertahankan suhu dalam ruangan ketika suhu luar ruangan dibawah suhu dalam ruangan.')
    st.markdown('Cooling load  : ')
    st.write('jumlah energi panas yang perlu di hilangkan untuk mempertahankan suhu dalam ruangan yang dibutuhkan ketika suhu luar ruangan lebih tinggi dari suhu dalam ruangan..')
    st.markdown('---')
    st.markdown('Dataset:')
    st.write(data.head())
    st.markdown('Surface Area VS Heating Load')
    st.bar_chart(data[['X2', 'Heating_Load']].head(20))

else:
    # Define sidebar inputs
    Relative_Compactness = st.sidebar.slider('Relative_Compactness', min_value=0.62, max_value=0.99, step=0.01)
    Surface_Area = st.sidebar.slider('Surface_Area', min_value=300.0, max_value=900.0, step=0.01)
    Wall_Area = st.sidebar.slider('Wall_Area', min_value=100.0, max_value=900.0, step=1.0)
    Roof_Area = st.sidebar.slider('Roof_Area', min_value=100.0, max_value=600.0, step=0.1)
    Overall_Height = st.sidebar.slider('Overall_Height', min_value=3.0, max_value=10.0, step=0.1)
    Orientation = st.sidebar.slider('Orientation', min_value=2, max_value=5, step=1)
    Glazing_Area = st.sidebar.slider('Glazing_Area', min_value=0.0, max_value=0.4, step=0.01)
    Glazing_Area_Distribution = st.sidebar.slider('Glazing_Area_Distribution', min_value=0, max_value=5, step=1)


    # Prepare the input data
    input_data = pd.DataFrame({
        'X1': [Relative_Compactness],
        'X2': [Surface_Area],
        'X3': [Wall_Area],
        'X4': [Roof_Area],
        'X5': [Overall_Height],
        'X6': [Orientation],
        'X7': [Glazing_Area],
        'X8': [Glazing_Area_Distribution]
    })


# Predict Heating Load
    predictions = model.predict(input_data)
    st.write('Prediksi Heating Load dan Cooling Load')
    st.write('Berikut ini adalah contoh satu data actual dan prediksinya')
    st.markdown('Akurasi Prediksi adalah 98% dan R2 dan MSE tidak terlalu besar')
    st.markdown('lets say input datanya kita masukan sata data actual dan hasilnya pasti mendekati data actualnya')
    st.image('sampel1.jpg')
    st.write('Predicted Heating Load: 16.34')
    st.write('Predicted Cooling Load: 22.30')
    st.markdown('---')

    st.header('Hasil Prediksi')
    
    predicted_heating_load = predictions[0, 0]
    predicted_cooling_load = predictions[0, 1]

       # Get the actual values from the dataset
    actual_heating_load = data['Heating_Load'].iloc[0]  # Example: Fetching the first row's actual heating load
    actual_cooling_load = data['Cooling_Load'].iloc[0]  # Example: Fetching the first row's actual cooling load

    st.write(f'Predicted Heating Load: {predicted_heating_load:.2f}')
    st.write(f'Predicted Cooling Load: {predicted_cooling_load:.2f}')
   
    # Plot actual vs predicted values
    st.markdown('### Actual vs Predicted Heating Load')

    feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    data[['Predicted_Heating_Load', 'Predicted_Cooling_Load']] = model.predict(data[feature_cols])

   # Plot the data
    fig, ax = plt.subplots()
    ax.plot(data['Heating_Load'], label='Actual Heating Load')
    ax.plot(data['Predicted_Heating_Load'], label='Predicted Heating Load', linestyle='--')
    ax.plot(data['Cooling_Load'], label='Actual Cooling Load')
    ax.plot(data['Predicted_Cooling_Load'], label='Predicted Cooling Load', linestyle='--')
    ax.legend()
    st.pyplot(fig)

