import streamlit as st
import os

import time
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer as ct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
from matplotlib import image


# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "laptop")

cur_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

st.markdown("<h1 style='text-align: center; color: Black;'>Laptop Price Prediction</h1>", unsafe_allow_html=True)


# Description
PAGE_TITLE = "LAPTOP PRICE RECOMMENDATION"
PAGE_ICON = ":computer:"




data_path = os.path.join(dir_of_interest,"df.pkl")
ml_data = os.path.join(dir_of_interest,  "rf.pkl")
model = os.path.join(dir_of_interest, "model.pkl")

lap = pickle.load(open(data_path, 'rb'))
rf = pickle.load(open(ml_data, 'rb'))

df = pd.DataFrame(lap)



# st.dataframe(df)
# ----------------------------------------ML section------------------------------------------
features = ["brand", "processor", "ram", "os", "Storage"]
f = df[["brand", "processor", "ram", "os", "Storage"]]
y = np.log(df['MRP'])
X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=47)
step1 = ct(transformers=[
    ('encoder',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,4])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)
# -----------------------------------Input Section---------------------------------------------
brand = st.selectbox("Enter the Brand:label: ", df["brand"].unique())
processor = st.selectbox("Enter the Processor:floppy_disk: ", df["processor"].unique())
ram = st.selectbox("Enter the RAM:minidisc: ", df["ram"].unique())
os = st.selectbox("Enter the Operating System:gear: ", df["os"].unique())
Storage = st.selectbox("Enter the Storage:file_cabinet:", df["Storage"].unique())

butt = st.button("Check:moneybag:")

if butt:
    
    query = np.array([brand, processor, ram, os, Storage])
    query = query.reshape(1, -1)
    p = pipe.predict(query)[0]
    result = np.exp(p)
    st.subheader("The Laptop Price is: "":green[₹{}]".format(result.round(2)))




    










