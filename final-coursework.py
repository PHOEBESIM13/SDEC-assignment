import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.image("https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.pixabay.com%2Fphoto%2F2020%2F06%2F30%2F14%2F06%2Fblossom-5356482__480.jpg&imgrefurl=https%3A%2F%2Fpixabay.com%2Fimages%2Fsearch%2Firis%2520flowers%2F&tbnid=IeTnL3Fd1BNihM&vet=12ahUKEwjespnRteb3AhW3_TgGHSB_BIAQMygGegUIARDhAQ..i&docid=gN2wLIXxhNcw1M&w=614&h=480&q=iris%20flower%20jpg&ved=2ahUKEwjespnRteb3AhW3_TgGHSB_BIAQMygGegUIARDhAQ")

st.sidebar.header('Select Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = pd.read_csv('https://raw.githubusercontent.com/PHOEBESIM13/SDEC-assignment/main/IRIS.csv')
X = iris.drop('species', axis=1)
Y = iris.species



clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(['Iris-setosa','Iris-versicolor', 'Iris-virginica'])

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
