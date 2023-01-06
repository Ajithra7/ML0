import streamlit as st
from PIL import Image
from food_classifier import classifier


col1 = st.columns(1)

st.sidebar.title("Food image classifier.")

buffer = st.sidebar.file_uploader("Select an image.", type=['png', 'jpg', 'jpeg'])

if buffer :
    file = buffer.read()
    img = Image.open(buffer)

    st.image(img, use_column_width='always')
    if st.sidebar.button("Classify"):           
        with st.spinner('Processing...'):
            class_name, score = classifier(img)
            if score < 95.0:
                st.sidebar.success("Unrecognized")
            else:
                st.sidebar.success("{} \t(confidence = {:.2f})".format(class_name, score))
