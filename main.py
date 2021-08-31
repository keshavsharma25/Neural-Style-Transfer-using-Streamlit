import streamlit as st
from PIL import Image
import random
import style

#To set Page Title
st.set_page_config(page_title='Neural Style Transfer', layout='centered', initial_sidebar_state='auto')

# To hide footer
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Set Title
st.title("Pytorch Image Style Transfer")

img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'png']) 

style_name = st.sidebar.selectbox('Select Style',
                                  ('candy', 'mosaic', 'rain_princess', 'udnie', 'alan-scales'))

model_path = 'saved_models/'+style_name+'.pth'


if img_file is not None:
    input_image = Image.open(img_file)
    output_image = 'images/output-images/' + style_name + str(random.randint(0, 100)) + '.jpg'

    st.write("### Source Image : ")
    st.image(input_image, width=600)

    clicked = st.button("Stylize")

    if clicked:
        model = style.load_model(model_path)
        style.stylize(model, img_file, output_image)

        st.write("### Output Image : ")
        out_image = Image.open(output_image)
        st.image(out_image, width=600)
