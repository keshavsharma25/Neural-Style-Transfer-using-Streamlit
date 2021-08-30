import streamlit as st
from PIL import Image
import random
import style

st.title("Pytorch Image Style Transfer")  # Title

img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'png']) 

style_name = st.sidebar.selectbox('Select Style',
                                  ('candy', 'mosaic', 'rain_princess', 'udnie', 'alan-scales'))

model = 'saved_models/'+style_name+'.pth'

if img_file is not None:
    input_image = Image.open(img_file)
    output_image = 'images/output-images/' + style_name + str(random.randint(0, 100)) + '.jpg'

    st.write("### Source Image : ")
    st.image(input_image, width=720)

    clicked = st.button("Stylize")

    if clicked:
        model = style.load_model(model)
        style.stylize(model, img_file, output_image)

        st.write("### Output Image : ")
        out_image = Image.open(output_image)
        st.image(out_image, width=720)


# To hide footer
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)