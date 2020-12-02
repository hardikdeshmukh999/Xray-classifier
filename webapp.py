import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = 'cnn_pneu_vamp_model.h5'
  model_loader = load_model(fp)
  return model_loader

model = loading_model()
st.write("""
# X-Ray Classification (Pneumonia/Normal)
""")

file = st.file_uploader("Upload X-Ray",type=["jpg","png"])

def preprocess_predict(image_data,cnn):

  # Preprocessing the image
  hardik_img = image_data.resize(500,500)
  hardik_img = ImageOps.grayscale(hardik_img )
  
  pp_hardik_img = np.asarray(hardik_img)
  pp_hardik_img = pp_hardik_img/255
  pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

  #predict
  hardik_preds= cnn.predict(pp_hardik_img)

  if hardik_preds>= 0.5: 
    out = ('I am {:.2%} percent confirmed that this is a Pneumonia case'.format(hardik_preds[0][0]))
    
  else: 
    out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-hardik_preds[0][0]))

  return out


if file is None:
  st.text("Oops! that doesn't look like an image. Try again.")

else:
  image = Image.open(file)
  st.image(image,use_column_width=True)
  prediction = preprocess_predict(image,model)
  st.success(prediction)
  
