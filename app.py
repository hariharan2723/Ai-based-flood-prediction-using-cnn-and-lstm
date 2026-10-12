import os
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

IMG_SIZE = 32
SEQ_LEN = 12
MODEL_PATH = "models/flood_cnn_lstm.pt"

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.lstm = nn.LSTM(1,32,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(96,32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32,1))

    def forward(self, image, rain):
        spatial = self.cnn(image).flatten(1)
        temporal,_ = self.lstm(rain)
        temporal = temporal[:,-1,:]
        return self.fc(torch.cat([spatial,temporal],1)).squeeze(1)

@st.cache_resource
def load_model():
    model = CNNLSTM()
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def make_image(water):
    rng = np.random.default_rng(7)
    img = rng.normal(.28,.05,(3,IMG_SIZE,IMG_SIZE)).clip(0,1)
    img[1] += .12
    size = int(5 + water/100*15)
    x0,y0 = 16-size//2,16-size//2
    img[2,x0:x0+size,y0:y0+size] += .55
    img[0,x0:x0+size,y0:y0+size] *= .55
    img[1,x0:x0+size,y0:y0+size] *= .75
    return np.clip(img,0,1).transpose(1,2,0)

def predict(model, rainfall, water):
    image = make_image(water)
    xi = torch.tensor(image.transpose(2,0,1)[None],dtype=torch.float32)
    xr = torch.tensor(np.array(rainfall,dtype=np.float32).reshape(1,SEQ_LEN,1)/100)
    with torch.no_grad():
        p = torch.sigmoid(model(xi,xr)).item()
    return image,p

st.set_page_config(page_title="AI Flood Prediction",page_icon="🌊",layout="wide")
st.title("🌊 AI-Based Flood Prediction using CNN and LSTM")
st.caption("PROJECT SECOND REVIEW • 50% WORKING OUTPUT DEMO")

if not os.path.exists(MODEL_PATH):
    st.error("Model file is missing. Please run train.py once.")
    st.stop()

model = load_model()

home, prediction, heatmap = st.tabs(["🏠 Home Page","📊 Prediction","🗺️ Flood Risk Heatmap"])

with home:
    st.subheader("Early Flood Warning System")
    st.write("Hybrid deep-learning demo combining satellite spatial information and rainfall time-series information.")
    a,b,c = st.columns(3)
    a.metric("CNN","Spatial Features")
    b.metric("LSTM","Temporal Features")
    c.metric("Output","Safe / Unsafe")
    st.info("Second Review Status: Data Collection ✓  Preprocessing ✓  CNN ✓  LSTM Training → In Progress  Hybrid Integration → Next Phase")

with prediction:
    st.subheader("Environmental Parameters")
    st.write("Enter the last 12 rainfall observations.")
    rainfall=[]
    cols=st.columns(4)
    defaults=[8,10,12,15,18,20,24,28,32,38,45,50]
    for i in range(SEQ_LEN):
        with cols[i%4]:
            rainfall.append(st.number_input(f"Rainfall T{i+1} (mm)",0.0,150.0,float(defaults[i]),1.0))
    water=st.slider("Satellite Water-Body Indicator (%)",0,100,65)

    if st.button("🚨 Predict Flood Risk",type="primary"):
        image,p=predict(model,rainfall,water)
        c1,c2=st.columns(2)
        with c1:
            st.image(image,caption="Satellite-like image input to CNN",width=300)
        with c2:
            st.metric("Risk Probability",f"{p*100:.1f}%")
            if p>=.5:
                st.error("Prediction: UNSAFE")
                st.warning("High flood-risk condition detected by the demo model.")
            else:
                st.success("Prediction: SAFE")
                st.info("Lower flood-risk condition detected by the demo model.")
            st.progress(p)

with heatmap:
    st.subheader("Flood Risk Heatmap")
    rng=np.random.default_rng(12)
    grid=np.clip(rng.random((8,8))*.45 + np.add.outer(np.linspace(0,.45,8),np.linspace(0,.45,8)),0,1)
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(grid)
    ax.set_title("Location-wise Flood Risk Heatmap")
    ax.set_xlabel("Longitude Grid")
    ax.set_ylabel("Latitude Grid")
    fig.colorbar(im,ax=ax,label="Risk")
    st.pyplot(fig)
    st.caption("Illustrative heatmap for the second-review demo; replace with geo-referenced prediction data in the final phase.")
