import streamlit as st
import numpy as np
import time
import random
import pandas as pd

md = '''<iframe src="http://localhost:5151" 
width="800" 
height="600" frameborder="0" 
marginheight="0" marginwidth="0">Loading…</iframe>'''

op = open("frontpage.md", "r")
opening = op.read()

st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     layout="wide",
     initial_sidebar_state="expanded",
 )

def render_result():
    st.success('Inference is done!')
    with st.expander("Metrics", expanded=True):
        st.metric(label="mAP all classes", value=0.7)
        col_cls, col_map = st.columns(2)
        with col_cls:
            st.metric(label="Best classified class", value="Car")
            st.metric(label="Poorly classified class", value="Bike")
            st.metric(label="Average inference time", value="10ms/batch")
        with col_map:
            st.metric(label="with mAP", value=0.9)
            st.metric(label="with mAP", value=0.1)
            st.metric(label="with batch size", value=16)
        with st.container():
            df = pd.DataFrame(
                 np.random.randn(9, 4),
                 index=("car", "bike", "person", "cat", "dog",
                          "street_sign", "kite", "laptop", "boat"),
                 columns=("mAP","mAP50", "mAP90", 'accuracy'))
            st.dataframe(df)

    with st.expander("Results in FiftyOne"):
        st.write('''<form action="http://localhost:5151">
                        <input type="submit" value="Open full page" formtarget="_blank"/>
                    </form>''',
                    unsafe_allow_html=True)
        st.markdown(md, unsafe_allow_html=True)

def render_loading():
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'Progress {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.01)

with st.sidebar:
   with st.form("my_form"):
    st.header("Inference Resources")
    uri = st.text_input("Repo URI")
    token = st.text_input("Access token (for private repo)")
    model = st.text_input("Model URI")
    dataset = st.text_input("Dataset URI")
    version = st.text_input("Version")
    submitted = st.form_submit_button("Run🏃")
    if submitted:
        st.write("run inference from: ", uri)

if submitted:
    render_loading()
    render_result()
else:
    st.markdown(opening, unsafe_allow_html=True)