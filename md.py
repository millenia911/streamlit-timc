from cmath import e
import streamlit as st
import numpy as np
import time
import random
import pandas as pd
from streamlit import session_state as states
from mlflow_trigger import trigger_mlflow_run


md = '''<iframe src="http://localhost:5151" 
width="800" 
height="600" frameborder="0" 
marginheight="0" marginwidth="0">Loading…</iframe>'''

op = open("frontpage.md", "r")
opening = op.read()

st.set_page_config(
     page_title="Model Evaluation",
     layout="wide",
     initial_sidebar_state="expanded",
 )

def init_states():
    states_list = ["form_submit", "result_ready", "input_error"]
    _v = False
    for _k in states_list:
        if _k not in states:
            states[_k] = _v

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
    with st.spinner('Wait for it...'):
        time.sleep(5)

def do_inference():
    with st.form("Do inference"):
        st.header("Inference Resources")
        tracking_uri = st.text_input("Tracking URI", value="http://192.168.103.67:5001/")
        uri = st.text_input("Repo URI")
        version = st.text_input("Version/Branch", value="main")
        model = st.text_input("Model URI")
        model_uri_origin = st.selectbox("Model URI Origin", 
                                        ["Blob-Storage",
                                        "MLFlow-Run"])
        dataset = st.text_input("Dataset URI")
        gpu = st.text_input("GPU Number", value="all")
        st.write("More Settings: ")
        entr_point = st.text_input("MLProject Entrypoint", value="test")
        experiment_name = st.text_input("MLFlow Experiment Name", value="test_inference")
        submitted = st.form_submit_button("Run🏃")
        states.form_submit = submitted
        _form_content = [uri, version, model, dataset, gpu, 
                         entr_point, experiment_name]
        if submitted:
            _filled = True
            for _c in _form_content:
                if len(_c) == 0:
                    _filled = False
            
            if _filled:
                states.input_error = False
                st.write("run inference from: ", uri)
                with st.spinner('Processing...'):
                    try:
                        _gif = st.image("https://c.tenor.com/7t8foti8FG8AAAAC/loading-screen-cat.gif")
                        run_id, run_status = trigger_mlflow_run( 
                                                        tracking_uri=tracking_uri,
                                                        uri=uri, 
                                                        entry_point=entr_point,
                                                        gpu=gpu, 
                                                        experiment_name=experiment_name,
                                                        version=version,
                                                        model=model,
                                                        dataset=dataset,
                                                        model_uri_origin=model_uri_origin)

                        st.write("Run ID: ", run_id)
                        st.write(run_status)
                        _gif.empty()
                        if run_status == "FINISHED":
                            states.result_ready = True
                    except Exception as e:
                        st.write(e)
                        run_id = None

                return run_id
            else:
                st.error("All field should be filled!")
                states.input_error = True
                return

def load_from_mlflow():
    with st.form("Load from MLflow"):
        st.header("MLflow Run Information")
        uri = st.text_input("Tracking URI")
        run_id = st.text_input("Run ID")
        arti_path = st.text_input("Artifact Path")
        dataset = st.text_input("Dataset URI")
        submitted = st.form_submit_button("Run🏃")
        states.form_submit = submitted
        if submitted:
            st.write("load from mlflow: ", run_id)

init_states()

with st.sidebar:
    options = ["Inference on New Dataset", "Load from MLflow Run"]
    mode = st.selectbox("Mode", options)
    if mode == options[0]:
        # result_run_id = do_inference()
        do_inference()
    elif mode == options[1]:
        load_from_mlflow()

if states.result_ready:
    render_result()

else:
    st.write("TODO: add env variable as input to set token(?)")
    st.markdown(opening, unsafe_allow_html=True)