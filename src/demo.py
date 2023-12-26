# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# predict module
from predict import predict


page_options = ["Retail Price Optimisation","Exploratory Data Analysis"]


page_selection = st.sidebar.selectbox("Choose Option", page_options)
if page_selection == "Retail Price Optimisation":
    print("Retail Price Optimisation")
    # Header contents
    st.write('# Retail Price Prediction')
    # Recommender System algorithm selection
    sys = st.radio("Select an algorithm",
                    ('Random Forest Regression',
                    'Collaborative Based Filtering'))
    st.write('### Enter features of the product ')

    # User-based preferences
    col1, col2 = st.columns(2)
    with col1:
        product_category_name = st.selectbox("Product category name", ['bed_bath_table', 'garden_tools', 'consoles_games', 'health_beauty', 'cool_stuff', 'perfumery', 'computers_accessories', 'watches_gifts', 'furniture_decor'])
        # month = st.selectbox("Month", ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12'])
        qty = st.number_input("Product quantity", min_value=0, value=0, step=1)
        unit_price = st.number_input("Unit price", min_value=0.0, value=0.0, step=10.0)
        freight_price = st.number_input("Freight price", min_value=0.0, value=13.0, step=1.0)
        customers = st.number_input("Number of customers", min_value=0, value=50, step=1)
    with col2:
        lag_price = st.number_input("Lag price", min_value=0.0, value=45.0, step=1.0)
        volume = st.number_input("Volume", min_value=0, value=3000, step=100)
        product_weight_g = st.number_input("Product weight", min_value=0, value=300, step=10)
        product_score = st.number_input("Product score", min_value=0.0, value=4.0, step=0.5)  
    
    if "df" not in st.session_state:
        st.session_state.df=pd.read_csv("/Users/lggvu/Programming/Business-Analysis/eda/retail_price.csv") 

    st.session_state.df["product_category_name"].iloc[0] = product_category_name
    st.session_state.df["qty"].iloc[0] = qty
    st.session_state.df["unit_price"].iloc[0] = unit_price
    st.session_state.df["freight_price"].iloc[0] = freight_price
    st.session_state.df["customers"].iloc[0] = customers
    st.session_state.df["lag_price"].iloc[0] = lag_price  
    

    # Perform top-10 movie recommendation generation
    if sys == 'Random Forest Regression':
        print("Random Forest Regression")
        # st.button("Predict", type="primary")
        if st.button("Predict"):
            try:
                with st.spinner('Crunching the numbers...'):
                    pred=predict(st.session_state.df[0:1])
                st.markdown(f'## Predicted Price: :green[{"{:.3f}".format(pred)}] USD')
                # st.subheader(pred)
            except:
                st.error("Oops! Looks like this algorithm does't work.\
                          We'll need to fix it!")


    if sys == 'Collaborative Based Filtering':
        print("Collaborative Based Filtering")
        if st.button("Predict"):
            print("hello")

# -------------------------------------------------------------------

# ------------- SAFE FOR ALTERING/EXTENSION -------------------
if page_selection == "Exploratory Data Analysis":
    st.title("EDA Dashboard")
    st.write("Some data insights")
    st.markdown('<iframe title="draft" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=4c6e2ae1-83ae-4de2-8e52-8f228ab4f256&autoAuth=true&ctid=06f1b89f-07e8-464f-b408-ec1b45703f31" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)
