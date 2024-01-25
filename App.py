import streamlit as st
import joblib
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder

# Load scaler and model
pt = joblib.load('pt.joblib')
clf = joblib.load('clf.joblib')
enc = joblib.load('enc.joblib')

# Page title
st.title('Loan Default Predictor 💰')

# Input
loan_amnt               = st.slider('loan_amnt', 500, 35000 )
funded_amnt             = st.slider('funded_amnt', 500, 35000)
funded_amnt_inv         = st.slider('funded_amnt_inv', 0,   35000)
term                    = st.radio('term', [' 36 months', ' 60 months'])
int_rate                = st.slider('int_rate', 6,      27)
installment             = st.slider('installment', 16,    1410)
grade                   = st.radio('grade', ['A', 'B', 'C',  'D', 'E', 'F',  'G'])
sub_grade               = st.selectbox('sub_grade', ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1',
       'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2',
       'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3',
       'G4', 'G5'])
home_ownership          = st.radio('home_ownership', ['NONE', 'ANY', 'OWN', 'MORTGAGE', 'RENT', 'OTHER'])
verification_status     = st.radio('verification_status', ['Not Verified', 'Source Verified', 'Verified'])
pymnt_plan              = st.radio('pymnt_plan', ['n', 'y'])
purpose                 = st.selectbox('purpose', ['car', 'credit_card', 'debt_consolidation', 'educational',
       'home_improvement', 'house', 'major_purchase', 'medical', 'moving',
       'other', 'renewable_energy', 'small_business', 'vacation',
       'wedding'])
dti                     = st.slider('dti', 0,      40)
revol_bal               = st.slider('revol_bal', 0, 2568995)
initial_list_status     = st.radio('initial_list_status', ['f', 'w'])
out_prncp               = st.slider('out_prncp', 0,   32161)
out_prncp_inv           = st.slider('out_prncp_inv', 0,   32161)
total_pymnt             = st.slider('total_pymnt', 0,   57778)
total_pymnt_inv         = st.slider('total_pymnt_inv', 0,   57778)
total_rec_prncp         = st.slider('total_rec_prncp', 0,   35001)
total_rec_late_fee      = st.slider('total_rec_late_fee', 0,     359)
recoveries              = st.slider('recoveries', 0,   33521)
collection_recovery_fee = st.slider('collection_recovery_fee', 0,    7003)
last_pymnt_amnt         = st.slider('last_pymnt_amnt', 0,   36235)

def predict():
    # Answer to dictionary
    data = {
    'loan_amnt': loan_amnt,
    'funded_amnt': funded_amnt,
    'funded_amnt_inv': funded_amnt_inv,
    'term': term,
    'int_rate': int_rate,
    'installment': installment,
    'grade': grade,
    'sub_grade': sub_grade,
    'home_ownership': home_ownership,
    'verification_status': verification_status,
    'pymnt_plan': pymnt_plan,
    'purpose': purpose,
    'dti': dti,
    'revol_bal': revol_bal,
    'initial_list_status': initial_list_status,
    'out_prncp': out_prncp,
    'out_prncp_inv': out_prncp_inv,
    'total_pymnt': total_pymnt,
    'total_pymnt_inv': total_pymnt_inv,
    'total_rec_prncp': total_rec_prncp,
    'total_rec_late_fee': total_rec_late_fee,
    'recoveries': recoveries,
    'collection_recovery_fee': collection_recovery_fee,
    'last_pymnt_amnt': last_pymnt_amnt
    }
    # Creating a DataFrame
    df = pd.DataFrame([data]).infer_objects()
    # Scale features
    numeric = df.select_dtypes(exclude=['object']).columns.values
    df[numeric] = pt.transform(df[numeric])
    # Ordinal encode
    obj = df.select_dtypes(include=['object']).columns.values
    df[obj] = enc.transform(df[obj])
    # Predict
    pred = clf.predict(df.values)[0]

    if pred == 0:
        st.error('Loan is a default ❌')
    else:
        st.success('Loan is normal ✅')

st.button('Predict', on_click=predict)
