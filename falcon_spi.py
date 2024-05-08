# Cuurent most latest

from re import X
from xml.sax.saxutils import XMLFilterBase
import streamlit as st
import pickle
import numpy as np

#------------------------------------------------------------------

def load_model():
    with open('svr_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_model2():
    with open('svr_cleanroom.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_model3():
    with open('svr_hvac.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_model4():
    with open('svr_elec.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

#-------------------------------------------------------------------

# START FIG SESSION 
## Mechanical 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def load_data():
    return pd.read_excel('Falcon Schedule - ML Mech - RUN ALGO.xlsx')

def preprocess_data(dataset):
    x = dataset.iloc[:, :-2].values
    y = dataset.iloc[:,-2].values

    y = y.reshape(len(y),1)

    ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[4])],remainder = 'passthrough')
    x = np.array(ct.fit_transform(x))

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    return x,y
    
def train_model(x,y):
    regressor = SVR(kernel = 'rbf')
    regressor.fit(x, y)
    return regressor

def calculate_permutation_importance (regressor,x,y):
    result = permutation_importance(regressor,x,y,n_repeats=10, random_state=42)
    return result.importances_mean

feature_names1 = ['submittal closure rate','machinery','IFF issuance rate','manpower','fabrication variance','safety incident','rfi closure rate','material PO issuance rate','2 wks ahead SPI']

def plot_feature_importance1(feature_names, importance_scores):
    # Convert importance_scores to list and shift the scores to the Right by 1 position
    importance_scores_list = list(importance_scores)
    importance_scores_list.append(importance_scores_list.pop(0))
    
    # Convert back to numpy array
    shifted_importance_scores = np.array(importance_scores_list)
    
    # Identify top 3 features
    top_3_indices = np.argsort(shifted_importance_scores)[-3:]
    
    # Prepare colors: 'red' for top 3, 'blue' for the rest
    colors = ['red' if i in top_3_indices else 'blue' for i in range(len(feature_names))]

    y_pos = np.arange(len(feature_names))

    fig, ax = plt.subplots()
    ax.bar(y_pos, shifted_importance_scores, align='center', color=colors)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(feature_names, rotation='vertical')
    ax.set_ylabel('Importance Score')
    ax.set_xlabel('Features')
    ax.set_title('Feature Importance Scores from AI Model (Mechanical Package)')
    plt.tight_layout()  # To ensure labels don't overlap

    # Pass the Matplotlib figure to st.pyplot
    st.pyplot(fig)
    

#---------------------------------------------------------------------
## Cleanroom
def load_data2():
    return pd.read_excel('Falcon Schedule - ML Cleanroom - RUN ALGO.xlsx')

def preprocess_data2(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:,-1].values

    y = y.reshape(len(y),1)

    ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[4])],remainder = 'passthrough')
    x = np.array(ct.fit_transform(x))

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    return x,y
    
feature_names2 = ['material submittal closure rate','SOC Submission','Safety Induction','manpower','safety incident','rfi closure rate','material PO issuance rate','2 weeks ahead acum SPI']

def plot_feature_importance2(feature_names, importance_scores):
    
    importance_scores_list = list(importance_scores)
    importance_scores_list.append(importance_scores_list.pop(0))
    shifted_importance_scores = np.array(importance_scores_list)
    
    top_3_indices = np.argsort(shifted_importance_scores)[-3:]
    
    colors = ['red' if i in top_3_indices else 'blue' for i in range(len(feature_names))]

    y_pos = np.arange(len(feature_names))
    
    fig, ax = plt.subplots()
    ax.bar(y_pos, shifted_importance_scores, align='center', color=colors)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(feature_names, rotation='vertical')
    ax.set_ylabel('Importance Score')
    ax.set_xlabel('Features')
    ax.set_title('Feature Importance Scores from AI Model (Cleanroom Package)')
    plt.tight_layout()  

    st.pyplot(fig)
    

#----------------------------------------------------------------------
## HVAC
def load_data3():
    return pd.read_excel('Falcon Schedule - ML HVAC - RUN ALGO.xlsx')

def preprocess_data3(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:,-1].values

    y = y.reshape(len(y),1)

    ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[5])],remainder = 'passthrough')
    x = np.array(ct.fit_transform(x))

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    return x,y
    
feature_names3 = ['material submittal closure rate','machinery','IFF Issuance rate','safety induction','manpower','safety incident','material arrival non-risky rate','2wks ahead acum SPI']

def plot_feature_importance3(feature_names, importance_scores):
    
    importance_scores_list = list(importance_scores)
    importance_scores_list.append(importance_scores_list.pop(0))
    shifted_importance_scores = np.array(importance_scores_list)
    
    top_3_indices = np.argsort(shifted_importance_scores)[-3:]
    
    colors = ['red' if i in top_3_indices else 'blue' for i in range(len(feature_names))]

    y_pos = np.arange(len(feature_names))
    
    fig, ax = plt.subplots()
    ax.bar(y_pos, shifted_importance_scores, align='center', color=colors)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(feature_names, rotation='vertical')
    ax.set_ylabel('Importance Score')
    ax.set_xlabel('Features')
    ax.set_title('Feature Importance Scores from AI Model (HVAC Package)')
    plt.tight_layout()  

    st.pyplot(fig)

#------------------------------------------------------------------------
## Electrical
def load_data4():
    return pd.read_excel('Falcon Schedule - ML Elec - RUN ALGO.xlsx')

def preprocess_data4(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:,-1].values

    y = y.reshape(len(y),1)

    ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[4])],remainder = 'passthrough')
    x = np.array(ct.fit_transform(x))

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    return x,y
    
feature_names4 = ['material submittal closure rate','IFF Issuance rate','SOC Submission','manpower','safety incident','rfi closure rate','2wks ahead acum SPI']

def plot_feature_importance4(feature_names, importance_scores):
    
    importance_scores_list = list(importance_scores)
    importance_scores_list.append(importance_scores_list.pop(0))
    shifted_importance_scores = np.array(importance_scores_list)
    
    top_3_indices = np.argsort(shifted_importance_scores)[-3:]
    
    colors = ['red' if i in top_3_indices else 'blue' for i in range(len(feature_names))]

    y_pos = np.arange(len(feature_names))
    
    fig, ax = plt.subplots()
    ax.bar(y_pos, shifted_importance_scores, align='center', color=colors)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(feature_names, rotation='vertical')
    ax.set_ylabel('Importance Score')
    ax.set_xlabel('Features')
    ax.set_title('Feature Importance Scores from AI Model (Electrical Package)')
    plt.tight_layout()  

    st.pyplot(fig)


#----------------------------------------------------------------------------------------------------------------------------------------
def show_predict_page1():    

    st.title(f"{project.upper()}")
    st.title("SPI Prediction (Mechanical Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")
    
    feature1 = st.number_input('submittal closure rate', value=0.0)
    feature2 = st.number_input('machinery', value=0.0)
    feature3 = st.number_input('IFF Issuance rate', value=0.0)
    feature4 = st.number_input('manpower', value=0.0)
    feature5 = st.number_input('fabrication variance', value=0.0)
    feature6 = st.number_input('safety incident', value=0.0)
    feature7 = st.number_input('rfi closure rate', value=0.0)
    feature8 = st.number_input('material PO issuance rate', value=0.0)

    ok = st.button("Predict")
    if ok:
        data = load_model()
        regressor = data["model"]
        sc_x = data["scaler_x"]
        sc_y = data["scaler_y"]
        
        # Transform input features
        x = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])
        x = x.astype(float)  # Convert to floats for consistency
        x_scaled = sc_x.transform(x) # Use the scaler fitted on training data
        
        # Make prediction
        prediction = sc_y.inverse_transform(regressor.predict(x_scaled).reshape(-1, 1))
        st.subheader(f"The estimated future SPI value in 2 weeks will be {prediction[0][0]:.2f}")
        
        
        # -----------------SHOW FIG ---------------- 
        st.title('Feature Importance Graph (Mechanical Package)')
    
        # Load data
        dataset = load_data()
    

        # Preprocess data
        x, y = preprocess_data(dataset)

        # Train model
        regressor2 = train_model(x, y)

        # Calculate permutation importance
        importance_scores = calculate_permutation_importance(regressor2, x, y)

        # Plot feature importance
        st.subheader('Feature Importance Plot:')
        plot_feature_importance1(feature_names1, importance_scores)

def show_predict_page2():
    st.title(f"{project.upper()}")
    st.title("SPI Prediction (Cleanroom Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")
    
    feature1 = st.number_input('material submittal closure rate', value=0.0)
    feature2 = st.number_input('SOC Submission', value=0.0)
    feature3 = st.number_input('Safety Induction', value=0.0)
    feature4 = st.number_input('manpower', value=0.0)
    feature5 = st.number_input('safety incident', value=0.0)
    feature6 = st.number_input('rfi closure rate', value=0.0)
    feature7 = st.number_input('material PO issuance rate', value=0.0)
    

    ok = st.button("Predict")
    if ok:
        data = load_model2()
        regressor = data["model"]

        sc_x = data["scaler_x"]
        sc_y = data["scaler_y"]
        
       
        x = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]])
        x = x.astype(float)  
        x_scaled = sc_x.transform(x) 
        
   
        prediction = sc_y.inverse_transform(regressor.predict(x_scaled).reshape(-1, 1))
        st.subheader(f"The estimated future SPI value in 2 weeks will be {prediction[0][0]:.2f}")

        # -----------------SHOW FIG ----------------
        st.title('Feature Importance Graph (Cleanroom Package)')

        dataset = load_data2()
        x, y = preprocess_data2(dataset)
        regressor2 = train_model(x, y)
        importance_scores = calculate_permutation_importance(regressor2, x, y)
        st.subheader('Feature Importance Plot:')
        plot_feature_importance2(feature_names2, importance_scores)

def show_predict_page3():
    st.title(f"{project.upper()}")
    st.title("SPI Prediction (HVAC Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")
    
    feature1 = st.number_input('material submittal closure rate', value=0.0)
    feature2 = st.number_input('machinery', value=0.0)
    feature3 = st.number_input('IFF Issuance rate', value=0.0)
    feature4 = st.number_input('safety induction', value=0.0)
    feature5 = st.number_input('manpower', value=0.0)
    feature6 = st.number_input('safety incident', value=0.0)
    feature7 = st.number_input('material arrival non-risky rate', value=0.0)
    

    ok = st.button("Predict")
    if ok:
        data = load_model3()
        regressor = data["model"]
        sc_x = data["scaler_x"]
        sc_y = data["scaler_y"]

        x = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7 ]])
        x = x.astype(float)
        x_scaled = sc_x.transform(x)
        
        prediction = sc_y.inverse_transform(regressor.predict(x_scaled).reshape(-1, 1))
        st.subheader(f"The estimated future SPI value in 2 weeks will be {prediction[0][0]:.2f}")
        



        # -----------------SHOW FIG ----------------
        st.title('Feature Importance Graph (HVAC Package)')

        dataset = load_data3()
        x, y = preprocess_data3(dataset)
        regressor2 = train_model(x, y)
        importance_scores = calculate_permutation_importance(regressor2, x, y)
        st.subheader('Feature Importance Plot:')
        plot_feature_importance3(feature_names3, importance_scores)

def show_predict_page4():
    st.title(f"{project.upper()}")
    st.title("SPI Prediction (Electrical Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")
    
    
    feature1 = st.number_input('material submittal closure rate', value=0.0)
    feature2 = st.number_input('IFF Issuance rate', value=0.0)
    feature3 = st.number_input('SOC Submission', value=0.0)
    feature4 = st.number_input('manpower', value=0.0)
    feature5 = st.number_input('safety incident', value=0.0)
    feature6 = st.number_input('rfi closure rate', value=0.0)
   
    
    ok = st.button("Predict")
    if ok:
        data = load_model4()
        regressor = data["model"]
        sc_x = data["scaler_x"]
        sc_y = data["scaler_y"]

        x = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])
        x = x.astype(float)
        x_scaled = sc_x.transform(x)
        
        prediction = sc_y.inverse_transform(regressor.predict(x_scaled).reshape(-1, 1))
        st.subheader(f"The estimated future SPI value in 2 weeks will be {prediction[0][0]:.2f}")

        # -----------------SHOW FIG ----------------
        st.title('Feature Importance Graph (Electrical Package)')

        dataset = load_data4()
        x, y = preprocess_data4(dataset)
        regressor2 = train_model(x, y)
        importance_scores = calculate_permutation_importance(regressor2, x, y)
        st.subheader('Feature Importance Plot:')
        plot_feature_importance4(feature_names4, importance_scores)


def show_predict_page5():
    st.title(f"{project.upper()}")
    st.title("SPI Prediction (Mechanical Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")

def show_predict_page6():
    st.title(f"{project.upper()}")
    st.title("SPI Prediction (Cleanroom Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")

def show_predict_page7():
    st.title(f"{project.upper()}")
    st.title("SPI Prediction (HVAC Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")

def show_predict_page8():
    st.title(f"{project.upper()}")
    st.title("SPI Prediction (Electrical Package)")
    st.write("""### Input leading indicators from subcontractors' weekly report to detect possible schedule delay""")


# Use a session state variable to control page switching
if "page" not in st.session_state:
    st.session_state.page = "home"

# Check if the Streamlit app is being run directly
if __name__ == '__main__':
    st.set_page_config(page_title='SPI Prediction App')

    # Render the home page
    if st.session_state.page == "home":
        st.title("Welcome to Intel's Project Prediction App")
        st.write("Click the button below to go to the SPI Prediction page.")
        st.image("fotor-ai-2023112914215.jpg", width=400)
      
        if st.button("Get Started"):
            st.session_state.page = "streamlit-app"
            st.session_state.prediction_page = "predict-page1"  # Set default page
            st.experimental_rerun()
            


    # Render the Streamlit app
    elif st.session_state.page == "streamlit-app":
        project = st.sidebar.selectbox("Select Project", ["Falcon", "Pelican"])
        st.session_state.project = project.lower()
        st.sidebar.write("Select Package")


        if project.lower()=='falcon':
            if st.sidebar.button("Mechanical"):
                st.session_state.prediction_page = "predict-page1"
                st.experimental_rerun()
            if st.sidebar.button("Cleanroom"):
                st.session_state.prediction_page = "predict-page2"
                st.experimental_rerun()
            if st.sidebar.button("HVAC"):
                st.session_state.prediction_page = "predict-page3"
                st.experimental_rerun()
            if st.sidebar.button("Electrical"):
                st.session_state.prediction_page = "predict-page4"
                st.experimental_rerun()
           

        if project.lower()=='pelican':
            if st.sidebar.button("Mechanical"):
                st.session_state.prediction_page = "predict-page5"
                st.experimental_rerun()
            if st.sidebar.button("Cleanroom"):
                st.session_state.prediction_page = "predict-page6"
                st.experimental_rerun()
            if st.sidebar.button("HVAC"):
                st.session_state.prediction_page = "predict-page7"
                st.experimental_rerun()
            if st.sidebar.button("Electrical"):
                st.session_state.prediction_page = "predict-page8"
                st.experimental_rerun()
                

        if st.session_state.prediction_page == "predict-page1":
            show_predict_page1()
        elif st.session_state.prediction_page == "predict-page2":
            show_predict_page2()
        elif st.session_state.prediction_page == "predict-page3":
            show_predict_page3()
        elif st.session_state.prediction_page == "predict-page4":
            show_predict_page4()
        elif st.session_state.prediction_page == "predict-page5":
            show_predict_page5()
        elif st.session_state.prediction_page == "predict-page6":
            show_predict_page6()
        elif st.session_state.prediction_page == "predict-page7":
            show_predict_page7()
        elif st.session_state.prediction_page == "predict-page8":
            show_predict_page8()