
import numpy as np 
import pickle 
import streamlit as st 


loaded_model = pickle.load(open('C:/Users/DELLS/Desktop/XGboost.sav', 'rb'))


def Sales_prediction(input_data):
   input_data_as_numpy_array= np.asarray(input_data)
   input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
   prediction= loaded_model.predict(input_data_reshaped)
   print(prediction)


def main():
    st.title('Sales Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Item_Weight= st.text_input('WEIGHT ')
    Item_Fat_Content= st.text_input('Fat')
    Item_Visibility= st.text_input('Visibility')
    Item_Type= st.text_input('item_type')
    Item_MRP= st.text_input('MRP')
    Outlet_Establishment_Year = st.text_input('Year')
    Outlet_Size = st.text_input('Outlet_Size')
    Outlet_Location_Type= st.text_input('Location')
    #Outlet_Type= st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Result'):
        diagnosis = Sales_prediction([Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()