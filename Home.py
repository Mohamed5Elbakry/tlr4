import streamlit as st
import os
import joblib
import pandas as pd
import subprocess
from PIL import Image

# Page configuration
st.set_page_config(
  page_title='Predict Activity of single molecule')

if 'smiles_input' not in st.session_state:
  st.session_state.smiles_input = ''

if os.path.isfile('molecule.smi'):
  os.remove('molecule.smi') 
  
def PUbchemfp_desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G  -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints  -descriptortypes ./PaDEL-Descriptor/AtomPairs2DFingerprinter.xml  -dir ./ -file descriptors.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')  
     
st.title('TLR4 Activity Prediction App')
st.info('The TLR4 Activity Prediction App can be used to predict whether a  molecule is active or inactive for TLR4 target protein .')

if st.session_state.smiles_input == '':
    

      smiles_txt = st.text_input('Enter Compound in Smile Format', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      with st.expander('Example SMILES'):

        st.code('CC5CN(c4cc3nc(NC2C=NN(C1CC1)C2C)ncc3cc4Cl)CCC5(C)O')
        st.code('O=C(Nc1ncco1)c1cc(Br)ccc1OCc1ccccc1')
submit_button = st.button('Predict')

      
      
if submit_button:
        st.subheader(' Input molecule:')
        with st.expander(' SMILES', expanded=True):
          
          st.text(st.session_state.smiles_input)


      
          smile_file = open('molecule.smi', 'w')
          smile_file.write(f'{st.session_state.smiles_input}\tName_00')
          smile_file.close()


if st.session_state.smiles_input != '':
        st.subheader(' Descriptors')
        if os.path.isfile('molecule.smi'):
             PUbchemfp_desc_calc()

        descriptors = pd.read_csv('descriptors.csv')
        descriptors.drop('Name', axis=1, inplace=True)
        st.expander('Full set of molecule discriptors')
        
        st.write(descriptors)
        st.write(descriptors.shape)

        st.header('set of descriptors used to prediction')
        train_set=pd.read_csv('Atompairs_best.csv')
        feature_list=list(train_set.columns)
        desc_subset = descriptors[feature_list]
        st.write(desc_subset)
        st.write(desc_subset.shape)




      
if st.session_state.smiles_input != '':
 model = joblib.load('TLR4_model.pkl')

if st.session_state.smiles_input != '':
        st.subheader('Predictions')
        prediction = model.predict(desc_subset)
        prediction_probability=model.predict_proba(desc_subset)
        prediction_output = pd.Series(prediction, name='Activity')
        x=pd.DataFrame(prediction_probability,columns=["Inactive probability","Active_probability"])
        Result= pd.concat([prediction_output,x], axis=1)
        result = []
        for x in Result["Activity"]:
          if x==1:
            result.append("Active")
          if x==0:
            result.append("Inactive")
        Result["Activity"]=result
        st.write(Result)
        prediction_csv = Result.to_csv(index=False)
        st.download_button(label="Download prediction result",data=prediction_csv,file_name="My_result.csv")   
st.subheader('TLR4_Receptor')
with st.expander("Unveiling the Mysteries of Multiple Sclerosis: TLR4's Impact"):
        st.write("Deep within the intricate web of the immune system lies a fascinating player known as Toll-like receptor 4 (TLR4), "
                 "whose enigmatic role has captivated researchers in the field of multiple sclerosis (MS) disease. TLR4, a transmembrane "
                 "protein, acts as a sentinel, detecting both internal damage-associated molecular patterns (DAMPs) and external "
                 "pathogen-associated molecular patterns (PAMPs). Its unwavering vigilance and remarkable ability to orchestrate "
                 "immune responses have propelled TLR4 into the spotlight of MS research.")
    
        st.write("MS, an autoimmune disorder characterized by the immune system's misguided attacks on the central nervous system, "
                 "is believed to find its roots in the intricate dance between TLR4 and its co-conspirators. TLR4's distinctive "
                 "activation mechanism, relying on the indispensable assistance of myeloid differentiation factor 2 (MD-2), sets it "
                 "apart from its fellow TLRs. Through this partnership, TLR4 triggers an intricate cascade of events, triggering "
                 "immune responses that can either exacerbate or mitigate the progression of MS.")




st.subheader("TLR4 3D Structure")
with st.expander('TLR4 surface pocket'):
     sur=Image.open('surface.jpg') 
     st.image(sur, use_column_width=True)


st.subheader("TLR4 Dataset")
with st.expander("Dataset"):
  st.write('''
    In our work, we retrieved a human TLR4 biological dataset from the ChEMBL database and Binding database. The data was curated and resulted in a non-redundant set of 181 TLR4 inhibitors, which can be divided into:
    - 17 active compounds
    - 164 inactive compounds
    ''')

st.subheader("Model Performance")
with st.expander("Model Performance"):
     st.write("We chosed AtomPairs2Dfingerprinter for building the model using RandomForest Classifier. SMOTE technique was applied to overcome imbalance problem.")

     st.write("Sensitivity (SN) : 60")
     st.write("Specificity (SP) : 93")
     st.write("Matthews’s correlation coefficient (MCC) : 0.53")
     st.write("Accuracy (Q) : 89")


