import streamlit as st
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

st.title('GoLem Pharm')
st.write('')

#input box
int_put2 =  st.text_input('Sequence of protein:   e.g. MLLETQDALYVALELVIAALSVAGNVLVCAAVG...')
@st.cache:
if int_put2:
 with st.spinner('Please wait...'):
  df1 = pd.read_csv('./data_ligand1.csv')
  df2 = pd.read_csv('./data_ligand2.csv')
  df3 = pd.read_csv('./data_ligand3.csv')
  df_bind = pd.concat([df1, df2, df3], axis=0)
  #df_bind = pd.read_csv('C:\\Users\\ncbir\\Desktop\\projekt\\data_ligand.csv', encoding = 'utf_8')
  df_bind['seq'] = str(int_put2)
  df_bind = df_bind.dropna()
  df_bind = df_bind.drop_duplicates(subset=['Ligand SMILES'])
  df_bind = df_bind.astype({'Ligand SMILES': str})
  df_bind = df_bind.astype({'seq': str})
  df_bind['to_a'] = df_bind['Ligand SMILES'].str.cat(df_bind['seq'],sep=" ")

  lines = df_bind['to_a'].values.tolist()
  review_lines = list()
  for line in lines:
   review_lines.append(line)

  MAX_SEQUENCE_LENGTH = 600

  word_index1 = np.load('./word_index_100.npy',allow_pickle='TRUE').item()

  tokenizer = Tokenizer(lower = False, char_level=True)
  tokenizer.word_index = word_index1

  sequences = tokenizer.texts_to_sequences(review_lines)
  review_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating = 'post')

  x_test = review_pad
  from keras.models import load_model
  model2 = load_model('./final_modellog.h5')
  predict_x=model2.predict(x_test)
  df_bind['value (pKi)'] = predict_x
  df_bind2 = df_bind.sort_values(by=['value (pKi)'], ascending = False)
  df_bind3 = df_bind2[['Ligand SMILES', 'value (pKi)']]
  df_bind4 = df_bind3.iloc[0:20]
  df_bind4 = df_bind4.drop_duplicates(subset=['Ligand SMILES'])
  df_bind4 = df_bind4.reset_index(drop=True)
  df_bind4 = df_bind4.shift()[1:]
  st.dataframe(df_bind4.style.format({'value (pKi)':'{:.2f}'}))
  #df_bind4 = df_bind4.reset_index(inplace=True)
  #df_bind4

  from rdkit import Chem
  from rdkit.Chem.rdDepictor import Compute2DCoords
  from rdkit.Chem.Draw import rdMolDraw2D
  from rdkit.Chem import Draw

  molecule = Chem.MolFromSmiles(df_bind4.iloc[0, 0])
  fig = Draw.MolToMPL(molecule, size = [200, 200])
  st.pyplot(fig)
