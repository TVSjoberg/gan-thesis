
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

def df_to_dataset(dataframe_in, shuffle=True, batch_size = 32):
  dataframe = dataframe_in.copy()
  ds = tf.data.Dataset.from_tensor_slices(dataframe.values)
  ds= ds.batch(batch_size)
  return ds

def dataset_to_df(dataset, col_names, batch = False):
  df = pd.DataFrame(columns = col_names)
  if batch:
    for batch in dataset:
      df = pd.concat([df, pd.DataFrame(batch.numpy(), columns = col_names)], ignore_index = True)
  else:
    df = pd.concat([df, pd.DataFrame(dataset.numpy(), columns=col_names)], ignore_index=True)
  return df

def load_credit_data():
    data = tf.keras.utils.get_file("Credit.csv", "http://faculty.marshall.usc.edu/gareth-james/ISL/Credit.csv")
    data = pd.read_csv(data)
    data = data.drop(data.columns[0],axis=1)
    return data

def data_reorder(dataframe, cat_cols):
  #put the 
  temp = dataframe[cat_cols]
  dataframe = dataframe.drop(cat_cols, axis = 1)
  dataframe = pd.concat([dataframe,temp], axis = 1)
  return dataframe

class dataScaler:

  def __init__(self):
    self.std_scaler = preprocessing.StandardScaler()
    self.oht_scaler = preprocessing.OneHotEncoder()

  def transform(self, df_in, cont_cols, cat_cols, fit = True):
    df = df_in.copy()
    df = self.std_scale(df, cont_cols, fit)
    df = self.oht_transform(df, cat_cols, fit)
    return df

  def inverse_transfrom(self, df_in):
    df = df_in.copy()
    df = self.inv_std_scale(df)
    df = self.inv_oht_transform(df)
    return df
  
  def oht_transform(self, df_in, cat_cols, fit = True):
    df = df_in.copy()
    self.cat_cols = cat_cols
    self.oht_scaler.fit(df[self.cat_cols])
    
    oht_enc = self.oht_scaler.transform(df[self.cat_cols])
    oht_pd = pd.DataFrame(oht_enc.toarray(), columns = self.oht_scaler.get_feature_names())
    
    df = df.drop(columns = self.cat_cols)
    return pd.concat([df, oht_pd], axis =1)

  def inv_oht_transform(self, df_in):
    df = df_in.copy()

    temp_cols = self.oht_scaler.get_feature_names()
    inv_enc = self.oht_scaler.inverse_transform(df[temp_cols])
    inv_pd = pd.DataFrame(inv_enc, columns = self.cat_cols)
    
    df = df.drop(columns = temp_cols)
    return pd.concat([df, inv_pd], axis = 1)
    
  def std_scale(self, df_in, cont_cols, fit = True):
    df = df_in.copy()
    if fit:
      self.cont_cols = cont_cols
      self.std_scaler.fit(df[self.cont_cols])
    df[self.cont_cols] = self.std_scaler.transform(df[self.cont_cols])
    return df

  def inv_std_scale(self, df_in):
    df = df_in.copy()
    df[self.cont_cols] = self.std_scaler.inverse_transform(df[self.cont_cols])
    return df