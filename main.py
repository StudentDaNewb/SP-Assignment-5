import pandas as pd
import numpy as np
import sklearn.linear_model as lin
import scipy.stats as stat
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import math


# calculates MAE
def MAE( data_real, data_pred ):
  return np.sum( np.abs( data_real - data_pred ) ) / len( data_real )

# calculates R2
def R2( data_real, data_pred ):
  mean = np.mean( data_real )
  sst = np.sum( ( data_real - mean ) * ( data_real - mean ) )
  ssr = np.sum( ( data_real - data_pred ) * ( data_real - data_pred ) )
  return 1.0 - ssr / sst

# calculates the spearman rank correlation + pvalue
def SC( data_real, data_pred ):
  return stat.spearmanr( data_real, data_pred )

# transformation needed because when transforming features some themperatures are negative, hence rooting negative numbers does not work
def convert_c_to_kelvin(celsius):
  return 273.15 + float(celsius)

# reads the csvs, combines them together, converts AT to Kelvin, and returns the resulting dataset
def read_csv(names, add_bias=True):
  l = []
  for name in names:
    l.append(pd.read_csv(name, header=None))

  df = pd.concat(l, axis=0, ignore_index=True)

  if add_bias:
    df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)

  df[0] = df[0].apply(convert_c_to_kelvin)

  return df

# Divides the dataframe in N segments
def segmentate_data(amt_segments, dataframe):
  segmented_df = []
  amt_entries_per_block = math.trunc(len(dataframe) / amt_segments)
  
  # Split up the data in N segments
  for i in range(0, amt_segments):
    if i == (amt_segments - 1):
      segmented_df.append(dataframe.iloc[i * amt_entries_per_block : len(dataframe)])

    segmented_df.append(dataframe.iloc[i * amt_entries_per_block : (i + 1) * amt_entries_per_block])
    
  return segmented_df


# performs phase 1 part a, returns the original and the predicted NOX
def partA():
  df = pd.concat([pd.read_csv('gt_2011.csv', header=None), pd.read_csv('gt_2012.csv', header=None)], axis=0, ignore_index=True)

  df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)

  X = df.drop(columns=[9, 10])
  y = df[10]
  df[0] = df[0].apply(convert_c_to_kelvin)

  lr = lin.LinearRegression()
  reg = lr.fit(X, y)

  df2 = pd.read_csv('gt_2013.csv', header = None)
  df.head()

  df2 = pd.concat([pd.Series(1, index=df2.index, name='00'), df2], axis=1)
  df2.head()

  X = df2.drop(columns=[9, 10])
  y = df2[10]

  yp = reg.predict( X )

  return y, yp

# performs phase 1 part b, returns the original and the predicted NOX
def partB():
  df = pd.concat([pd.read_csv('gt_2011.csv', header=None), pd.read_csv('gt_2012.csv', header=None), pd.read_csv('gt_2013.csv', header=None)], axis=0, ignore_index=True)

  df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)

  X = df.drop(columns=[9, 10])
  y = df[10]

  lr = lin.LinearRegression()
  reg = lr.fit(X, y)

  b = []
  b.append(pd.read_csv('gt_2014.csv', header = None))
  b.append(pd.read_csv('gt_2015.csv', header = None))

  df2 = pd.concat(b, axis=0, ignore_index=True)
  df.head()

  df2 = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
  df2.head()

  X = df.drop(columns=[9, 10])
  y = df[10]

  yp = reg.predict( X )

  return y, yp

  
# results for best power-feature combinations
results = [float('-inf'), '', float('-inf'), '']

# performs phase 2, puts results into results list
def transform_features(i, pwrs):
  # 0: AT
  # 1: AP - rem
  # 2: AH - Remove from the feature set
  # 3: AFDP -rem
  # 4: GTEP - Remove from the feature set
  # 5: TIT
  # 6: TAT
  # 7: TEY
  # 8: CDP
  # 9: CO
  # 10: NOX
  
  features = [0, 5, 6, 7, 8]

  df_train = read_csv(['gt_2011.csv', 'gt_2012.csv'], add_bias=False)
  df_val = read_csv(['gt_2013.csv'], add_bias=False)
  df_test = read_csv(['gt_2014.csv', 'gt_2015.csv'], add_bias=False)

  # Amount of different permutations possible
  for j in range(5):
    if pwrs[j] != 0: continue
    if i != 3: continue
    X = []
    y = df_train[10]

    for k in range(len(df_train[0])):
      temp = [df_train[0][k], df_train[5][k], df_train[6][k], df_train[7][k], df_train[8][k]]
      for l in range(5):
        temp[l] = np.power(temp[l], np.power(2.0, i)) if l == j else np.power(temp[l], np.power(2.0, pwrs[l]))
      X.append(temp)

    lr = lin.LinearRegression()
    reg = lr.fit(X, y)

    print(reg.coef_)

    ############################

    X = []
    y = df_val[10]
    for k in range(len(df_val[0])):
      temp = [df_val[0][k], df_val[5][k], df_val[6][k], df_val[7][k], df_val[8][k]]
      for l in range(5):
        temp[l] = np.power(temp[l], np.power(2.0, i)) if l == j else np.power(temp[l], np.power(2.0, pwrs[l]))
      X.append(temp)

    yp = reg.predict(X)

    R2_res_val = R2(y, yp)

    ###########################

    X = []
    y = df_test[10]
    for k in range(len(df_test[0])):
      temp = [df_test[0][k], df_test[5][k], df_test[6][k], df_test[7][k], df_test[8][k]]
      for l in range(5):
        temp[l] = np.power(temp[l], np.power(2.0, i)) if l == j else np.power(temp[l], np.power(2.0, pwrs[l]))
      X.append(temp)

    yp = reg.predict( X )

    R2_res_test = R2( y, yp )

    print( yp[312] )
    print( yp[6276])

    if R2_res_val > results[0]:
      results[0] = R2_res_val
      results[1] = "Best Val" + str(i) + "_" + str(features[j]) + " = [" + str(R2_res_val) + ", " + str(R2_res_test) + "]\n"

    if R2_res_test > results[2]:
      results[2] = R2_res_test
      results[3] = "Best Test" + str(i) + "_" + str(features[j]) + " = [" + str(R2_res_val) + ", " + str(R2_res_test) + "]"

    print("Test Val" + str(i) + "_" + str(features[j]) + " = [" + str(R2_res_val) + ", " + str(R2_res_test) + "]")


original_sc_list = []
selected_sc_list = []

#performs phase 3.
def phase3(training_data, validation_data, segments):
  # Whole data set
  df_train = training_data
  df_val = validation_data

  # Original feature set and selected feature set
  df_train_original = df_train.drop(columns=[9, 10])
  df_train_selected = df_train.drop(columns=[1, 2, 3, 4, 9, 10])
  df_val_original = df_val.drop(columns=[9, 10])
  df_val_selected = df_val.drop(columns=[1, 2, 3, 4, 9, 10])

  # Above mentioned data sets segmentated
  df_val_original_seg = segmentate_data(segments, df_val_original)
  df_val_selected_seg = segmentate_data(segments, df_val_selected)

  # Y values for each set
  df_val_y_seg = segmentate_data(segments, df_val[10])

  #Defining initial train data sets
  lr = lin.LinearRegression()
  X_original=df_train_original
  X_selected=df_train_selected

  #Adding  engineered set of features powers
  for k in range(len(X_selected[0])):
    X_selected[0][k] = np.power(X_selected[0][k], (1.0/32.0))
    X_selected[5][k] = np.power(X_selected[5][k], 2.0)
    X_selected[6][k] = np.power(X_selected[6][k], (1.0/4.0))
    X_selected[7][k] = np.power(X_selected[7][k], 4.0)
    X_selected[8][k] = np.power(X_selected[8][k], 8.0)

  y=df_train[10]

  original_val_sum = 0 # original feature set validation value sum
  selected_val_sum = 0 # selected feature set validation value sum
  original_sc_cor_sum = 0 # original feature set Spearman Rank Correlation value sum
  selected_sc_cor_sum = 0 # selected feature set Spearman Rank Correlation value sum
  original_sc_p_sum = 0 # original feature set Spearman Rank Correlation p-value sum
  selected_sc_p_sum = 0 # selected feature set Spearman Rank Correlation p-value sum

  for i in range(segments):
    #fitting original data
    reg = lr.fit(X_original, y)
    yp_original = reg.predict(df_val_original_seg[i])
    #fitting selected features data
    reg = lr.fit(X_selected, y)
    yp_selected = reg.predict(df_val_selected_seg[i])
    #calculate R squared
    R2_original_val = R2(df_val_y_seg[i], yp_original)
    R2_selected_val = R2(df_val_y_seg[i], yp_selected)

    original_val_sum += R2_original_val
    selected_val_sum += R2_selected_val
    #adding a block to the original data set
    X_original = pd.concat([X_original, df_val_original_seg[i]], axis=0)
    #adding a block to the selected features data set
    X_selected = pd.concat([X_selected, df_val_selected_seg[i]], axis=0)
    
    y = pd.concat([y, df_val_y_seg[i]], axis=0)
    #calculate Spearman Rank Correlation
    original_sc = SC(df_val_y_seg[i], yp_original)
    selected_sc = SC(df_val_y_seg[i], yp_selected)

    original_sc_cor_sum += original_sc[0]
    selected_sc_cor_sum += selected_sc[0]
    original_sc_p_sum += original_sc[1]
    selected_sc_p_sum += selected_sc[1]
    original_sc_list.append(original_sc[0])
    selected_sc_list.append(selected_sc[0])

    print("Prediction for original features for block: "+ str(i))
    print("Original Validation: "+ str(R2_original_val) + "\nSC: " + str(original_sc))
    
    print("Selected Validation: "+ str(R2_selected_val) + "\nSC: " + str(selected_sc))

    print("Original MAE: "+ str(MAE(df_val_y_seg[i], yp_original)))
    print("Selected MAE: "+ str(MAE(df_val_y_seg[i], yp_selected))+"\n")
  
  print("Combined prediction score")
  print("Original validation: " + str(original_val_sum/segments))
  print("Original SC correlation: " + str(original_sc_cor_sum/segments))
  print("Original SC p-value: " + str(original_sc_p_sum/segments))
  print("Selected validation: " + str(original_val_sum/segments))
  print("Selected SC correlation: " + str(selected_sc_cor_sum/segments))
  print("Selected SC p-value: " + str(selected_sc_p_sum/segments))
  


#UNCOMMENT FOR PHASE 1
#y, yp = partA()
#print(MAE( y, yp ))
#print(R2(y, yp))
#print(SC(y, yp))

#y, yp = partB()
#print(MAE( y, yp ))
#print(R2(y, yp))
#print(SC(y, yp))



#UNCOMMENT FOR PHASE 2
#for i in range(5):
#  transform_features(i, [0, 0, 0, 0, 0])

#UNCOMMENT FOR PHASE 3
#phase3(read_csv(['gt_2011.csv', 'gt_2012.csv'], False), read_csv(['gt_2013.csv'], False), 10)
#phase3(read_csv(['gt_2011.csv', 'gt_2012.csv', 'gt_2013.csv'], False), read_csv(['gt_2014.csv', 'gt_2015.csv'], False), 20)

#stat, p = ttest_ind(original_sc_list, selected_sc_list)
#print('Statistics=%.3f, p=%.3f' % (stat, p))

#stat, p = ttest_rel(original_sc_list, selected_sc_list)
#print('Statistics=%.3f, p=%.3f' % (stat, p))

