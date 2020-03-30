import sys
import os
sys.path.append(
    os.path.join(os.path.dirname(__file__), os.path.pardir))
import numpy as np
import pandas as pd
import sklearn.linear_model


def pMSE(dataframe, ind_var):
    #This should be implemented with multiple models chosen by some variable
    #For now it will be done with logistic regression as there is an analytical solutionn to the null value.
    #ind_var is the name of the indicator variable
    model = sklearn.linear_model.LogisticRegression(penalty = 'none').fit(dataframe.drop(ind_var, axis = 1), dataframe[ind_var])
    prediction = model.predict_proba(dataframe.drop(ind_var, axis = 1))
    c = sum(dataframe[ind_var]==0)/len(dataframe)
    pmse = (sum((prediction[:,1] - 0.5)**2))/len(dataframe)
    return pmse
    
def null_pmse_est(dataframe, ind_var, n_iter):
    ##Randomly assigns the indicator variable
    df = dataframe.copy()
    pmse = 0
    
    for i in range(n_iter):
        df[ind_var] = df[ind_var].sample(frac = 1).reset_index(drop = True)
        pmse += pMSE(df, ind_var)

    pmse /= n_iter
    return pmse

        
def df_concat_ind(real_df, gen_df, ind = 'ind'):
    #help function to set up the indicator function
    r_df = real_df.copy()
    g_df = gen_df.copy()
    df = pd.concat((r_df, gen_df), axis=0)
    
    ind_col = pd.DataFrame(np.ones( (len(df), 1) ) , columns= [ind])
    df = pd.concat((df, ind_col), axis = 1)
    return df

    
# def testfun():
#     testing_df, info_1 = multivariate_df(100, [0,0], np.eye(2))
#     df2, info_2 = multivariate_df(100, [1,1], [[1,0.5],[0.5,1]])
#     testing_df = pd.concat([testing_df, pd.DataFrame(np.ones(len(testing_df)), columns = ['ind'])], axis = 1) 
#     df2 = pd.concat([df2, pd.DataFrame(np.zeros(len(df2)), columns = ['ind'])], axis = 1) 
#     df = pd.concat([testing_df, df2])
#     df = df.reset_index(drop=True)

#     print(null_pmse_est(df, 'ind', n_iter = 100))
#     print(pMSE(df, 'ind'))

