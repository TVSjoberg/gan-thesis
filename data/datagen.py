#data synth

import numpy as np
import pandas as pd


def multivariate_df(n_samples, mean, cov):
    
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    cols = col_name_gen(len(mean), 'c')
    df = pd.DataFrame(data, columns=cols)
    info = {
        'mean' : mean,
        'covariance' : cov
    }
    return df, info
    

def mixtureGauss(n_samples, means, covs):
   #Note that means and covs are lists of means and Covariance matrices
    info = {}
    
    cols = col_name_gen(len(means[0]),'c')
    df = pd.DataFrame(np.zeros((n_samples, len(means[0]))), columns = cols)
    
    for i in range(len(means)):
        temp_df, temp_info = multivariate_df(n_samples,means[i], covs[i])
        df += temp_df
        info['Gaussian '+str(i)] = temp_info
    return df, info


        
def categoricalData(n_samples, probabilities):
    #n_samples: int ,  probabilites = nested list of probabilities
    info = {}
    
    column_names = col_name_gen(len(probabilities),'feature_')
    df = pd.DataFrame(columns = column_names)
    
    count = 0
    for prob in probabilities:
        temp_label_names = col_name_gen(len(prob), 'feature_'+str(count)+'_label_')
        temp_data = np.random.choice(temp_label_names, size = n_samples, p = prob)
        
        df[column_names[count]] = temp_data
        info[column_names[count]] = prob
        count += 1
    return df, info

        
        

    
    



def col_name_gen(num_cols, common_name):
    common_name_list = [common_name]*num_cols
    num_string_list = [num_string for num_string in map(
        lambda num: str(num), [num for num in range(num_cols)]
    )]
    res_list = [a + b for a, b in zip(common_name_list, num_string_list)]
    return res_list
    
