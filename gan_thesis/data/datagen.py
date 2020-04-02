
import numpy as np
import pandas as pd


def multivariate_df(n_samples, mean, cov, seed = False):
    if seed:
        np.random.seed(seed)
        
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    cols = col_name_gen(len(mean), 'c')
    df = pd.DataFrame(data, columns=cols)
    info = {
        'type' : 'mvn',
        'mean' : mean,
        'covariance' : cov
    }
    return df, info

def log_normal_df(n_samples, mean, cov, seed = False):
    df, info = multivariate_df(n_samples, mean, cov, seed)
    
    df = df.applymap(lambda x: np.exp(x))
    print(df)
    info['type'] = 'log-normal'
    return df, info


def mixture_gauss(n_samples, proportions, means, covs, seed = False):
    #Note that means and covs are lists of means and Covariance matrices
    info = {}
    if seed:
        np.random.seed(seed)
    
    cols = col_name_gen(len(means[0]),'c')
    df = pd.DataFrame(columns = cols)
    
    for i in range(len(means)):
        temp_df, temp_info = multivariate_df(int(np.floor(n_samples*proportions[i])), means[i], covs[i], seed)
        df = pd.concat((df, temp_df))
        temp_info['Proportion of total'] = proportions[i]
        info['dist '+ str(i)] = temp_info
    return df, info

def mixture_log_normal(n_samples, proportions, means, covs, seed = False):
    #Note that means and covs are lists of means and Covariance matrices
    info = {}
    if seed:
        np.random.seed(seed)
    
    cols = col_name_gen(len(means[0]),'c')
    df = pd.DataFrame(columns = cols)
    
    for i in range(len(means)):
        temp_df, temp_info = log_normal_df(int(np.floor(n_samples*proportions[i])), means[i], covs[i], seed)
        df = pd.concat((df, temp_df))
        temp_info['Proportion of total'] = proportions[i]
        info['dist '+ str(i)] = temp_info
    return df, info



def categorical_data(n_samples, probabilities, seed = False):
    #n_samples: int ,  probabilites = nested list of probabilities
    info = {}
    if (seed != False):
        np.random.seed(seed)
        
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



    
