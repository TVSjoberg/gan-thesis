
import numpy as np
import pandas as pd
from scipy.stats import random_correlation


def multivariate_df(n_samples, mean, var, corr, seed=False, name = 'c'):
    if seed:
        np.random.seed(seed)
    
    cov = corr_var_to_cov(corr, var)
    if (len(mean) == 1):    
        data = np.random.normal(mean, cov[0]**2, n_samples)
    else:
        data = np.random.multivariate_normal(mean, cov, n_samples)

    cols = col_name_gen(len(mean), name)
    df = pd.DataFrame(data, columns=cols)
    info = {
        'type': 'mvn',
        'mean': mean,
        'correlation' : corr,
        'variance': var,
        'dim' : len(mean)
    }
    return df, info


def log_normal_df(n_samples, mean, var, corr, seed=False):
    df, info = multivariate_df(n_samples, mean, var, corr, seed)

    df = df.applymap(lambda x: np.exp(x))
    info['type'] = 'log-normal'
    return df, info


def mixture_gauss(n_samples, proportions, means, varis, corrs, seed=False):
    # Note that means and var, corr are lists of means, variances and Correlation matrices
    
    info = {}
    if seed:
        np.random.seed(seed)
    
    k = len(means)
    cols = col_name_gen(len(means[0]), 'c')
    df = pd.DataFrame(columns=cols)
    n_samples_li = np.random.multinomial(n_samples, proportions)

    for i in range(k):
        temp_df, temp_info = multivariate_df(n_samples_li[i],
                                              means[i], varis[i], corrs[i], seed)
        df = pd.concat((df, temp_df))
        temp_info['Proportion of total'] = proportions[i]
        info['dist ' + str(i)] = temp_info
    
    
    info['dim'] = len(means[0])
    return df, info


def mixture_log_normal(n_samples, proportions, means, varis, corrs, seed=False):
    # Note that means and covs are lists of means and Covariance matrices
    info = {}
    if seed:
        np.random.seed(seed)

    k = len(means)
    cols = col_name_gen(len(means[0]), 'c')
    df = pd.DataFrame(columns=cols)
    n_samples_li = np.random.multinomial(n_samples, proportions)

    for i in range(k):
        temp_df, temp_info = log_normal_df(n_samples_li[i],
                                            means[i], varis[i], corrs[i], seed)
        df = pd.concat((df, temp_df))
        temp_info['Proportion of total'] = proportions[i]
        info['dist ' + str(i)] = temp_info
        
    info['dim'] = len(means[0])
    return df, info


def cat_mixture_gauss(cond_df, cond_info, means, varis, corrs, seed = False):
    # Note label in feature vectors need have names in the same order 
    # as means/covs
    # 
    info = {'Conditional info' : cond_info,
            'mixture info': {} }
    if seed:
        np.random.seed(seed)
    dim_count = cond_info['dim']
    
    
    unique = []
    n_samples = []
    for i in range(len(cond_df.columns)): #For each categorical features
        
        temp_li = cond_df[cond_df.columns[i]].unique() #Find unique labels of features
        temp_li.sort()
        unique.append(temp_li)
        
        temp_li = []
        for j in range(len(unique[i])): #Find number of samples with i,j label
            temp_li.append(
                sum(cond_df[cond_df.columns[i]]==unique[i][j])
                )
        n_samples.append(temp_li)
    
    for i in range(len(unique)): #For every categorical feature
        df = pd.DataFrame()
        dim_count += len(means[i][0])
        
        for j in range(len(unique[i])): #for each unique label

            temp_df, temp_info = multivariate_df(n_samples[i][j], means[i][j],
                                                  varis[i][j], corrs[i][j], name = (cond_df.columns[i]+ '_c'))
            df = pd.concat((df, temp_df))
            df = df.reset_index(drop = True)
            info['mixture info']['Cat_feature_{0} label_{1}'.format(str(i),str(j))] = temp_info
        
        cond_df = cond_df.sort_values(cond_df.columns[i]).reset_index(drop=True)    
        cond_df = pd.concat((cond_df, df), axis = 1)
        
    info['dim'] = dim_count
    
    return cond_df, info
    
    

def multinomial(n_samples, probabilities, seed=False, name='f_'):
    # n_samples: int ,  probabilites = nested list of probabilities
    info = {}
    if seed:
        np.random.seed(seed)

    column_names = col_name_gen(len(probabilities), name)
    df = pd.DataFrame(columns=column_names)

    count = 0
    for prob in probabilities:
        temp_label_names = col_name_gen(len(prob), name+str(count)+'_l_')
        temp_data = np.random.choice(temp_label_names, size=n_samples, p=prob)

        df[column_names[count]] = temp_data
        info[column_names[count]] = prob
        count += 1
    info['dim'] = sum(map(lambda prob: len(prob), probabilities))
    return df, info


def multinomial_cond(n_samples, ind_probabilities, cond_probabilities, seed=False):
    # n_samples: int,
    # ind_probabilities : nested list of probabilities one per feature,
    # cond_probabilities : double nested list of probabilities with [0][0] representing p(cond_feature_1|ind_feature_1=label_1)
    
    # Example Use:
    # multinomial_cond(20, [[0.5, 0.5],[0.5, 0.5]], [
    #     [
    #         [
    #             [0.8, 0.2],[0, 1]
    #         ], [
    #             [0, 1],[1, 0]
    #         ]
            
    #     ], [
    #         [
    #             [1, 0, 0], [0, 1]
    #         ], [
    #             [0, 0, 1], [1, 0]
    #         ], [
    #             [0.1, 0.4, 0.5], [1, 0]
    #         ]
            
    #     ]
    # ]
    
    
    info = {}
    if seed:
        np.random.seed()

    ind_df, ind_info = multinomial(
        n_samples, ind_probabilities, seed, 'indf_')
    cond_df = pd.DataFrame()
    
    dim_count = ind_info['dim']
    info['source distributions'] = ind_info

    for i in range(len(ind_probabilities)):
        cond_df = pd.DataFrame()

        unique_labels = ind_df[ind_df.columns[i]].unique()
        unique_labels.sort()
        ind_df = ind_df.sort_values(ind_df.columns[i])
        ind_df = ind_df.reset_index(drop=True)

        temp_li1 = []
        temp_li2 = []
        
        for j in range(len(unique_labels)):
            temp_n = len(ind_df[ind_df[ind_df.columns[i]] == unique_labels[j]])
            temp_df, temp_info = multinomial(
                temp_n, cond_probabilities[i][j], seed, 'cf_'+str(i))
            
            temp_li1.append(temp_df)
            temp_info['conditional on'] = unique_labels[j]
            temp_li2.append(temp_info)
        
        dim_count += temp_info['dim']
        
        temp = pd.concat(temp_li1)
        cond_df = pd.concat((cond_df, temp), axis=0)
        cond_df = cond_df.reset_index(drop=True)
        ind_df = pd.concat([ind_df, cond_df], axis=1)

        
        info['conditional on ' +str(i)] = temp_li2
    info['dim'] = dim_count

    return ind_df, info


def multinomial_cond_extension(n_samples, true_ind_prob, ind_prob, cond_prob, seed = False):
    ##Used to add truly independent multinomials to a conditional set
    info = {}
    if seed:
        np.random.seed()
    
    true_ind_df, true_ind_info = multinomial(n_samples, true_ind_prob, seed, 'tif_')    
    cond_df, cond_info = multinomial_cond(n_samples, ind_prob, cond_prob, seed)
    
    df = pd.concat((true_ind_df, cond_df), axis=1)
    info['true independent'] = true_ind_info
    info['conditionals'] = cond_info
    info['dim'] = cond_info['dim'] + true_ind_info['dim']
    return df, info
  
  
  
  ### Helper functions
  
def corr_var_to_cov(corr, var):
    corr = np.array(corr)
    var = np.array(np.sqrt(var))
    
    res = corr*var
    var = var.reshape(len(var),1)
    
    res = res*var
    return res
     
    
def r_corr(size):
    r_arr = np.random.uniform(0,5, size = size)
    r_arr = size*r_arr/sum(r_arr)
    return random_correlation.rvs(r_arr)   
    
def normalize(vec):
    return vec/sum(vec)

def rand_prop(size):
    return(normalize(np.random.uniform(0,1,size = size)))
    

def col_name_gen(num_cols, common_name):
    common_name_list = [common_name]*num_cols
    num_string_list = [num_string for num_string in map(
        lambda num: str(num), [num for num in range(num_cols)]
    )]
    res_list = [a + b for a, b in zip(common_name_list, num_string_list)]
    return res_list



