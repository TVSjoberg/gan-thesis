
import numpy as np
import pandas as pd


def multivariate_df(n_samples, mean, cov, seed=False):
    if seed:
        np.random.seed(seed)

    data = np.random.multivariate_normal(mean, cov, n_samples)

    cols = col_name_gen(len(mean), 'c')
    df = pd.DataFrame(data, columns=cols)
    info = {
        'type': 'mvn',
        'mean': mean,
        'covariance': cov,
        'dim' : len(mean)
    }
    return df, info


def log_normal_df(n_samples, mean, cov, seed=False):
    df, info = multivariate_df(n_samples, mean, cov, seed)

    df = df.applymap(lambda x: np.exp(x))
    print(df)
    info['type'] = 'log-normal'
    return df, info


def mixture_gauss(n_samples, proportions, means, covs, seed=False):
    # Note that means and covs are lists of means and Covariance matrices
    info = {}
    if seed:
        np.random.seed(seed)
    
    k = len(means)
    cols = col_name_gen(len(means[0]), 'c')
    df = pd.DataFrame(columns=cols)

    for i in range(k-1):
        temp_df, temp_info = multivariate_df(
            int(np.floor(n_samples*proportions[i])), means[i], covs[i], seed)
        df = pd.concat((df, temp_df))
        temp_info['Proportion of total'] = proportions[i]
        info['dist ' + str(i)] = temp_info
    
    temp_df, temp_info = multivariate_df((n_samples-len(df)), means[k-1], covs[k-1], seed)
    df = pd.concat((df, temp_df))
    temp_info['Proportion of total'] = proportions[k-1]
    info['dist ' + str(k-1)] = temp_info
    
    info['dim'] = len(means[0])
    return df, info


def mixture_log_normal(n_samples, proportions, means, covs, seed=False):
    # Note that means and covs are lists of means and Covariance matrices
    info = {}
    if seed:
        np.random.seed(seed)

    k = len(means)
    cols = col_name_gen(len(means[0]), 'c')
    df = pd.DataFrame(columns=cols)

    for i in range(k-1):
        temp_df, temp_info = log_normal_df(
            int(np.floor(n_samples*proportions[i])), means[i], covs[i], seed)
        df = pd.concat((df, temp_df))
        temp_info['Proportion of total'] = proportions[i]
        info['dist ' + str(i)] = temp_info
    
    temp_df, temp_info = log_normal_df((n_samples-len(df), means[k-1], covs[k-1], seed))
    df = pd.concat((df, temp_df))
    temp_info['Proportion of total'] = proportions[k-1]
    info['dist ' + str(i)] = temp_info
        
    info['dim'] = len(means[0])
    return df, info

def cat_mixture_gauss(cond_df, means, covs, seed = False):
    # Note label in feature vectors need have names in the same order 
    # as means/covs
    # 
    info = {}
    if seed:
        np.random.seed(seed)
    
    unique = []
    n_samples = []
    for i in range(len(cond_df.columns)):
        
        temp_li = cond_df[cond_df.columns[i]].unique()
        temp_li.sort()
        unique.append(temp_li)
        
        temp_li = []
        for j in range(len(unique[i])):
            temp_li.append(
                sum(cond_df[cond_df.columns[i]]==unique[i][j])
                )
        n_samples.append(temp_li)
    #return unique, n_samples    
    
    
    for i in range(len(unique)):
        df = pd.DataFrame()
        for j in range(len(unique[i])):
            print(means[i][j])
            print(covs[i][j])
            temp_df, temp_info = multivariate_df(n_samples[i][j], means[i][j], covs[i][j])
            df = pd.concat((df, temp_df))
            ## info Handling
            
        cond_df = pd.concat((cond_df, df), axis = 1)
    return df, info
    
    

def multinomial(n_samples, probabilities, seed=False, name='feature_'):
    # n_samples: int ,  probabilites = nested list of probabilities
    info = {}
    if seed:
        np.random.seed(seed)

    column_names = col_name_gen(len(probabilities), name)
    df = pd.DataFrame(columns=column_names)

    count = 0
    for prob in probabilities:
        temp_label_names = col_name_gen(len(prob), name+str(count)+'_label_')
        temp_data = np.random.choice(temp_label_names, size=n_samples, p=prob)

        df[column_names[count]] = temp_data
        info[column_names[count]] = prob
        count += 1
    return df, info


def multinomial_cond(n_samples, ind_probabilities, cond_probabilities, seed=False):
    # n_samples: int,
    # ind_probabilities : nested list of probabilities one per feature,
    # cond_probabilities : double nested list of probabilities with [0][0] representing p(cond_feature_1|ind_feature_1=label_1)
    
    # Example Use:
    # multinomial_cond(20, [[0.5, 0.5],[0.5, 0.5]], [
    #     [
    #         [
    #             [1, 0],[0, 1]
    #         ], [
    #             [0, 1],[1, 0]
    #         ]
            
    #     ], [
    #         [
    #             [1, 0],[0, 1]
    #         ], [
    #             [0, 1],[1, 0]
    #         ]
            
    #     ]
    # ])
    
    
    info = {}
    if seed:
        np.random.seed()

    ind_df, ind_info = multinomial(
        n_samples, ind_probabilities, seed, 'ind_feat_')
    cond_df = pd.DataFrame()

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
                temp_n, cond_probabilities[i][j], seed, 'cond_feat_'+str(i))
            temp_li1.append(temp_df)
            temp_li2.append(temp_info)

        temp = pd.concat(temp_li1)
        cond_df = pd.concat((cond_df, temp), axis=0)
        cond_df = cond_df.reset_index(drop=True)
        ind_df = pd.concat([ind_df, cond_df], axis=1)

    return ind_df, info


def multinomial_cond_extension(n_samples, true_ind_prob, ind_prob, cond_prob, seed = False):
    ##Used to add truly independent multinomials to a conditional set
    info = {}
    if seed:
        np.random.seed()
    
    true_ind_df, true_ind_info = multinomial(n_samples, true_ind_prob, seed, 'true_ind_')    
    cond_df, cond_info = multinomial(cond, ind_prob, cond_prob, seed)
    
    df = pd.concat((true_ind_df, cond_df))
    return df, info
  
  
    

def col_name_gen(num_cols, common_name):
    common_name_list = [common_name]*num_cols
    num_string_list = [num_string for num_string in map(
        lambda num: str(num), [num for num in range(num_cols)]
    )]
    res_list = [a + b for a, b in zip(common_name_list, num_string_list)]
    return res_list



