import pandas as pd
import re
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from operator import add
from functools import reduce
pd.set_option('display.max_colwidth',-1)
from tf_idf import ngrams,awesome_cossim_top,get_matches_df

def pro_processor(data):
    data=pd.DataFrame(data)
    data['ADD_tot'] = data["ADD_tot"].astype('str').replace('', ".")
    data = data[data['Length_of_Add'] >5]
    _data = data.groupby('Numbers')
    final= pd.DataFrame(columns=['Customer_ID', "Group_ID"])
    for group in _data.groups:
        d = pd.DataFrame(_data.get_group(group))
        d = d.reset_index()
        if len(d) <= 1:
            continue
        else:
            print("length of group", len(d), group)
            addresses = d['ADD_tot']
            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
            tf_idf_matrix = vectorizer.fit_transform(addresses)
            t1 = time.time()
            matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10000, 0.8)
            t = time.time() - t1
            print("SELFTIMED:", t)
            matches_df = get_matches_df(matches, addresses, d, top=d.shape[0])
            # matches_df = matches_df[matches_df['similairity'] < 0.99999]  # Remove all exact matches
            matches_df['index1'] = matches_df['index1'].astype(int).astype(str)
            matches_df['index2'] = matches_df['index2'].astype(int).astype(str)
            s = matches_df['index1'].value_counts()
            index = s[s > 1].index
            data01 = matches_df[matches_df['index1'].isin(index)]
            new_data = data01.groupby('index1')[['index2']].apply(lambda x: x.values.tolist())
            new_data = new_data.reset_index()
            print(len(new_data))
            if len(new_data) == 0:
                continue
            new_data.columns = ['CUST_ID', 'List_of_Cust_ID']
            new_data['List_of_Cust_ID'] = new_data['List_of_Cust_ID'].apply(lambda x: reduce(add, x))
            a = []
            index = []
            for i in range(len(new_data['CUST_ID'])):
                if i in index:
                    continue
                b = new_data['List_of_Cust_ID'][i]
                for j in new_data['List_of_Cust_ID'][i]:
                    for k in range(i + 1, len(new_data['CUST_ID'])):
                        if j in new_data['List_of_Cust_ID'][k]:
                            index.append(k)
                            for l in new_data['List_of_Cust_ID'][k]:
                                if l not in b:
                                    b.append(l)
                a.append(b)
            import random
            groups = random.sample(range(2000, 100000), 24)
            d = dict(zip(groups, a))
            dict1 = {}
            for i, j in d.items():
                for k in range(len(j)):
                    dict1[j[k]] = i
            final_data = pd.DataFrame.from_dict(dict1, orient='index').reset_index()
            final_data.columns = ['Customer_ID', "Group_ID"]
            final=final.append(final_data)
            print(final_data)
    return final
