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
def awesome_cossim_top(A, B, ntop, lower_bound=0):
    A = A.tocsr()

    B = B.tocsr()

    M, _ = A.shape

    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop
    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    #     print(data)
    #     print(B.data)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr,
        indices,
        data)
    return csr_matrix((data, indices, indptr), shape=(M, N))

def get_matches_df(sparse_matrix, name_vector, _data,top=100):
    non_zeros = sparse_matrix.nonzero()
    sparserows = non_zeros[0]
    sparsecols =non_zeros[1]
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    index1=np.zeros(nr_matches)
    index2=np.zeros(nr_matches)
    print(nr_matches)
#     print(name_vector[sparserows[6]])
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
        index1[index] =_data['COD_CUST_ID'][sparserows[index]]
        index2[index]=_data['COD_CUST_ID'][sparsecols[index]]
    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similairity': similairity,
                        "index1":index1,
                        "index2":index2})

def ngrams(string, n=3):
    string = re.sub(r'[,-/]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def cleaning(data01):
    data01['NAM_CUSTADR_CITY'] = data01['NAM_CUSTADR_CITY'].astype('str')
    data01['NAM_CUSTADR_STATE'] = data01['NAM_CUSTADR_STATE'].astype('str')
    # data01 = data01[data01['NAM_CUSTADR_STATE'] == 'MAHARASHTRA']
    # data01.dropna(subset=['TXT_CUSTADR_ZIP'], inplace=True)
    data01['TXT_CUSTADR_ZIP'] = pd.to_numeric(data01['TXT_CUSTADR_ZIP'], errors='coerce')
    data01['COD_CUST_ID'] = pd.to_numeric(data01['COD_CUST_ID'], errors='coerce')
    data01['ADD_tot'] = data01['ADD_tot'].astype('str').replace('nan', ".")
    data01['NAM_CUSTADR_CITY'] = data01['NAM_CUSTADR_CITY'].apply(lambda x: x.lower())
    data01['ADD_tot'] = data01['ADD_tot'].apply(lambda x: x.lower())
    data01["ADD_tot"] = data01[["TXT_CUSTADR_ZIP", "ADD_tot", "NAM_CUSTADR_CITY"]].apply(
        lambda x: str(x["ADD_tot"]).replace(str(x["TXT_CUSTADR_ZIP"]), "").replace(str(x["NAM_CUSTADR_CITY"]), ""),
        axis=1)
    data01["ADD_tot"] = data01["ADD_tot"].apply(lambda x: x.replace(",", "").replace(".", "").replace("-", ""))
    data01["ADD_tot"] = data01["ADD_tot"].apply(lambda x: x.lower())
    data01['AAA'] = data01['ADD_tot']
    data01['Numbers'] = data01['ADD_tot'].apply(lambda x: list(filter(str.isdigit, x)))
    data01['Numbers'] = data01.Numbers.apply(lambda x: sorted(x))
    data01['Numbers'] = data01['Numbers'].apply(lambda x: "".join(x))
    data01['ADD_tot'] = data01['ADD_tot'].apply(lambda x: x.lstrip().rstrip())
    data01['ADD_tot'] = data01['ADD_tot'].apply(lambda x: str(x).replace(" ", ""))
    data01['ADD_tot'] = data01['ADD_tot'].apply(lambda x: "".join(list(filter(str.isalpha, x))))
    data01['Length_of_Add'] = data01['ADD_tot'].apply(lambda x: len(x))
    return data01

def read_dump():
    dt = {
        "COD_CUST_ID": str,
        "Last_name": str,
        "NAM_CUST_SHRT": str,
        'ADD_tot': str,
        "TXT_CUSTADR_ZIP": str,
        "REF_PHONE_MOBILE": str,
        "NOMINEE_PHONE_NUMBER": str,
        "LOAN_ID": str,
        "NAM_CUSTADR_CITY": str,
        "NAM_CUSTADR_STATE": str,
        "COD_CC_BRN": str,
        "COD_ACCT_NO": str
    }
    print("Reading Dump")
    data01 = pd.read_csv(r"D:\Delhi_Groups\final_data\NODUP_CLEAN_DATA.csv", skiprows=0, dtype=dt)
    data01.drop_duplicates(subset="COD_CUST_ID", inplace=True)
    columns = ["COD_CUST_ID", "Last_name", "NAM_CUST_SHRT", 'ADD_tot', "TXT_CUSTADR_ZIP", "REF_PHONE_MOBILE",
               "NOMINEE_PHONE_NUMBER", "LOAN_ID", "NAM_CUSTADR_CITY", "NAM_CUSTADR_STATE", "COD_CC_BRN",
               "COD_ACCT_NO"]
    data01.columns = columns
    data01.dropna(subset=['TXT_CUSTADR_ZIP'], inplace=True)
    data01 = data01[data01["TXT_CUSTADR_ZIP"].str.isdigit()]
    data01['TXT_CUSTADR_ZIP'] = pd.to_numeric(data01['TXT_CUSTADR_ZIP'], errors='coerce').dropna()
    print(len(data01))
    return data01