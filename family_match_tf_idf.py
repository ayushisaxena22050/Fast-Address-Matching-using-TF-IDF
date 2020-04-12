import pandas as pd
import numpy as np
from group_by_tf_idf import pro_processor
from tf_idf import cleaning
pd.options.display.max_colwidth = 100
import time
from multiprocessing import Pool
start_time = time.time()
if __name__ == '__main__':
    dt = {
        "NAM_CUST_SHRT": str,
        'ADD_tot': str,
        "TXT_CUSTADR_ZIP": str,
    }
    print("Started")
    data01 = pd.read_csv(r"data.csv", skiprows=0,dtype=dt)
    data01.dropna(subset=['TXT_CUSTADR_ZIP'], inplace=True)
    data01=data01[data01["TXT_CUSTADR_ZIP"].str.isdigit()]
    data01['TXT_CUSTADR_ZIP'] = pd.to_numeric(data01['TXT_CUSTADR_ZIP'], errors='coerce').dropna()
    print(len(data01))
    gk=data01.groupby("TXT_CUSTADR_ZIP")
    print(data01.shape)
    print('Cleansing Done')
    master_df=pd.DataFrame(columns=['Customer_ID', "Group_ID"])
    print("Total pincode",len(gk.groups))
    print(data01["TXT_CUSTADR_ZIP"].value_counts())
    Total_groups=len(gk.groups)
    i=0

    for group in gk.groups:
        print("Pincode Number",group)
        print("Pincode Left:",Total_groups-i)
        gk_302034=gk.get_group(group)
        gk_302034=gk_302034.reset_index()
        gk_302034=cleaning(gk_302034)
        print("length of pincode",group ,"is",len(gk_302034))
        if gk_302034.shape[0]<=1:
            i+=1
            continue
        df = pro_processor(gk_302034)
        if df is not None:
            master_df=pd.concat([master_df,df])
            i+=1
    master_df.to_csv(r"D:\Delhi_Groups\All\Final_output.csv",index=False)
print("--- %s seconds ---" % (time.time() - start_time))

