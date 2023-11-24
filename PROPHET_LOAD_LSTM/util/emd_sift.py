#%%
import pandas as pd
import emd

def emd_sift(df,col='Load (kW)'):
    imf = emd.sift.sift(df[col].values)

    for i in range(imf.shape[1]):
        df['IMF%s'%(i+1)] = imf[:,i]    

    return df    

#%%
df = pd.read_csv('data/train_JPL_v2.csv', index_col=0, parse_dates=True)

df = emd_sift(df,'power')

df

# %%

df.to_csv('data/train_JPL_v3.csv')