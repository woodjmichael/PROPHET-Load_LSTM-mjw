import pandas as pd
import emd

def emd_sift(df,col='Load (kW)'):
    imf = emd.sift.sift(df[col].values)

    for i in range(imf.shape[1]):
        df['IMF%s'%(i+1)] = imf[:,i]    

    return df    

df = pd.read_csv('data\Impianto_4_clean.csv', index_col=0, parse_dates=True,comment='#')

df = emd_sift(df,'Potenza')

print(df)

print(df.corr().loc['Potenza'])

df.to_csv('data\Impianto_4_clean_emd.csv')