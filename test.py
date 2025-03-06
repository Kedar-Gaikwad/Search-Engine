import pandas as pd
# path=(r"D:\data and text mining\AS4\archive\dataset\train.csv")
# df=pd.read_csv(path)
# print(df.columns)

url = 'https://mystoragekedar.blob.core.windows.net/dataset/train.csv?sp=racw&st=2025-03-06T21:16:27Z&se=2025-03-07T05:16:27Z&spr=https&sv=2022-11-02&sr=b&sig=2NDQLc9AaIOWwKPAKTQVNPH7E5clugqGk3qd7gIbUo4%3D'

df = pd.read_csv(url)
print(df.head())