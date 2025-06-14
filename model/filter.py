# En este archivo limpiamos los datos y los convertimos a un formato parquet

import pandas as pd

path = "model/train_data.sas7bdat"

df = pd.read_sas(path, format="sas7bdat", encoding="latin1")


df.to_parquet("model/train_data.parquet", index=False)
