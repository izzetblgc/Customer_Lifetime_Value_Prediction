import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Online Retail II isimli veri seti İngiltere merkezli online bir satış
# mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını
# içeriyor

df_= pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape

### Veri Setinin Modele Hazırlanması ###

df.describe().T
df.isnull().sum()
df.dropna(inplace=True)

### İade edilen ürünler faturada C ile ifade edilmiştir. Bu ürünleri çıkaralım ###

df = df[~df["Invoice"].str.contains("C",na=False)]
df = df[df["Quantity"]>0]

### Aykırı değerleri düzenleyelim ###

replace_with_thresholds(df, "Quantity")

replace_with_thresholds(df, "Price")

### Müşterilerin harcadığı toplam harcamanın veri setine eklenmesi ###

df["TotalPrice"] = df["Quantity"] * df["Price"]

### CLTV metriklerinin oluşturulması ###

today_date = dt.datetime(2011,12,11)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                              lambda x: (today_date - x.min()).days],
                               "Invoice": lambda x: x.nunique(),
                               "TotalPrice": lambda x: x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]


cltv_df = cltv_df[cltv_df["frequency"]>1]
cltv_df = cltv_df[cltv_df["monetary"] > 0]


cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7


### BG/NBD Modeli Kurulumu ###

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

### 1 haftalık satın alma miktarını tahmin edelim ###

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

### 1 aylık satın alma miktarını tahmin edelim ###

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

### GAMMA-GAMMA Modelinin Kurulumu ###

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

## Ortalama Kar Tahmini ###
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

### 6 aylık CLTV Prediction ###


cltv_6_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_6_month = cltv_6_month.reset_index()
cltv_final_6_month = cltv_df.merge(cltv_6_month, on="Customer ID", how="left")

### CLV değerlerini 0 ile 5 arasında ölçeklendirelim ###

scaler = MinMaxScaler(feature_range=(0, 5))
scaler.fit(cltv_final_6_month[["clv"]])
cltv_final_6_month["scaled_clv"] = scaler.transform(cltv_final_6_month[["clv"]])
cltv_final_6_month.sort_values(by="scaled_clv", ascending=False).head(30)


### 6 aylık CLTV değerlerinin segmentlere ayrılması ###

cltv_final_6_month["segment"] = pd.qcut(cltv_final_6_month["scaled_clv"], 4 , ["D","C","B","A"])

### Segment Analizi ###

cltv_final_6_month.groupby("segment").agg(["mean","count","sum"])









