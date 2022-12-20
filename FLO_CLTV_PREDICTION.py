# BG-NBD veGamma-Gamma ile CLTV Tahmini

##########################################
## TASK 1:  Veriyi  Hazırlama ##
##########################################

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
df_ = pd.read_csv("FLOMusteriSegmentasyonu/flo_data_20k.csv")

#Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit= quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] =round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

# "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
df_["order_num_total_ever_online"].T.describe()
df_["order_num_total_ever_offline"].T.describe()

replace_with_thresholds(df_, "order_num_total_ever_online")
replace_with_thresholds(df_, "order_num_total_ever_offline")
replace_with_thresholds(df_, "customer_value_total_ever_offline")
replace_with_thresholds(df_, "customer_value_total_ever_online")


# Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df_["order_num_total_ever_omnichannel"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
df_["customer_value_total_ever_omnichannel"] = df_["customer_value_total_ever_offline"] + df_["customer_value_total_ever_online"]

# Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
import datetime as dt
date_columns = df_.columns[df_.columns.str.contains("date")]
df_[date_columns] = df_[date_columns].apply(pd.to_datetime)

################################################
## TASK 2:  CLTV Veri Yapısının Oluşturulması ##
################################################

#Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df_["last_order_date"].max() # alışveriş yapılan son tarih
determined_date = dt.datetime(2021,6,1) # 2 gün sonrası


#customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin
# yer aldığı yeni bir cltv dataframe'i oluşturunuz. Monetary değeri satın alma başına
# ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

df_["T"] = (determined_date-df_["first_order_date"]).astype('timedelta64[D]')
cltv_df_ =pd.DataFrame()
cltv_df_["customer_id"] = df_["master_id"]
cltv_df_["recency_cltv_weekly"] = (df_["last_order_date"]-df_["first_order_date"]).astype('timedelta64[D]') / 7
cltv_df_["T_weekly"] = df_["T"] / 7
cltv_df_["frequency"] =  df_["order_num_total_ever_omnichannel"]
cltv_df_["monetary_cltv_avg"] = df_["customer_value_total_ever_omnichannel"]/df_["order_num_total_ever_omnichannel"]

cltv_df_.head()
############################################################################
TASK 3:  BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
############################################################################

# BG/NBD modelinifit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df_['frequency'],
        cltv_df_['recency_cltv_weekly'],
        cltv_df_['T_weekly'])

#• 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
bgf.predict(4 * 3,
            cltv_df_['frequency'],
            cltv_df_['recency_cltv_weekly'],
            cltv_df_['T_weekly']).sum()

cltv_df_["expected_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df_['frequency'],
                                               cltv_df_['recency_cltv_weekly'],
                                               cltv_df_['T_weekly'])
#• 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarakcltv dataframe'ineekleyiniz.
bgf.predict(4 * 6,
            cltv_df_['frequency'],
            cltv_df_['recency_cltv_weekly'],
            cltv_df_['T_weekly']).sum()

cltv_df_["expected_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df_['frequency'],
                                               cltv_df_['recency_cltv_weekly'],
                                               cltv_df_['T_weekly'])

cltv_df_[["expected_sales_3_month","expected_sales_6_month"]]

#Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df_['frequency'], cltv_df_['monetary_cltv_avg'])

cltv_df_["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df_['frequency'],
                                                                                 cltv_df_['monetary_cltv_avg'])

#6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df_['frequency'],
                                       cltv_df_['recency_cltv_weekly'],
                                       cltv_df_['T_weekly'],
                                       cltv_df_['monetary_cltv_avg'],
                                       time=6,  # 6 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)
cltv_df_['cltv']= cltv
#• Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv.sort_values(ascending=False).head(20)
######################################################
TASK 4:  CLTV Değerine Göre Segmentlerin Oluşturulması
######################################################
#6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve
# grup isimlerini veri setine ekleyiniz.
cltv_df_ = cltv_df_.reset_index()

cltv_df_["segment"] = pd.qcut(cltv_df_["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df_.head(20)

#4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon
# önerilerinde bulununuz.

# A segmenti için en yüksek değerlere sahip gruptur. Bu gruptaki müşterilere özel indirim,
#hediye çeki gibi düzenlemeler yapılabilir. Bunu yanında, workshop,belirli limitte premium
#kart gibi şeyler verilebilir.Müşteri potansiyeline göre tiyatro,sinema gibi aktivitelere
#bedava bileti verilebilir.

# C segmenti için alışveriş yapmaya teşvik edici aksiyonlar alınmalıdır. Mesela,bu segmentteki
# müşterilerin en çok satın aldığı kategoride onlara özel indirim yapılabilir. Mail ya da
#mesaj üzerinden hatırlatma mesajı atılabilir. Hesaplarına hediye kuponu eklenebilir.