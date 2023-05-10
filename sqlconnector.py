#data preprocessing
import numpy as np
import pyodbc
import pandas as pd

#utils/visualization
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import dill as pickle
from ydata_profiling import ProfileReport
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import itertools
import functools
import operator 
from more_itertools import flatten
from sqlalchemy.engine import URL
from sqlalchemy import create_engine

import sqlalchemy as sa
import urllib
from sqlalchemy import text
from collections import Counter
#config vars
pd.options.display.max_columns=1000
pd.options.display.max_rows = 10000
pd.options.display.max_seq_items = 100
verbose=1
tenant = '0001ai'

def connect_to_db_better(connection_string,
                    database,
                    driver = 'SQL Server Native Client 11.0',
                    user = 'CCAdmin',
                    password = 'Miw6RjnTGmPHLYF9mG1o'
):
    odbc_str = 'DRIVER='+driver+';SERVER='+connection_string+';PORT=1433;UID='+user+';DATABASE='+ database + ';PWD='+ password
    connect_str = 'mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(odbc_str)

    return connect_str


def get_table_data_CCLOG(table_name, connect_str, startdate, enddate):
    engine = create_engine(connect_str)
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM [dbo].["+table_name+"] where LogTime >= "+startdate+" and LogTime <= "+enddate+''""), conn)
        return df

def get_table_data_ALL(table_name, connect_str):
    engine = create_engine(connect_str)
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM [dbo].["+table_name+"]"), conn)
        return df


def split_datframe_into_whatever(splitkey, df_new):
    df_split = df_new.loc[(df_new['Attribute_Name'] == splitkey)] 

    def split_dataframe(df_debitornum, column):
        split_dfs = {}
        for value in df_debitornum[column].unique():
            split_dfs[value] = df_debitornum[df_debitornum[column] == value][['DocumentID']]   
        return split_dfs

    dc_split_attribute_after = split_dataframe(df_split, 'Attribute_After')


    def find_corresponding_values(dict_of_dataframes, column_to_match, large_dataframe):
        result_dict = {}
        for key, df in dict_of_dataframes.items():
            temp_df = large_dataframe[large_dataframe[column_to_match].isin(df[column_to_match])]
            result_dict[key] = temp_df
        return result_dict

    dc_sorted_df = find_corresponding_values(dc_split_attribute_after, "DocumentID", df_new)
    return dc_sorted_df
    #dc_sorted_df_by_creditor[""].to_csv('data/codia.csv', index=False, header= True, encoding='utf-8')

def filter_df_by_time(df, start_date, end_date):
    # Filtern des DataFrames nach dem Zeitfenster
    filtered_df = df[(df['LogTime'] >= start_date) & (df['LogTime'] <= end_date)]  
    df_new = filtered_df.drop(['LogTime'], axis =1)
    # Rückgabe des gefilterten DataFrames
    df_new.to_csv('data/cclogattributes_T_'+tenant+'_reduced.csv', index=False, header= True, encoding='utf-8')#iso-8859-15
    return df_new

def get_data_values2(df, items): 
    df_ordernum_correct = df.loc[(df['Attribute_Name'] == items[0]) &(df['Delta'] == items[1])] 
    documentid_list = df_ordernum_correct[['DocumentID']].values.tolist()
    merged = list(itertools.chain.from_iterable(documentid_list))
    denonimator_ordernum = len(documentid_list)
    
    df_debitor_ordernum = df.loc[(df['Attribute_Name'] == items[0])|(df['Attribute_Name'] == items[2])]

    counter_do=0
    for entry_1 in merged:
        speicher_entry_new = df_debitor_ordernum[df_debitor_ordernum.DocumentID == ''.join(str(entry_1))]
        if ((speicher_entry_new.Attribute_Name == items[0]) & (speicher_entry_new.Delta == items[1])).any(): 
            if ((speicher_entry_new.Attribute_Name == items[2])&(speicher_entry_new.Delta == False)).any():
                    counter_do += 1

    try:
        #return counter_do/denonimator_ordernum
        return counter_do/denonimator_ordernum,denonimator_ordernum
    except:
         return pd.NA, pd.NA


def get_data_values_two_booleans2(df, items):    
    counter_vom=0

    df_order_mandant_delta = df.loc[(df['Attribute_Name'] == items[0])&(df['Delta'] == items[1])]
    df_order_delta_new = df_order_mandant_delta.loc[(df_order_mandant_delta['Attribute_Name'] == items[2])&(df['Delta'] == items[3])]
    df_order_delta_mandant_documentid = df_order_delta_new[['DocumentID']]
    order_delta_mandant_list = df_order_delta_mandant_documentid.values.tolist()
    order_delta_mandant_list_merged = list(itertools.chain.from_iterable(order_delta_mandant_list))
    denominator_order_delta_mandant_num = len(order_delta_mandant_list_merged)


    df_invoicenumber_order_delta_mandant = df.loc[(df['Attribute_Name'] == items[0])|(df['Attribute_Name'] == items[2])|(df['Attribute_Name'] == items[4])]
    for entry_order_delta_mandant in order_delta_mandant_list_merged:
        speicher_entry_order_delta_mandant = df_invoicenumber_order_delta_mandant[df_invoicenumber_order_delta_mandant.DocumentID == ''.join(str(entry_order_delta_mandant))]
        if (((speicher_entry_order_delta_mandant.Attribute_Name == items[0]) & (speicher_entry_order_delta_mandant.Delta == items[1]))&((speicher_entry_order_delta_mandant.Attribute_Name == items[2]) & (speicher_entry_order_delta_mandant.Delta == items[3]))).any(): 
            if ((speicher_entry_order_delta_mandant.Attribute_Name == items[4]) & (speicher_entry_order_delta_mandant.Delta == False)).any():
                    counter_vom += 1

    #if((counter_vom == 0) | (denominator_order_delta_mandant_num == 0 & counter_vom == 0)):
    try:
        #return counter_vom/denominator_order_delta_mandant_num
        return counter_vom/denominator_order_delta_mandant_num, denominator_order_delta_mandant_num
    except:
         return pd.NA , pd.NA

def get_single_value(df, value):
    df_ordernum = df.loc[(df['Attribute_Name'] == value)] 
    dist = df_ordernum['Delta'].value_counts(normalize=True)
    try:
        score_false = dist.loc[False]
        if(score_false >= 0):
            score = score_false
        elif(score_false == 0):
            score = dist.loc[True]
    except:
        return pd.NA , pd.NA

        
    return score, len(df_ordernum)


list1 = [['OrderNum',False,'DEBITOR_NUM'],
         ['OrderNum',False,'VENDOR_NUM'],
         ['OrderNum',True,'DEBITOR_NUM'],
         ['VENDOR_NUM',False,'InvoiceNumber'],
         ['VENDOR_NUM',True,'InvoiceNumber'],
         ['VENDOR_NUM',False,'InvoiceDate'],
         ['VENDOR_NUM',True,'InvoiceDate'],
         ['VENDOR_NUM',False,'GrossAmount'],
         ['VENDOR_NUM',True,'GrossAmount'],
         ['VENDOR_NUM',False,'NetAmount1'],
         ['VENDOR_NUM',True,'NetAmount1'],
         ['VENDOR_NUM',False,'VatAmount1'],
         ['VENDOR_NUM',True,'VatAmount1'],
         ]

list2 = [['OrderNum',True,'DEBITOR_NUM',False,'VENDOR_NUM'],
            ['OrderNum',True,'DEBITOR_NUM',True,'VENDOR_NUM'],
            ]

def get_data_values_complete_results(tenant, splitkey,df_new):

    dc_sorted_df = split_datframe_into_whatever(splitkey, df_new)
    
    df_results = pd.DataFrame(columns=['q(OrderNum)','q(VENDOR_NUM)',
            'q(DEBITOR_NUM|OrderNum)',
            'q(VENDOR_NUM|OrderNum)',
            'q(DEBITOR_NUM|!OrderNum)',
            'q(InvoiceNumber|VENDOR_NUM)',
            'q(InvoiceNumber|!VENDOR_NUM)',
            'q(InvoiceDate|VENDOR_NUM)',
            'q(InvoiceDate|!VENDOR_NUM)',
            'q(GrossAmount|VENDOR_NUM)',
            'q(GrossAmount|!VENDOR_NUM)',
            'q(NetAmount1|VENDOR_NUM)',
            'q(NetAmount1|!VENDOR_NUM)',
            'q(VatAmount1|VENDOR_NUM)',
            'q(VatAmount1|!VENDOR_NUM)',
            'q(VENDOR_NUM | !OrderNum & DEBITOR_NUM)',
            'q(VENDOR_NUM| !OrderNum & ! DEBITOR_NUM)',
            ])
    
    df_results_frequency= pd.DataFrame(columns=['q(OrderNum)','q(VENDOR_NUM)',
            'q(DEBITOR_NUM|OrderNum)',
            'q(VENDOR_NUM|OrderNum)',
            'q(DEBITOR_NUM|!OrderNum)',
            'q(InvoiceNumber|VENDOR_NUM)',
            'q(InvoiceNumber|!VENDOR_NUM)',
            'q(InvoiceDate|VENDOR_NUM)',
            'q(InvoiceDate|!VENDOR_NUM)',
            'q(GrossAmount|VENDOR_NUM)',
            'q(GrossAmount|!VENDOR_NUM)',
            'q(NetAmount1|VENDOR_NUM)',
            'q(NetAmount1|!VENDOR_NUM)',
            'q(VatAmount1|VENDOR_NUM)',
            'q(VatAmount1|!VENDOR_NUM)',
            'q(VENDOR_NUM | !OrderNum & DEBITOR_NUM)',
            'q(VENDOR_NUM| !OrderNum & ! DEBITOR_NUM)',
            ])

    for key, df in dc_sorted_df.items():
    
        result_list = []
        result_list_frequency = []
        #get_sollwerte(key, df)
        ordernum, ordernum_frequency = get_single_value(df, 'OrderNum')
        result_list.append(ordernum)
        result_list_frequency.append(ordernum_frequency)

        #print(''+key+': '+'OrderNum = '+str(ordernum)+'')
        #Für Simon ai1
        #x, y= get_data_values(df, 'VENDOR_NUM', False, 'InvoiceNumber', True)
        #print(key)
        #print(x,y)

        vendornum, vendornum_frequency = get_single_value(df, 'VENDOR_NUM')
        #print(''+key+': '+'VENDOR_NUM = '+str(vendornum)+'')
        result_list.append(vendornum)
        result_list_frequency.append(vendornum_frequency)

        for items1 in list1:
            value, value_frequency = get_data_values2(df, items1)
            result_list.append(value)
            result_list_frequency.append(value_frequency)

        for items2 in list2:
            twovalues, twovalues_frequency = get_data_values_two_booleans2(df, items2)
            result_list.append(twovalues)
            result_list_frequency.append(twovalues_frequency)
            

        df_results.loc[key] = result_list
        df_results_frequency.loc[key] = result_list_frequency

        
    return df_results, df_results_frequency ,dc_sorted_df


def first_element(row):
    return row[0]

def get_outliers(df_results):
    outliers_dataframe = pd.Series([],dtype=pd.StringDtype())

    for col in sollwerte_transposed:
        #print(sollwerte_transposed[col])
        std = df_results[col].std()
        sollwert = sollwerte_transposed.iloc[0][col]
        gewichtung = sollwerte_transposed.iloc[1][col]
        absoluter_abstand = np.abs(sollwert - df_results[col])
        outliers = df_results[(df_results[col] - sollwert).abs() > 0.3] #2 * std ((1-sollwert)/2)
        #print(outliers.index.tolist())
        if not outliers.empty:
            outliers_dataframe[col] = outliers.index.tolist()
            #print(f"Ausreißer in Spalte {col}:")
            #print(outliers.index)
            #return(col, outliers.index)

    return(outliers_dataframe)

#median = df_results.median()

def sort_outliers(outliers_results, df_results_frequency):
    sorted_outliers_dataframe = pd.Series([],dtype=pd.StringDtype())
    for entry in outliers_results.items():
        filtered_df2 = df_results_frequency[df_results_frequency.index.isin(entry[1])]
        sorted_filtered_df2 = filtered_df2.sort_values(by=entry[0], ascending=False)
        sorted_filtered_df2 = sorted_filtered_df2[entry[0]]
        sorted_outliers_dataframe[entry[0]]  = sorted_filtered_df2.index.tolist()
    #print(df_results_debitor_frequency[entry[0]].sort_values(entry[1]))
    return sorted_outliers_dataframe
    #entry[1]


def sort_numbers_by_position(df, sollwerte_transposed ,col_name, threshold):
    count_numbers = {}
    positions = {}
    for i, row in df.iterrows():
        gewichtung = sollwerte_transposed.iloc[1][i]
        for number in row[col_name]:
            if number in count_numbers:
                count_numbers[number] += 1 * gewichtung
            else:
                count_numbers[number] = 1
            if number in positions:
                positions[number].append(row[col_name].index(number))
            else:
                positions[number] = [row[col_name].index(number)]
    
    sorted_counts = sorted(count_numbers.items(), key=lambda x: sum(positions[x[0]]) / len(positions[x[0]]))
    sorted_numbers = [x[0] for x in sorted_counts]
    
    threshold_numbers = [x[0] for x in sorted_counts if x[1] > threshold]
    
    return sorted_numbers, threshold_numbers



sollwerte = pd.read_csv('data/sollwerte.csv', encoding='utf-8')
sollwerte_transposed = sollwerte.set_index('Unnamed: 0').T

startdate = "'2022-12-20 09:44:23.030'"
enddate = "'2023-04-19 12:28:20.000'"

df_table_cclogattributes = get_table_data_CCLOG('CCLogAttributes', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+'')
                                           ,""+startdate+"",""+enddate+"")
df_table_cclogattributes = df_table_cclogattributes.drop(['Zone','LogTime','Attribute_DataType','LogTimeTicks'], axis=1)
df_table_cclogattributes = df_table_cclogattributes.replace('\n',' ', regex=True)
df_table_cclogattributes = df_table_cclogattributes.replace('\r',' ', regex=True)
#df_table_cclogattributes.to_csv('data/cclogattributes_T_'+tenant+'.csv', index=False, header= True, encoding='utf-8')#iso-8859-15


#df1 = pd.read_csv('data/cclogattributes_T_'+tenant+'.csv', encoding='utf-8')

#df_new = filter_df_by_time(df_table_cclogattributes,'2022-12-20 09:44:23.030','2023-04-19 12:28:20.000')

score_card = pd.Series([],dtype=pd.StringDtype())
sollwerte = pd.read_csv('data/sollwerte.csv', encoding='utf-8')
sollwerte_transposed = sollwerte.set_index('Unnamed: 0').T

df_results_debitor,df_results_frequency ,dc_sorted_df_debitor = get_data_values_complete_results(tenant, 'DEBITOR_NUM', df_table_cclogattributes)
outliers_results = get_outliers(df_results_debitor)
outliers_results_sorted_debitor = sort_outliers(outliers_results, df_results_frequency)
outliers_results_debitor_frame = outliers_results_sorted_debitor.to_frame()
sorted_counts, high_frequency_numbers_debitor = sort_numbers_by_position(outliers_results_debitor_frame, sollwerte_transposed,0, threshold=3)

bad_vendors = []
for entry in high_frequency_numbers_debitor:
    df_results_vendor, df_results_frequency_vendor, dc_sorted_df_vendor = get_data_values_complete_results(tenant, 'VENDOR_NUM', dc_sorted_df_debitor[entry])
    outliers_results_vendor = get_outliers(df_results_vendor)
    outliers_results_sorted_vendor = sort_outliers(outliers_results_vendor, df_results_frequency_vendor)
    outliers_results_vendor_frame = outliers_results_sorted_vendor.to_frame()
    sorted_counts_vendor, high_frequency_numbers_vendor = sort_numbers_by_position(outliers_results_vendor_frame, sollwerte_transposed, 0, threshold=3)
    bad_vendors.append(high_frequency_numbers_vendor)
    score_card[entry]  = high_frequency_numbers_vendor
    
df_table_ccvendors = get_table_data_ALL('CC_VENDORS', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+''))
df_table_ccvendors = df_table_ccvendors.replace('\n',' ', regex=True)
df_table_ccvendors = df_table_ccvendors.replace('\r',' ', regex=True)

count = 0
score_card_missing_vendor_vat_registration_id = pd.Series([],dtype=pd.StringDtype())

for entry in score_card:
    wrong_number = []
    for item in entry:
        entry_wrong_number = df_table_ccvendors[(df_table_ccvendors['COMPANY_NUM'] == score_card.index[count]) & (df_table_ccvendors["VENDOR_NUM"] == item)& (df_table_ccvendors["VENDOR_VAT_REGISTRATION_ID"] == '')] 
        if not entry_wrong_number.empty:
            wrong_number.append(entry_wrong_number['VENDOR_NUM'].item())
            score_card_missing_vendor_vat_registration_id[score_card.index[count]] = wrong_number

    count += 1
    
print(score_card_missing_vendor_vat_registration_id)
