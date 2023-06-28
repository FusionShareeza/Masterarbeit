#data preprocessing
import numpy as np
import pyodbc
import pandas as pd
import csv 

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
from json import dumps
from flask_jsonpify import jsonify
import json
import os
import os.path
import sqlalchemy as sa
import urllib
from sqlalchemy import text
from collections import Counter
from collections import defaultdict
from datetime import datetime 

tenant = '0009a5'
path = 'data/'+tenant+''
startdate = "'2022-11-01 09:44:23.030'"
enddate = "'2023-06-21 13:28:20.000'"

try: 
    os.mkdir(path) 
except: 
    print('schon da') 

if(os.path.exists('data/'+str(tenant)+'/log_'+str(tenant)+'_.csv')  == True):
    print('log da')
else:
    with open('data/'+str(tenant)+'/log_'+str(tenant)+'_.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Logtime", "ValueMandant", "ValueLieferant", "ValueRechnungskopf"])
    print('log erstellt')

critical = -1

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
    
def get_table_data_CCLOGAUTOTRAIN(table_name, connect_str, startdate, enddate):
    engine = create_engine(connect_str)
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM [dbo].["+table_name+"] where CONVERT(datetime2, STOP_TIME, 104) BETWEEN "+startdate+" and "+enddate+""), conn)
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
    df_new.to_csv('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_.csv', index=False, header= True, encoding='utf-8')
    return df_new

def get_data_values2(df, items): 
    df_ordernum_correct = df.loc[(df['Attribute_Name'] == items[0]) &(df['Delta'] == items[1])] 
    documentid_list = df_ordernum_correct[['DocumentID']].values.tolist()
    merged = list(itertools.chain.from_iterable(documentid_list))
    denominator_ordernum = len(documentid_list)
    
    df_debitor_ordernum = df.loc[(df['Attribute_Name'] == items[0])|(df['Attribute_Name'] == items[2])]

    counter_do=0
    for entry_1 in merged:
        speicher_entry_new = df_debitor_ordernum[df_debitor_ordernum.DocumentID == ''.join(str(entry_1))]
        if ((speicher_entry_new.Attribute_Name == items[0]) & (speicher_entry_new.Delta == items[1])).any(): 
            if ((speicher_entry_new.Attribute_Name == items[2])&(speicher_entry_new.Delta == False)).any():
                    counter_do += 1


    try:
        return counter_do/denominator_ordernum,denominator_ordernum
    except ZeroDivisionError:
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

    try:
        return counter_vom/denominator_order_delta_mandant_num, denominator_order_delta_mandant_num
    except:
         return pd.NA , pd.NA
    
def get_single_value(df, value):
    df_ordernum = df.loc[(df['Attribute_Name'] == value)] 
    dist = df_ordernum['Delta'].value_counts(normalize=True)

    try: 
        score_false = dist.loc[False]
    except:
        score_false = pd.NA

    try: 
        score_true = dist.loc[True]
    except:
        score_true = pd.NA
    
    if not pd.isna(score_true):
        score = 1 - score_true
    else:
        score = score_false
    
    try:
        return score, len(df_ordernum)

    except:
        return score , pd.NA


def get_data_values_complete_results(tenant, splitkey,df_new):

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
    dc_vendors = {}
    dc_frequency = {}

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

        df_speicher = pd.DataFrame(columns=['q(OrderNum)','q(VENDOR_NUM)',
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
        
        df_speicher2 = pd.DataFrame(columns=['q(OrderNum)','q(VENDOR_NUM)',
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
        
        result_list = []
        result_list_frequency = []

        ordernum, ordernum_frequency = get_single_value(df, 'OrderNum')
        result_list.append(ordernum)
        result_list_frequency.append(ordernum_frequency)

        vendornum, vendornum_frequency = get_single_value(df, 'VENDOR_NUM')

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
            
        df_speicher.loc[key] = result_list 
        df_speicher2.loc[key] = result_list_frequency
  
        df_results.loc[key] = result_list
        df_results_frequency.loc[key] = result_list_frequency


        
        dc_vendors[key] = df_speicher    
        dc_frequency[key] = df_speicher2
        
    return df_results, df_results_frequency ,dc_sorted_df, dc_vendors, dc_frequency

def save_dc_to_json(dc,name):
    def convert_dataframe_to_json(df):
        return df.replace({pd.NA: None}).to_dict(orient="records")

    def convert_dict_to_json(dictionary):
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                converted_dict[key] = convert_dict_to_json(value)
            elif isinstance(value, pd.DataFrame):
                converted_dict[key] = convert_dataframe_to_json(value)
            else:
                converted_dict[key] = value
        return converted_dict

    converted_data = convert_dict_to_json(dc)

    json_data = json.dumps(converted_data)

    with open("data/"+str(tenant)+"/"+name+"_.json", "w") as file:
        file.write(json_data)

def first_element(row):
    return row[0]

def get_outliers(df_results, sollwerte_transposed):
    outliers_dataframe = pd.Series([],dtype=pd.StringDtype())

    for col in sollwerte_transposed:
        #print(sollwerte_transposed[col])
        std = df_results[col].std()
        sollwert = sollwerte_transposed.iloc[0][col]
        gewichtung = sollwerte_transposed.iloc[1][col]
        absoluter_abstand = np.abs(sollwert - df_results[col])
        outliers = df_results[(df_results[col] - sollwert).abs() > 0.1] #2 * std ((1-sollwert)/2)
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
    
    threshold_numbers = [x[0] for x in sorted_counts if x[1] >= threshold]
    
    return sorted_numbers, threshold_numbers

def check_for_eap_error(vendor_num, debitor, dc_sorted_df_vendor_complete):  
        df_sort = dc_sorted_df_vendor_complete[debitor][vendor_num]
        df_sort_vendor = df_sort[((df_sort['Attribute_Name'] == 'VENDOR_NUM') & (df_sort['Delta'] == False))]
        documentid_list = df_sort_vendor[['DocumentID']].values.tolist()
        merged = list(itertools.chain.from_iterable(documentid_list))
        df_sorted = df_sort[df_sort['DocumentID'].isin(merged)]

        df_sort_new = df_sorted[((df_sorted['Attribute_Name'] == 'VatAmount1') & (df_sorted['Delta'] == True)) | (df_sorted['Attribute_Name'] == 'NetAmount1') & (df_sorted['Delta'] == True) | (df_sorted['Attribute_Name'] == 'VatRate1') & (df_sorted['Delta'] == True)]        
        unique_documentid = df_sort_new['DocumentID'].unique()
        without_vat = [x for x in merged if x not in unique_documentid]
        
        for document_id in without_vat:
                df_sorted = df_sort[(df_sort["DocumentID"] == document_id) & (df_sort["Delta"] == True)]     
                #display(df_sorted)     

        wrong_documentids = []
        error_codes = []

        for entry in unique_documentid:
                sorted_by_documentid = df_sort_new.loc[(df_sort_new["DocumentID"] == entry)]
                len_sorted_by_documentid = len(sorted_by_documentid)
                sorted_by_documentid = sorted_by_documentid.loc[(sorted_by_documentid["Attribute_Name"] == 'VatAmount1')]
                if not sorted_by_documentid.empty:
                        if 0.0 < abs(float(sorted_by_documentid["Attribute_After"]) - float(sorted_by_documentid["Attribute_Before"])) <= 0.05:
                                #print('Bei: '+entry+' Rundungsfehler')
                                x=1

                if not sorted_by_documentid.empty:
                        wrong_documentids.append(entry)
                        error_codes.append(len_sorted_by_documentid)

        return wrong_documentids, error_codes


def convert_date(date_string):
    input_format = "%d.%m.%Y %H:%M:%S"
    output_format = "%Y-%m-%d %H:%M:%S"
    date_object = datetime.strptime(date_string, input_format)
    converted_date = datetime.strftime(date_object, output_format)
    
    return converted_date

def check_distribution(df, changed_type, stop_time ):
    converted_date = convert_date(stop_time)
    df_before = df[df['LogTime'] < converted_date]
    df_after = df[df['LogTime'] >= converted_date]
    if not df_before.empty or not df_after.empty:
        if changed_type == 'TrustedAmount':
            changed_type = 'GrossAmount'
        if changed_type == 'invoicenumber':
            changed_type = 'InvoiceNumber'
        if changed_type == 'invoicedate':
            changed_type = 'InvoiceDate'
        distribution_before = get_single_value(df_before, changed_type)
        distribution_after = get_single_value(df_after, changed_type)

    return distribution_before, distribution_after

def get_improvement_results():
    if not os.path.isfile('data/'+str(tenant)+'/scorecard_df_'+str(tenant)+'_.json') or not os.path.isfile('data/'+str(tenant)+'/results_complete_frequency_'+str(tenant)+'_.json'):
        print("bin drin ")
        sollwerte = pd.read_csv('data/sollwerte.csv', encoding='utf-8')
        sollwerte_transposed = sollwerte.set_index('Unnamed: 0').T

        df_table_cclogattributes = get_table_data_CCLOG('CCLogAttributes', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+'')
                                                ,""+startdate+"",""+enddate+"")
        df_table_cclogattributes = df_table_cclogattributes.drop(['Zone','Attribute_DataType','LogTimeTicks'], axis=1)
        df_table_cclogattributes = df_table_cclogattributes.replace('\n',' ', regex=True)
        df_table_cclogattributes = df_table_cclogattributes.replace('\r',' ', regex=True)
        print("hier bin ich ")
        df_table_cclogattributes.to_csv('./data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv', index=False, header= True, encoding='utf-8')

        score_card = pd.Series([],dtype=pd.StringDtype())
        df_scorecard_dataframe = pd.DataFrame(columns=['Mandant','Lieferant','Fehlercode','DocumentID','MissingCode'])

        sollwerte = pd.read_csv('data/sollwerte.csv', encoding='utf-8')
        sollwerte_transposed = sollwerte.set_index('Unnamed: 0').T
        dc_sorted_df_vendor_complete = dict()
        dc_all_frequency_df_vendor_complete = {}
        dc_all_results_df_vendor_complete = {}

        df_results_debitor,df_results_frequency ,dc_sorted_df_debitor, schmutz, schmutz2 = get_data_values_complete_results(tenant, 'DEBITOR_NUM', df_table_cclogattributes)
        outliers_results = get_outliers(df_results_debitor, sollwerte_transposed)
        outliers_results_sorted_debitor = sort_outliers(outliers_results, df_results_frequency)
        outliers_results_debitor_frame = outliers_results_sorted_debitor.to_frame()
        sorted_counts, high_frequency_numbers_debitor = sort_numbers_by_position(outliers_results_debitor_frame, sollwerte_transposed,0, threshold=3)
        print("jetzt hier")
        bad_vendors = []
        count = 0
        for entry in high_frequency_numbers_debitor:
            dc_sorted_df_vendor_complete[entry] = {}
            dc_all_results_df_vendor_complete[entry] = {}
            dc_all_frequency_df_vendor_complete[entry] = {}
            df_results_vendor, df_results_frequency_vendor, dc_sorted_df_vendor, dc_vendors, dc_frequency = get_data_values_complete_results(tenant, 'VENDOR_NUM', dc_sorted_df_debitor[entry])
            dc_all_results_df_vendor_complete[entry].update(dc_vendors)
            dc_sorted_df_vendor_complete[entry].update(dc_sorted_df_vendor)
            #display(dc_frequency)
            dc_all_frequency_df_vendor_complete[entry].update(dc_frequency)
            outliers_results_vendor = get_outliers(df_results_vendor, sollwerte_transposed)
            outliers_results_sorted_vendor = sort_outliers(outliers_results_vendor, df_results_frequency_vendor)
            outliers_results_vendor_frame = outliers_results_sorted_vendor.to_frame()
            sorted_counts_vendor, high_frequency_numbers_vendor = sort_numbers_by_position(outliers_results_vendor_frame, sollwerte_transposed, 0, threshold=3)
            bad_vendors.append(high_frequency_numbers_vendor)
            print(entry)
            print(high_frequency_numbers_vendor)

            for entry_high_frequency_numbers_vendor in high_frequency_numbers_vendor:
                df_scorecard_dataframe.loc[count, 'Mandant'] = entry
                df_scorecard_dataframe.loc[count, 'Lieferant'] = entry_high_frequency_numbers_vendor 
                count += 1
                
            score_card[entry]  = high_frequency_numbers_vendor

        for entry in list(set(df_results_debitor.index) - set(high_frequency_numbers_debitor)):
            dc_sorted_df_vendor_complete[entry] = {}
            dc_all_results_df_vendor_complete[entry] = {}
            dc_all_frequency_df_vendor_complete[entry] = {}
            df_results_vendor, df_results_frequency_vendor, dc_sorted_df_vendor, dc_vendors,dc_frequency = get_data_values_complete_results(tenant, 'VENDOR_NUM', dc_sorted_df_debitor[entry])
            dc_all_results_df_vendor_complete[entry].update(dc_vendors)
            dc_sorted_df_vendor_complete[entry].update(dc_sorted_df_vendor)
            dc_all_frequency_df_vendor_complete[entry].update(dc_frequency)

        save_dc_to_json(dc_all_frequency_df_vendor_complete, 'results_complete_frequency_'+str(tenant)+'')
        save_dc_to_json(dc_all_results_df_vendor_complete, 'results_complete_'+str(tenant)+'')
        df_results_debitor["Mandant"] = df_results_debitor.index
        df_results_debitor.to_json('data/'+str(tenant)+'/results_debitor_'+str(tenant)+'_.json', orient="records")
                
        df_table_ccvendors = get_table_data_ALL('CC_VENDORS', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+''))
        df_table_ccvendors_bank = get_table_data_ALL('CC_VENDOR_BANK', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+''))

        df_table_ccvendors = df_table_ccvendors.replace('\n',' ', regex=True)
        df_table_ccvendors = df_table_ccvendors.replace('\r',' ', regex=True)
        df_table_ccvendors_bank = df_table_ccvendors_bank.replace('\n',' ', regex=True)
        df_table_ccvendors_bank = df_table_ccvendors_bank.replace('\r',' ', regex=True)

        current_mandant = None
        current_lieferant = None

        # Schleife über die Spalten des DataFrames
        for column in df_scorecard_dataframe.columns:
            if column == 'Mandant':
                current_mandant = df_scorecard_dataframe[column]
            elif column == 'Lieferant':
                current_lieferant = df_scorecard_dataframe[column]

        i = 0

        while i < len(current_mandant):
            fehlercount = 0
            wrong_documentids, error_codes = (check_for_eap_error(current_lieferant[i], current_mandant[i], dc_sorted_df_vendor_complete))
            entry_wrong_number_vat_registration_id = df_table_ccvendors[(df_table_ccvendors['COMPANY_NUM'] == current_mandant[i]) & (df_table_ccvendors["VENDOR_NUM"] == current_lieferant[i])& ~(df_table_ccvendors["VENDOR_VAT_REGISTRATION_ID"] == '')]
            entry_wrong_number_registration_id = df_table_ccvendors[(df_table_ccvendors['COMPANY_NUM'] == current_mandant[i]) & (df_table_ccvendors["VENDOR_NUM"] == current_lieferant[i])& ~(df_table_ccvendors["VENDOR_REGISTRATION_ID"] == '')] 
            entry_wrong_number_iban = df_table_ccvendors_bank[(df_table_ccvendors_bank['COMPANY_NUM'] == current_mandant[i]) & (df_table_ccvendors_bank["VENDOR_NUM"] == current_lieferant[i])& ~(df_table_ccvendors_bank["IBAN"] == '')]
            

            if entry_wrong_number_vat_registration_id.empty: 
                fehlercount += 100
            if entry_wrong_number_registration_id.empty: 
                fehlercount += 10
            if entry_wrong_number_iban.empty: 
                fehlercount += 1
            
            df_scorecard_dataframe.loc[i, 'Fehlercode'] = fehlercount
            df_scorecard_dataframe.loc[i, 'DocumentID'] = wrong_documentids
            df_scorecard_dataframe.loc[i, 'MissingCode'] = error_codes

            i+=1

        df_scorecard_dataframe.to_json('data/'+str(tenant)+'/scorecard_df_'+str(tenant)+'_.json', orient="records")
        df_scorecard_dataframe = df_scorecard_dataframe.to_json(orient="records")

        get_autotrain_variants(dc_sorted_df_vendor_complete)
        x =  '{ "here":"1"}'

        return x
    else:
        x =  '{ "here":"0"}'

        return x
        

def get_autotrain_variants(dc_sorted_df_vendor_complete):
        df_table_cclogvariants = get_table_data_ALL('CCLOGVARIANTS', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+''))
        df_table_cclogautotrain = get_table_data_CCLOGAUTOTRAIN('CCLOGAUTOTRAIN', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+''),""+startdate+"",""+enddate+"")

        df_table_cclogvariants = df_table_cclogvariants.replace('\n',' ', regex=True)
        df_table_cclogautotrain = df_table_cclogautotrain.replace('\r',' ', regex=True)
        df_table_cclogvariants = df_table_cclogvariants.replace('\n',' ', regex=True)
        df_table_cclogautotrain = df_table_cclogautotrain.replace('\r',' ', regex=True)

        df_table_cclogvariants = df_table_cclogvariants.drop(['ID', 'ATTRIB_STRUCTURE','DCMODULE_NAME'], axis=1)
        df_table_cclogvariants = df_table_cclogvariants.rename({'TRAINING_ID': 'ID'}, axis=1)

        merged_df = pd.merge(df_table_cclogautotrain, df_table_cclogvariants, on='ID')
        if(merged_df.empty == False):
            merged_df[['VENDOR', 'VENDOR_NUM', 'CHANGED_TYPE']] = merged_df['ATTRIB_PATH'].str.split('.', expand=True)

            flat_dict = defaultdict(pd.DataFrame)
            ergebnis_df = pd.DataFrame

            for mandant, lieferanten_dict in dc_sorted_df_vendor_complete.items():
                for lieferant, df in lieferanten_dict.items():
                    if not flat_dict[lieferant].empty:
                        flat_dict[lieferant] = pd.concat([flat_dict[lieferant], df])
                    else:
                        flat_dict[lieferant] = df
                    
            flat_dict = dict(flat_dict)

            for df_name, df in flat_dict.items():
                sorted_df = df.sort_values(by='LogTime', ascending=False)
                unique_ids = []


                for index, row in sorted_df.iterrows():
                    if row['DocumentID'] not in unique_ids:
                        unique_ids.append(row['DocumentID'])
                    if len(unique_ids) == 5:
                        break

                def join_entries(entry_list):
                    return ','.join(entry_list) if entry_list else ''
                    
                if str(df_name) in merged_df['VENDOR_NUM'].unique():
                    print(df_name)
                    y = merged_df[merged_df['VENDOR_NUM'] == df_name]
                    if 'inserted' in y['STATUS'].unique(): 
                        stop_time = y['STOP_TIME']
                        changed_type = y['CHANGED_TYPE']
                        for item, item2 in zip(stop_time, changed_type):
                            x = merged_df[merged_df['CHANGED_TYPE'] == item2]
                            score_before, score_after = check_distribution(df, item2, item )
                            abc = y.index
                        for thing in abc:
                            merged_df.at[thing, 'ScoreBefore'] = score_before[0]
                            merged_df.at[thing, 'ScoreAfter'] = score_after[0]
                            merged_df.at[thing, 'FrequencyBefore'] = score_before[1]
                            merged_df.at[thing, 'FrequencyAfter'] = score_after[1]


                    res = merged_df.index[merged_df['VENDOR_NUM'].isin({df_name})]
                    for item in res:
                        merged_df.at[item, 'LastDocuments'] = join_entries(unique_ids)

            merged_df = merged_df.replace({np.nan: None})
            data_dict = merged_df.set_index('VENDOR_NUM').T.to_dict()
            json_obj = {key: value for key, value in data_dict.items()}
            #display(merged_df)
            save_file = open('data/'+str(tenant)+'/autotrain_'+str(tenant)+'_.json', "w")  
            json.dump(json_obj, save_file, indent = 6)  
            save_file.close()
        else:
            print("Keine Autotrainer Daten")  



def get_autotrain_results():
    if(os.path.isfile('data/'+str(tenant)+'/autotrain_'+str(tenant)+'_.json') == True):
        with open('data/'+str(tenant)+'/autotrain_'+str(tenant)+'_.json') as user_file:
            file_contents = user_file.read()
        return file_contents
    else:
        return False


def get_results_from_json():
    if(os.path.isfile('data/'+str(tenant)+'/scorecard_df_'+str(tenant)+'_.json') == True):
        with open('data/'+str(tenant)+'/scorecard_df_'+str(tenant)+'_.json') as user_file:
            file_contents = user_file.read()
        return file_contents
    else:
        return False




def get_results_from_json_debitor():
    if(os.path.isfile('data/'+str(tenant)+'/results_debitor_'+str(tenant)+'_.json') == True):
        with open('data/'+str(tenant)+'/results_debitor_'+str(tenant)+'_.json') as user_file:
            file_contents = user_file.read()

        return file_contents
    else:
        return False

def get_results_from_json_complete():
    with open('data/'+str(tenant)+'/results_complete_'+str(tenant)+'_.json') as user_file:
        file_contents = user_file.read()

    return file_contents

def get_results_from_json_complete_frequency():
    with open('data/'+str(tenant)+'/results_complete_frequency_'+str(tenant)+'_.json') as user_file:
        file_contents = user_file.read()

    return file_contents

def get_sollwerte():
    sollwerte = pd.read_csv('data/sollwerte.csv', encoding='utf-8')
    sollwerte_transposed = sollwerte.set_index('Unnamed: 0').T
    sollwerte_json = sollwerte_transposed.iloc[0].to_json()
    print(sollwerte_json)
    return sollwerte_json

def get_commments_from_sollwerte():
    def generate_json_object():
        json_object = {}

        with open('data/sollwerte.csv', 'r', encoding='utf-8') as file:
            csv_data = csv.DictReader(file)
            
            for row in csv_data:
                key = row['']
                value = row['Comment']
                
                if key and value:
                    json_object[key] = value
                    
        return json_object
    
    json_data = generate_json_object()
    json_string = json.dumps(json_data, indent=4)

    return json_string

def update_csv(log, value1=None, value2=None, value3=None):
    csv_filename = 'data/'+str(tenant)+'/log_'+str(tenant)+'_.csv'
    logtime_exists = False
    rows = []

    # CSV-Datei vollständig einlesen
    with open(csv_filename, "r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Überprüfen, ob Logtime bereits vorhanden ist
    for row in rows:
        if row and row[0] == log:
            logtime_exists = True
            if value1 is not None:
                row[1] = str(value1)  # Wert1 aktualisieren
            if value2 is not None:
                row[2] = str(value2)  # Wert2 aktualisieren
            if value3 is not None:
                row[3] = str(value3)  # Wert3 aktualisieren
            break

    # Wenn Logtime nicht vorhanden ist, neue Zeile hinzufügen
    if not logtime_exists:
        new_row = [log]
        if value1 is not None:
            new_row.append(str(value1))
        else:
            new_row.append("")
        if value2 is not None:
            new_row.append(str(value2))
        else:
            new_row.append("")
        if value3 is not None:
            new_row.append(str(value3))
        else:
            new_row.append("")
        rows.append(new_row)

    # CSV-Datei aktualisieren
    with open(csv_filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)
    

def get_current_status():
    global critical

    print(critical)
    if(critical == 0): 
        x =  '{ "status":"Grün"}'
    elif(2>=critical >=1):
        x =  '{ "status":"Gelb"}'
    elif(critical > 2):
        x =  '{ "status":"Rot"}'
    critical = 0
    return x

def get_debitor_results():
    if (os.path.exists('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv')):
        df = pd.read_csv('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv', encoding='utf-8')
        value = 0
        len = 0
        value, len = get_single_value(df, 'DEBITOR_NUM')

        update_csv(enddate, value1=value)

        score = '{"score":"'+str(value)+'","frequency":"'+str(len)+'"}'
        if(value < 0.93):
            global critical
            critical += 1

        return score

def get_vendor_results():
    if (os.path.exists('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv')):
        df = pd.read_csv('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv', encoding='utf-8')
        value = 0
        len = 0
        value, len = get_single_value(df, 'VENDOR_NUM')
        score = '{"score":"'+str(value)+'","frequency":"'+str(len)+'"}'

        update_csv(enddate, value2=value)

        if(value < 0.90):
            global critical
            critical += 1
        return score


def get_pos_results():
    if (os.path.exists('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv')):
        df = pd.read_csv('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv', encoding='utf-8')
        df_sorted = df.sort_values(by='DocumentID')
        attribute_names = ['VatAmount1', 'NetAmount1', 'VatRate1', 'InvoiceNumber', 'InvoiceDate']
        filtered_df = df_sorted[(df_sorted['Attribute_Name'].isin(attribute_names)) & (df_sorted['Delta'] == True)]
        grouped_df = filtered_df.groupby(df_sorted['DocumentID'].ne(df_sorted['DocumentID'].shift()).cumsum())

        count_per_group = grouped_df.size()
        bad_documents = len(count_per_group)
        whole_documents = df_sorted['DocumentID'].nunique()
        value = 1-(bad_documents/whole_documents)
        update_csv(enddate, value3=value)

        good_documents = whole_documents-bad_documents
        score = '{"score":"'+str(value)+'","frequency":"'+str(whole_documents)+'"}'

        if(value < 0.80):
            global critical
            critical += 1
        return score
    
def get_results_table():
    csvFilePath = 'data/'+str(tenant)+'/log_'+str(tenant)+'_.csv'
    jsonFilePath = 'data/'+str(tenant)+'/table_data'+str(tenant)+'_.json'

    data = {}
        
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
            
        for rows in csvReader:
            key = rows['Logtime']
            data[key] = rows


    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
    
    with open('data/'+str(tenant)+'/table_data'+str(tenant)+'_.json') as user_file:
        file_contents = user_file.read()

    return file_contents

def get_smart_invoice_error():
    if (os.path.exists('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv')):
        df = pd.read_csv('data/'+str(tenant)+'/cclogattributes_T_'+tenant+'_reduced.csv', encoding='utf-8')
        df_sorted_without_vendor_bad = df[~((df['Attribute_Name'] == 'VENDOR_NUM') & (df['Delta'] == True))]
        df_sorted_without_vendor_bad = df_sorted_without_vendor_bad[df_sorted_without_vendor_bad['DocumentID'].isin(df_sorted_without_vendor_bad[df_sorted_without_vendor_bad['Attribute_Name'] == 'VENDOR_NUM']['DocumentID'])]
        attribute_names = ['InvoiceDate','InvoiceNumber','GrossAmount']
        attribute_names_new = ['VatRate1','NetAmount1','VatAmount1']

        def get_score(attributes):
            filtered_df = df_sorted_without_vendor_bad[(df_sorted_without_vendor_bad['Attribute_Name'].isin(attributes)) & (df_sorted_without_vendor_bad['Delta'] == True)]
            grouped_attributes = filtered_df.groupby('DocumentID')['Attribute_Name'].agg(lambda x: ', '.join(sorted(set(x))))
            combination_counts = grouped_attributes.value_counts()
            x = filtered_df['Attribute_Name'].value_counts()
            grouped_df = filtered_df.groupby(df_sorted_without_vendor_bad['DocumentID'].ne(df_sorted_without_vendor_bad['DocumentID'].shift()).cumsum())
            count_per_group = grouped_df.size()
            bad_documents = len(count_per_group)
            whole_documents = df_sorted_without_vendor_bad['DocumentID'].nunique()
            value = (x[0]/whole_documents)
            good_documents = whole_documents-bad_documents
            print(good_documents)
            score = '{"score":"'+str(value)+'","frequency":"'+str(whole_documents)+'"}'

            if(value <= 0.03):
                global critical
                critical += 1

            print(score)
            return score, x
        
        score, x = get_score(attribute_names)
        score2, y = get_score(attribute_names_new)
                
        df = pd.concat([x, y], axis=0).to_frame()
        df['Attribute_Name'].to_json('data/'+str(tenant)+'/smartinvoice_distribution_'+str(tenant)+'_.json', orient="index")
        return score
    

def get_double_iban_error():
    if not os.path.isfile('./data/'+str(tenant)+'/ccvendors_bank'+tenant+'_.csv'):

        df_table_ccvendors_bank = get_table_data_ALL('CC_VENDOR_BANK', connect_to_db_better(connection_string= 'classconprocessingger.database.windows.net', database = 'T_'+tenant+''))

        df_table_ccvendors_bank = df_table_ccvendors_bank.replace('\n',' ', regex=True)
        df_table_ccvendors_bank = df_table_ccvendors_bank.replace('\r',' ', regex=True)
        df_table_ccvendors_bank.to_csv('./data/'+str(tenant)+'/ccvendors_bank'+tenant+'_.csv', index=False, header= True, encoding='utf-8')

    else:
        df_table_ccvendors_bank = pd.read_csv('./data/'+str(tenant)+'/ccvendors_bank'+tenant+'_.csv', encoding='utf-8')

    grouped = df_table_ccvendors_bank.groupby(['IBAN', 'COMPANY_NUM'])

    filtered_groups = grouped.filter(lambda x: len(x['VENDOR_NUM'].unique()) > 1)

    unique_combinations = filtered_groups.groupby(['IBAN', 'COMPANY_NUM', 'VENDOR_NUM']).size().reset_index(name='Anzahl')
    laenge = len(unique_combinations.index)

    unique_combinations = unique_combinations.drop('Anzahl', axis=1)
    out = unique_combinations.to_json(orient='records')

    score = '{"frequency":"'+str(laenge)+'","data":'+out+'}'

    if (laenge > 1):
        return score
    else: 
        return False, False
  
def get_smart_invoice_error_distribution():
    with open('data/'+str(tenant)+'/smartinvoice_distribution_'+str(tenant)+'_.json') as user_file:
        file_contents = user_file.read()

    return file_contents