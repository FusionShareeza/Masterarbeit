import pandas as pd

def get_data_values4(df, items): 
    df_ordernum_correct = df[(df['Attribute_Name'] == items[0]) & (df['Delta'] == items[1])]
    documentid_list = df_ordernum_correct['DocumentID'].tolist()
    denominator_ordernum = len(documentid_list)
    df_debitor_ordernum = df[(df['Attribute_Name'] == items[0]) | (df['Attribute_Name'] == items[2])]

    counter_do = 0
    for entry_1 in documentid_list:
        speicher_entry_new = df_debitor_ordernum[df_debitor_ordernum['DocumentID'] == entry_1]
        if ((speicher_entry_new['Attribute_Name'] == items[0]) & (speicher_entry_new['Delta'] == items[1])).any():
            if ((speicher_entry_new['Attribute_Name'] == items[2]) & (speicher_entry_new['Delta'] == False)).any():
                counter_do += 1

    try:
        return counter_do / denominator_ordernum, denominator_ordernum
    except ZeroDivisionError:
        return pd.NA, pd.NA

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

df_table_cclogattributes = pd.read_csv('data/cclogattributes_T000git_reduced.csv', encoding='utf-8')

for items in list1:
    value, value_frequency = get_data_values4(df_table_cclogattributes, items)
    print(value)