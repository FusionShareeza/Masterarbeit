import pandas as pd
import numpy as np

def fun():
    df = pd.read_csv('data/cclogattributes_T000ht3_reduced.csv', encoding='utf-8')
    df_sorted = df.sort_values(by='DocumentID')
    document_changes = df_sorted['DocumentID'].ne(df_sorted['DocumentID'].shift())
    previous_document = None
    bad_documents = 0
    whole_documents = 0
    for index, row in df_sorted.iterrows():
        count = 0
        if document_changes.iloc[index]:
            whole_documents += 1
            previous_document = row['DocumentID']
        if row['Attribute_Name'] == 'VatAmount1' and row['Delta'] == True:
            count += 1
        if row['Attribute_Name'] == 'NetAmount1' and row['Delta'] == True:
            count += 1
        if row['Attribute_Name'] == 'VatRate1' and row['Delta'] == True:
            count += 1
        if row['Attribute_Name'] == 'InvoiceNumber' and row['Delta'] == True:
            count += 1
        if row['Attribute_Name'] == 'InvoiceDate' and row['Delta'] == True:
            count += 1
        if(count>=1):
            bad_documents += 1

    print(bad_documents,whole_documents)