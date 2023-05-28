import pandas as pd
def testi():
    df = pd.read_csv('data/cclogattributes_T000ht3_reduced.csv', encoding='utf-8')
    df_sorted = df.sort_values(by='DocumentID')
    attribute_names = ['VatAmount1', 'NetAmount1', 'VatRate1', 'InvoiceNumber', 'InvoiceDate']
    filtered_df = df_sorted[(df_sorted['Attribute_Name'].isin(attribute_names)) & (df_sorted['Delta'] == True)]
    grouped_df = filtered_df.groupby(df_sorted['DocumentID'].ne(df_sorted['DocumentID'].shift()).cumsum())

    count_per_group = grouped_df.size()
    bad_documents = len(count_per_group)
    whole_documents = df_sorted['DocumentID'].nunique()

    print(bad_documents, whole_documents)