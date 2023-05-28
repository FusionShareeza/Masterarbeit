import pandas as pd
import numpy as np
cimport numpy as np

# Definition der Cython-Optimierungen
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

def fun():
    # Laden des DataFrames
    df = pd.read_csv('data/cclogattributes_T000ht3_reduced.csv', encoding='utf-8')
    
    # Sortieren des DataFrames nach 'DocumentID'
    df_sorted = df.sort_values(by='DocumentID')
    
    # Berechnung der Änderungen in 'DocumentID'
    cdef np.ndarray[bool] document_changes = df_sorted['DocumentID'].values != np.roll(df_sorted['DocumentID'].values, 1)
    
    # Variableninitialisierung
    cdef long previous_document = -1
    cdef int bad_documents = 0
    cdef int whole_documents = 0
    
    # Iteration über den DataFrame
    for index in range(df_sorted.shape[0]):
        count = 0
        
        if document_changes[index]:
            whole_documents += 1
            previous_document = df_sorted.iloc[index]['DocumentID']
        
        row = df_sorted.iloc[index]
        attribute_name = row['Attribute_Name']
        delta = row['Delta']
        
        if attribute_name == 'VatAmount1' and delta:
            count += 1
        if attribute_name == 'NetAmount1' and delta:
            count += 1
        if attribute_name == 'VatRate1' and delta:
            count += 1
        if attribute_name == 'InvoiceNumber' and delta:
            count += 1
        if attribute_name == 'InvoiceDate' and delta:
            count += 1
        
        if count >= 1:
            bad_documents += 1
    
    return bad_documents, whole_documents
