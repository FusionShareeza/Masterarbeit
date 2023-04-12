import numpy as np
import pyodbc
import pandas as pd



#connection with default parameter --> can be overwritten with setting parameters in function call
def connect_to_db(driver = 'SQL Server Native Client 11.0',
                  connection_string = 'testclassconprocessingger.database.windows.net',
                    database = 'ConsolidationDB',
                    user = 'CCAdmin',
                    password = 'Miw6RjnTGmPHLYF9mG1o'
):
    connection = pyodbc.connect("Driver={"+driver+"};"
                        "Server="+connection_string+";"
                        "Database="+database+";"
                        "uid="+user+";pwd="+password+"")
    return connection


def get_table_data(table_name, connection):
    query = "SELECT * FROM {}".format(table_name)
    df = pd.read_sql_query(query, connection)

    return df

#get all tables
cursor = connect_to_db().cursor()
tables_list = list(cursor.tables())

df_table_cclogattributes = get_table_data('CCLogAttributes', connect_to_db(database='T_0000xu'))
df_table_cclogdocuments = get_table_data('CCLogDocuments', connect_to_db(database='T_0000xu'))

print(df_table_cclogattributes)
print(df_table_cclogdocuments)


#cursor.execute(query)
