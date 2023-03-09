import networkx as nx
import matplotlib.pyplot as plt
import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os
import pandas as pd 
file_path = os.path.abspath(os.path.join(__file__, "../.."))
BASE_DIR = os.path.dirname(file_path)

demands_sheet=pd.read_excel(BASE_DIR + '/Stats/Model_Stats_LN1_M2_1.xlsx', sheet_name='Demands',header=0)
nodes=[]
nodes.extend(i for i in demands_sheet['Destination'].unique() )
nodes.extend(i for  i in demands_sheet['Source'].unique())
links_sheet=pd.read_excel(BASE_DIR + '/Stats/Model_Stats_LN1_M2_1.xlsx', sheet_name='Links',header=0)


print(links_sheet)