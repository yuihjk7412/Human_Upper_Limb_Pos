import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt

Maxium_Graph_Length = 300

def ploting_data():
    plt.figure('Pos')
    plt.ion()

    plt.figure("Euler Angle")
    plt.ion()
    plt.ylabel('Angle/deg')
    plt.xlabel('counts/times')
    plt.title('Euler angle')
    plt.ylim(-100, 100)

    file_name_list = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1] == '.csv']
    file_name = file_name_list[-1]  # read the most updated csv file
    data_frame = pd.read_csv('%s' % file_name, header=None, index_col=False)
    data_array = np.array(data_frame)

    Data_Length = data_array.shape[0]
    Num_List = list(range(Data_Length))
    if Data_Length > Maxium_Graph_Length:
        graph_Start_Index = Data_Length - Maxium_Graph_Length
    else:
        graph_Start_Index = 0

    plt.figure("Euler Angle")
    plt.clf()
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1, 0], 'r', label='xtheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1, 1], 'g', label='ytheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1, 2], 'b', label='ztheta')
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)

    plt.figure('Pos')
    plt.clf()
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1, 3], 'r', label='xtheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1, 4], 'g', label='ytheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1, 5], 'b', label='ztheta')
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)

if __name__ == '__main__':

    plt.figure('Pos')
    plt.ion()

    plt.figure("Euler Angle")
    plt.ion()
    plt.ylabel('Angle/deg')
    plt.xlabel('counts/times')
    plt.title('Euler angle')
    plt.ylim(-100, 100)

    file_name_list = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.csv']
    file_name = file_name_list[-1]  #read the most updated csv file
    data_frame = pd.read_csv('%s'%file_name,header=None,index_col=False)
    data_array = np.array(data_frame)

    Data_Length = data_array.shape[0]
    Num_List = list(range(Data_Length))
    if Data_Length > Maxium_Graph_Length:
        graph_Start_Index = Data_Length - Maxium_Graph_Length
    else:
        graph_Start_Index = 0

    plt.figure("Euler Angle")
    plt.clf()
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1,0], 'r', label='xtheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1,1], 'g', label='ytheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1,2], 'b', label='ztheta')
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)

    plt.figure('Pos')
    plt.clf()
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1,3], 'r', label='xtheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1,4], 'g', label='ytheta')
    plt.plot(Num_List[graph_Start_Index:-1], data_array[graph_Start_Index:-1,5], 'b', label='ztheta')
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)