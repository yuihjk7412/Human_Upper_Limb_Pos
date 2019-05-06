import ctypes
from ctypes import *
import serial
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from multiprocessing import Process, Queue
import os

total_lib = cdll.LoadLibrary("./libtotal.so")
Quat = (c_float * 32)()
Initialize_Num = 5
Initialize_Threshold = 0.03
Flag_Serial_Read = False
Limb_Length = 200; #the length between the origin of the platform and the base
Upper_Limb_Origin = np.array([[0],[0],[0],[1]])
R_initial = np.array([[0,0,1],[0,1,0],[-1,0,0]])
RSs2Js = np.array([[0,0,1],[0,1,0],[-1,0,0]])
Maxium_Graph_Length = 300

def Imu_Data_Decode_Init():
    '''
    wrap the imu_data_decode_init()
    :return: void
    '''
    total_lib.imu_data_decode_init()

def Packet_Decode_w(c):
    '''

    :param c:
    :return:
    '''
    total_lib.Packet_Decode(c_char(c))

def Get_Quat(pointer):
    '''

    :param pointer:
    :return:
    '''
    total_lib.get_quat(byref(pointer))

def Print_Quat_Info():
    '''

    :return:
    '''
    print("POINT1:quat(W X Y Z):%0.3f %0.3f %0.3f %0.3f\r\n" % (Quat[0], Quat[1], Quat[2], Quat[3]))
    print("POINT2:quat(W X Y Z):%0.3f %0.3f %0.3f %0.3f\r\n" % (Quat[4], Quat[5], Quat[6], Quat[7]))

def Set_Initial_Pos():
    temp = np.zeros([1,32])
    print("Initialization Processing")
    i = 0
    while i < Initialize_Num:
        Get_Quat(Quat)
        for j in range(32):
            temp[i,j] = Quat[j]
        print("The %d time initialization:"%(i+1))
        #print(temp)
        #Print_Quat_Info()
        Max_Dif = np.max(temp,axis=0,keepdims=True) - np.min(temp,axis=0,keepdims=True)
        assert (Max_Dif.shape == (1,32))
        if ((Max_Dif > Initialize_Threshold).any()):
            i = 0
            print(Max_Dif > Initialize_Threshold)
            print("Initialization Failure!Please stay still!")
            temp = np.zeros([1, 32])
            continue
        time.sleep(0.5)
        temp = np.row_stack((temp, np.zeros([1,32])))
        i = i + 1

    temp = np.delete(temp,-1,0)
    assert (temp.shape == (Initialize_Num,32))
    average = np.average(temp, axis=0)
    average = np.reshape(average,(1,32))
    assert(average.shape == (1,32))
    print("Initialization Completed!")
    print('The 1 IMU initial Quat is:',average[0][0:4])
    print('The 2 IMU initial Quat is:',average[0][4:8])
    return average

def Serial_Read():
    while (Flag_Serial_Read):
        with serial.Serial("/dev/ttyUSB%d"%int(Port_Num), 115200) as ser:
            time.sleep(0.01)
            num = ser.in_waiting
            buf = ser.read(num)

        if num:
            for i in range(num):
                Packet_Decode_w(buf[i])


'''

def Serial_Read():
    while (Flag_Serial_Read):
        with serial.Serial("/dev/ttyUSB%d"%int(Port_Num), 115200, timeout=0.2) as ser:
            buf = ser.read(1024)

        for i in range(1024):
            Packet_Decode_w(buf[i])

        time.sleep(0.02)
'''
def Quat2R(q0,q1,q2,q3):
    '''

    :param q0:
    :param q1:
    :param q2:
    :param q3:
    :return:
    '''
    R = np.zeros([3,3])
    R[0,0] = 1 - 2 * q2**2 - 2 * q3**2
    R[0,1] = 2 * q1 * q2 - 2 * q0 * q3
    R[0,2] = 2 * q1 * q3 + 2 * q0 * q2
    R[1,0] = 2 * q1 * q2 + 2 * q0 * q3
    R[1,1] = 1 - 2 * q1**2 - 2 * q3**2
    R[1,2] = 2 * q2 * q3 - 2 * q0 * q1
    R[2,0] = 2 * q1 * q3 - 2 * q0 * q2
    R[2,1] = 2 * q2 * q3 + 2 * q0 * q1
    R[2,2] = 1 - 2 * q1**2 - 2 * q2**2
    assert (R.shape == (3,3))
    return R

def Cur_Quat2Relative_R(Relative_Zero_Points, Current_Points,Total_Arr_Length = 32):
    RSs2Js = np.eye(3)

    assert (Current_Points.shape == (1,Total_Arr_Length))
    assert (Relative_Zero_Points.shape == (1,Total_Arr_Length))
    RSs02e= Quat2R(Relative_Zero_Points[0,0],Relative_Zero_Points[0,1],Relative_Zero_Points[0,2],Relative_Zero_Points[0,3])
    RSu02e = Quat2R(Relative_Zero_Points[0,4],Relative_Zero_Points[0,5],Relative_Zero_Points[0,6],Relative_Zero_Points[0,7])
    RSs2e = Quat2R(Current_Points[0,0],Current_Points[0,1],Current_Points[0,2],Current_Points[0,3])
    RSu2e = Quat2R(Current_Points[0,4],Current_Points[0,5],Current_Points[0,6],Current_Points[0,7])
    assert (RSs02e.shape == (3, 3))
    assert (RSu02e.shape == (3, 3))
    assert (RSs2e.shape == (3, 3))
    assert (RSu2e.shape == (3, 3))
    RSs2Ss0 = np.dot(np.linalg.inv(RSs02e), RSs2e)
    RSu2Su0 = np.dot(np.linalg.inv(RSu02e),RSu2e)

    RJs2Js0 = np.dot(RSs2Js, RSs2Ss0)
    RJs2Js0 = np.dot(RJs2Js0, np.linalg.inv(RSs2Js))

    RSu2Ju = np.dot(RSs2Js, np.linalg.inv(RSs02e))
    RSu2Ju = np.dot(RSu2Ju, RSu02e)
    RJu2Ju0 = np.dot(RSu2Ju, RSu2Su0)
    RJu2Ju0 = np.dot(RJu2Ju0, np.linalg.inv(RSu2Ju))

    RJu2Js = np.dot(np.linalg.inv(RJs2Js0), RJu2Ju0)

    assert (RJu2Js.shape == (3,3))
    return RJu2Js

def Get_Limb_Pos(Rot_Mat):
    assert (Rot_Mat.shape == (3,3))
    Trans = np.column_stack((Rot_Mat, np.zeros([3,1])))
    Trans = np.row_stack((Trans, np.zeros((1,4))))
    Linear_Trans = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-Limb_Length],[0,0,0,1]])
    Trans = np.dot(Trans,Linear_Trans)
    u_o = np.dot(Trans, Upper_Limb_Origin)
    return [u_o]

def Stop_The_Process():
    while(1):
        temp = input("input q to terminate the process:")
        if temp == 'q':
            return False

def Get_Euler_Angle(Rot_Mat):
    assert (Rot_Mat.shape == (3, 3))
    xtheta = np.degrees(np.arcsin(-Rot_Mat[1][2]))
    ytheta = np.degrees(np.arctan(Rot_Mat[0][2] / Rot_Mat[2][2]))
    ztheta = np.degrees(np.arctan(Rot_Mat[1][0] / Rot_Mat[1][1]))
    return [xtheta, ytheta, ztheta]


def Serial_Read_Data_Analysis(q,Port_Num):
    #Imu_Data_Decode_Init()
    total_lib.imu_data_decode_init()

    with serial.Serial("/dev/ttyUSB%d" % int(Port_Num), 115200, timeout=0.2) as ser:
        print("Serial Port OK!")
    ser.close()
    try:
        ts = threading.Thread(target=Serial_Read, name='Serial_Read_Thread')
    except:
        raise EOFError

    ts.start()  # start reading the data from the Serial port
    Quat_Relative_Zero_Point = Set_Initial_Pos()  # getting the initial position

    i = 0

    while (1):
        Get_Quat(Quat)
        Rot_Mat_u2s = Cur_Quat2Relative_R(Quat_Relative_Zero_Point, (np.asarray(Quat)).reshape((1, -1)))
        upper_o = Get_Limb_Pos(Rot_Mat_u2s)
        [xtheta_temp, ytheta_temp, ztheta_temp] = Get_Euler_Angle(Rot_Mat_u2s)

        q.put([xtheta_temp,ytheta_temp,ztheta_temp,upper_o[0][0, 0],upper_o[0][1, 0],upper_o[0][2, 0]])

        i = i+1
        print("NO%d\tX:%0.3f\tY:%0.3f\tZ:%0.3f" % (i, upper_o[0][0, 0], upper_o[0][1, 0], upper_o[0][2, 0]))
        print("NO%d\tX:%0.3f\tY:%0.3f\tZ:%0.3f" % (i, xtheta_temp, ytheta_temp, ztheta_temp))

def Plot_Data(q):
    xtheta = [0]
    ytheta = [0]
    ztheta = [0]
    xpos = [0]
    ypos = [0]
    zpos = [0]
    plt.figure('Pos')
    plt.ion()

    plt.figure("Euler Angle")
    plt.ion()
    plt.ylabel('Angle/deg')
    plt.xlabel('counts/times')
    plt.title('Euler angle')
    plt.ylim(-100, 100)
    while(1):
        temp = q.get(True)
        xtheta.append(temp[0])
        ytheta.append(temp[1])
        ztheta.append(temp[2])
        xpos.append(temp[3])
        ypos.append(temp[4])
        zpos.append(temp[5])

        Points_Num = list(range(len(xtheta)))
        if len(xtheta) > Maxium_Graph_Length:
            graph_Start_Index = len(xtheta) - Maxium_Graph_Length
        else:
            graph_Start_Index = 0

        plt.figure("Euler Angle")
        plt.clf()
        plt.plot(Points_Num[graph_Start_Index:-1], xtheta[graph_Start_Index:-1],'r',label = 'xtheta')
        plt.plot(Points_Num[graph_Start_Index:-1], ytheta[graph_Start_Index:-1],'g',label = 'ytheta')
        plt.plot(Points_Num[graph_Start_Index:-1], ztheta[graph_Start_Index:-1],'b',label = 'ztheta')
        plt.grid(True)
        plt.draw()
        plt.pause(0.01)

        plt.figure('Pos')
        plt.clf()
        plt.plot(Points_Num[graph_Start_Index:-1], xpos[graph_Start_Index:-1], 'r', label='xtheta')
        plt.plot(Points_Num[graph_Start_Index:-1], ypos[graph_Start_Index:-1], 'g', label='ytheta')
        plt.plot(Points_Num[graph_Start_Index:-1], zpos[graph_Start_Index:-1], 'b', label='ztheta')
        plt.grid(True)
        plt.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    q = Queue()

    Port_Num = input('PLEASE INPUT THE PORT NUMBER(/dev/ttyUSB*):')
    p_Serial_Read = Process(target=Serial_Read_Data_Analysis,args=(q,Port_Num,))
    #p_Plot = Process(target=Plot_Data,args=(q,))

    p_Serial_Read.start()
    #p_Plot.start()
    while 1:
        time.sleep(60)
        '''a = input()
        if a == 'q':
            break'''
    p_Serial_Read.terminate()
    #p_Plot.terminate()
