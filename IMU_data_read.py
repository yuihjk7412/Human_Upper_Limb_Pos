import ctypes
from ctypes import *
import serial
import time
import numpy as np
import threading

Initialize_Num = 5
Initialize_Threshold = 0.003
Flag_Serial_Read = False

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
            print(Max_Dif > 0.5)
            print("Initialization Failure!Please stay still!")
            temp = np.zeros([1, 32])
            continue
        time.sleep(0.5)
        temp = np.row_stack((temp, np.zeros([1,32])))
        i = i + 1

    temp = np.delete(temp,-1,0)
    assert (temp.shape == (Initialize_Num,32))
    print("Initialization Completed!")
    return np.average(temp, axis=0)

def Serial_Read():
    while (Flag_Serial_Read):
        with serial.Serial("/dev/ttyUSB%d"%int(Port_Num), 115200, timeout=0.2) as ser:
            buf = ser.read(1024)

        for i in range(1024):
            Packet_Decode_w(buf[i])

        time.sleep(0.02)

if __name__ == '__main__':
    total_lib = cdll.LoadLibrary("./libtotal.so")
    Imu_Data_Decode_Init()
    Quat = (c_float * 32)()
    Port_Num = input('PLEASE INPUT THE PORT NUMBER(/dev/ttyUSB*):')
    with serial.Serial("/dev/ttyUSB%d" % int(Port_Num), 115200, timeout=0.2) as ser:
        print("Serial Port OK!")
    ser.close()
    Flag_Serial_Read = True
    try:
        ts = threading.Thread(target=Serial_Read, name='Serial_Read_Thread')
    except:
        raise EOFError
    ts.start()
    Quat_Relative_Zero_Point = Set_Initial_Pos()
    print(Quat_Relative_Zero_Point[0:8])
    Get_Quat(Quat)
    Flag_Serial_Read = False


'''
if __name__ == '__main__':
    #packet_lib = cdll.LoadLibrary("./libpacket.so")
    #imu_data_decode_lib = cdll.LoadLibrary("./libimu_data_decode.so")
    #imu_data_decode_lib.imu_data_decode_init()
    total_lib = cdll.LoadLibrary("./libtotal.so")
    #total_lib.imu_data_decode_init()
    Imu_Data_Decode_Init()
    Quat = (c_float * 32)()
    while(1):

        with serial.Serial("/dev/ttyUSB0",115200,timeout=0.2) as ser:
            buf = ser.read(1024)


        for i in range(1024):
            #packet_lib.Packet_Decode(c_char(buf[i]))
            #total_lib.Packet_Decode(c_char(buf[i]))
            Packet_Decode_w(buf[i])
        ini_quat = Set_Initial_Pos()
        #imu_data_decode_lib.get_quat(byref(Quat))
        #total_lib.get_quat(byref(Quat))
        Get_Quat(Quat)
        print("POINT1:quat(W X Y Z):%0.3f %0.3f %0.3f %0.3f\r\n"%(Quat[0], Quat[1], Quat[2], Quat[3]))
        print("POINT2:quat(W X Y Z):%0.3f %0.3f %0.3f %0.3f\r\n"%(Quat[4], Quat[5], Quat[6], Quat[7]))
        time.sleep(0.02)'''




