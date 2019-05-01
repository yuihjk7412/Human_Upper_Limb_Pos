import ctypes
from ctypes import *
import serial
import time
import numpy as np
import threading

Initialize_Num = 5
Initialize_Threshold = 0.003
Flag_Serial_Read = False
Limb_Length = 200; #the length between the origin of the platform and the base
Upper_Limb_Origin = np.array([[0],[0],[0],[1]])
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
    average = np.average(temp, axis=0)
    average = np.reshape(average,(1,32))
    assert(average.shape == (1,32))
    print("Initialization Completed!")
    print('The 1 IMU initial Quat is:',average[0][0:4])
    print('The 2 IMU initial Quat is:',average[0][4:8])
    return average

def Serial_Read():
    while (Flag_Serial_Read):
        with serial.Serial("/dev/ttyUSB%d"%int(Port_Num), 115200, timeout=0.2) as ser:
            buf = ser.read(1024)

        for i in range(1024):
            Packet_Decode_w(buf[i])

        time.sleep(0.02)

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
    assert (Current_Points.shape == (1,Total_Arr_Length))
    assert (Relative_Zero_Points.shape == (1,Total_Arr_Length))
    Rs02o = Quat2R(Relative_Zero_Points[0,0],Relative_Zero_Points[0,1],Relative_Zero_Points[0,2],Relative_Zero_Points[0,3])
    Ru02o = Quat2R(Relative_Zero_Points[0,4],Relative_Zero_Points[0,5],Relative_Zero_Points[0,6],Relative_Zero_Points[0,7])
    Rs2o = Quat2R(Current_Points[0,0],Current_Points[0,1],Current_Points[0,2],Current_Points[0,3])
    Ru2o = Quat2R(Current_Points[0,4],Current_Points[0,5],Current_Points[0,6],Current_Points[0,7])
    assert (Rs02o.shape == (3, 3))
    assert (Ru02o.shape == (3, 3))
    assert (Rs2o.shape == (3, 3))
    assert (Ru2o.shape == (3, 3))
    Ru2u0 = np.dot(np.linalg.inv(Ru02o),Ru2o)
    Rs2s0 = np.dot(np.linalg.inv(Rs02o),Rs2o)
    Ru2s = np.dot(np.linalg.inv(Rs2s0),Ru2u0)
    assert (Ru2s.shape == (3,3))
    return Ru2s

def Get_Limb_Pos(Rot_Mat):
    assert (Rot_Mat.shape == (3,3))
    Trans = np.column_stack((Rot_Mat, np.zeros([3,1])))
    Trans = np.row_stack((Trans, np.zeros((1,4))))
    Linear_Trans = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-Limb_Length],[0,0,0,1]])
    Trans = np.dot(Trans,Linear_Trans)
    u_o = np.dot(Trans, Upper_Limb_Origin)
    return [u_o]

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
    ts.start() #start reading the data from the Serai port
    Quat_Relative_Zero_Point = Set_Initial_Pos() #getting the initial position

    while(1):
        Get_Quat(Quat)
        Rot_Mat_u2s = Cur_Quat2Relative_R(Quat_Relative_Zero_Point, (np.asarray(Quat)).reshape((1,-1)))
        upper_o = Get_Limb_Pos(Rot_Mat_u2s)
        print("The position of the upper limb:X:%0.3f\tY:%0.3f\tZ:%0.3f"%(upper_o[0][0,0],upper_o[0][1,0],upper_o[0][2,0]))

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




