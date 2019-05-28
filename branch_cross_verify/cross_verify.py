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
import xsensdeviceapi as xda
from threading import Lock

Initialize_Num = 5
Initialize_Threshold = 0.02
Flag_Serial_Read = False
Limb_Length = 200;  # the length between the origin of the platform and the base
Upper_Limb_Origin = np.array([[0], [0], [0], [1]])
R_initial = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
RSs2Js = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
Maxium_Graph_Length = 300


class XdaCallback(xda.XsCallback):
    """
    Define a class to read data from the xsens sensor
    """
    def __init__(self, max_buffer_size = 5):
        xda.XsCallback.__init__(self)
        self.m_maxNumberOfPacketsInBuffer = max_buffer_size
        self.m_packetBuffer = list()
        self.m_lock = Lock()

    def packetAvailable(self):
        self.m_lock.acquire()
        res = len(self.m_packetBuffer) > 0
        self.m_lock.release()
        return res

    def getNextPacket(self):
        self.m_lock.acquire()
        assert(len(self.m_packetBuffer) > 0)
        oldest_packet = xda.XsDataPacket(self.m_packetBuffer.pop(0))
        self.m_lock.release()
        return oldest_packet

    def onLiveDataAvailable(self, dev, packet):
        self.m_lock.acquire()
        assert(packet is not 0)
        while len(self.m_packetBuffer) >= self.m_maxNumberOfPacketsInBuffer:
            self.m_packetBuffer.pop()
        self.m_packetBuffer.append(xda.XsDataPacket(packet))
        self.m_lock.release()


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
    temp = np.zeros([1, 32])
    print("Initialization Processing")
    i = 0
    while i < Initialize_Num:
        Get_Quat(Quat)
        for j in range(32):
            temp[i, j] = Quat[j]
        print("The %d time initialization:" % (i + 1))
        # print(temp)
        # Print_Quat_Info()
        Max_Dif = np.max(temp, axis=0, keepdims=True) - np.min(temp, axis=0, keepdims=True)
        assert (Max_Dif.shape == (1, 32))
        if ((Max_Dif > Initialize_Threshold).any()):
            i = 0
            print(Max_Dif > Initialize_Threshold)
            print("Initialization Failure!Please stay still!")
            temp = np.zeros([1, 32])
            continue
        time.sleep(0.5)
        temp = np.row_stack((temp, np.zeros([1, 32])))
        i = i + 1

    temp = np.delete(temp, -1, 0)
    assert (temp.shape == (Initialize_Num, 32))
    average = np.average(temp, axis=0)
    average = np.reshape(average, (1, 32))
    assert (average.shape == (1, 32))
    print("Initialization Completed!")
    print('The 1 IMU initial Quat is:', average[0][0:4])
    print('The 2 IMU initial Quat is:', average[0][4:8])
    return average

def Set_Initial_Pos(callback):
    temp = np.zeros([1, 32])
    temp_ = np.zeros([1,4])
    print("Initialization Processing")
    i = 0
    while i < Initialize_Num:
        Get_Quat(Quat)
        xsens_Data = Read_Xsens_Quaternion(callback)
        for j in range(32):
            temp[i, j] = Quat[j]
        for k in range(4):
            temp_[i, k] = xsens_Data[k]
        print("The %d time initialization:" % (i + 1))
        Max_Dif = np.max(temp, axis=0, keepdims=True) - np.min(temp, axis=0, keepdims=True)
        Max_Dif_ = np.max(temp_, axis=0, keepdims=True) - np.min(temp_, axis=0, keepdims=True)
        if ((Max_Dif > Initialize_Threshold).any() or (Max_Dif_ > Initialize_Threshold).any()):
            i = 0
            print("Initialization Failure!Please stay still!")
            temp = np.zeros([1, 32])
            temp_ = np.zeros([1, 4])
            continue
        time.sleep(0.5)
        temp = np.row_stack((temp, np.zeros([1, 32])))
        temp_ = np.row_stack((temp_, np.zeros([1, 4])))
        i = i + 1

    temp = np.delete(temp, -1, 0)
    temp_ = np.delete(temp_, -1, 0)
    average = np.average(temp, axis=0)
    average = np.reshape(average, (1, 32))
    average_ = np.average(temp_, axis=0)
    average_ = np.reshape(average_, (1, 4))
    average_ = np.array([average[0][0:4], average_[0][0:4]])
    average_ = np.reshape(average_, (1,8))
    print("Initialization Completed!")
    print('The 1 IMU initial Quat is:', average[0][0:4])
    print('The 2 IMU initial Quat is:', average[0][4:8])
    print('The xsens imu initial Quat is:', average_[0][4:8])
    return [average,average_]


def Serial_Read():
    while (Flag_Serial_Read):
        with serial.Serial("/dev/ttyUSB%d" % int(Port_Num), 115200) as ser:
            time.sleep(0.01)
            num = ser.in_waiting
            buf = ser.read(num)

        if num:
            for i in range(num):
                Packet_Decode_w(buf[i])

def Quat2R(q0, q1, q2, q3):
    '''

    :param q0:
    :param q1:
    :param q2:
    :param q3:
    :return: the rotational matrix calculated from the Quatnion
    '''
    R = np.zeros([3, 3])
    R[0, 0] = 1 - 2 * q2 ** 2 - 2 * q3 ** 2
    R[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    R[0, 2] = 2 * q1 * q3 + 2 * q0 * q2
    R[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    R[1, 1] = 1 - 2 * q1 ** 2 - 2 * q3 ** 2
    R[1, 2] = 2 * q2 * q3 - 2 * q0 * q1
    R[2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    R[2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    R[2, 2] = 1 - 2 * q1 ** 2 - 2 * q2 ** 2
    assert (R.shape == (3, 3))
    return R


def Cur_Quat2Relative_R(Relative_Zero_Points, Current_Points, REu2Es, RSs2Js, Total_Arr_Length=32):
    RSs02e = Quat2R(Relative_Zero_Points[0, 0], Relative_Zero_Points[0, 1], Relative_Zero_Points[0, 2],
                    Relative_Zero_Points[0, 3])
    RSu02e = Quat2R(Relative_Zero_Points[0, 4], Relative_Zero_Points[0, 5], Relative_Zero_Points[0, 6],
                    Relative_Zero_Points[0, 7])
    RSs2e = Quat2R(Current_Points[0, 0], Current_Points[0, 1], Current_Points[0, 2], Current_Points[0, 3])
    RSu2e = Quat2R(Current_Points[0, 4], Current_Points[0, 5], Current_Points[0, 6], Current_Points[0, 7])
    RSs2Ss0 = np.dot(np.linalg.inv(RSs02e), RSs2e)
    RSu2Su0 = np.dot(np.linalg.inv(RSu02e), RSu2e)

    RJs2Js0 = np.dot(RSs2Js, RSs2Ss0)
    RJs2Js0 = np.dot(RJs2Js0, np.linalg.inv(RSs2Js))

    RSu2Ju = np.dot(RSs2Js, np.linalg.inv(RSs02e))
    RSu2Ju = np.dot(RSu2Ju, REu2Es)
    RSu2Ju = np.dot(RSu2Ju, RSu02e)
    RJu2Ju0 = np.dot(RSu2Ju, RSu2Su0)
    RJu2Ju0 = np.dot(RJu2Ju0, np.linalg.inv(RSu2Ju))

    RJu2Js = np.dot(np.linalg.inv(RJs2Js0), RJu2Ju0)

    return RJu2Js


def Get_Limb_Pos(Rot_Mat):
    assert (Rot_Mat.shape == (3, 3))
    Trans = np.column_stack((Rot_Mat, np.zeros([3, 1])))
    Trans = np.row_stack((Trans, np.zeros((1, 4))))
    # 初始位姿为双臂垂于体侧
    # Linear_Trans = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-Limb_Length],[0,0,0,1]])
    # 初始位姿为Tpose
    Linear_Trans = np.array([[1, 0, 0, 0], [0, 1, 0, -Limb_Length], [0, 0, 1, 0], [0, 0, 0, 1]])
    Trans = np.dot(Trans, Linear_Trans)
    u_o = np.dot(Trans, Upper_Limb_Origin)
    return [u_o]


def Stop_The_Process():
    while (1):
        temp = input("input q to terminate the process:")
        if temp == 'q':
            return False


def Get_Euler_Angle(Rot_Mat):
    assert (Rot_Mat.shape == (3, 3))
    #前提条件是Tpose，次序为132
    '''xtheta = np.degrees(np.arcsin(-Rot_Mat[1][2]))
    ytheta = np.degrees(np.arctan2(Rot_Mat[0][2], Rot_Mat[2][2]))
    ztheta = np.degrees(np.arctan2(Rot_Mat[1][0], Rot_Mat[1][1]))'''
    xtheta = np.degrees(np.arctan2(Rot_Mat[2][1], Rot_Mat[1][1]))
    ytheta = np.degrees(np.arctan2(Rot_Mat[0][2], Rot_Mat[0][0]))
    ztheta = np.degrees(np.arcsin(-Rot_Mat[0][1]))
    return [xtheta, ytheta, ztheta]


def cal_RSsJs(q0, q1, q2, q3, ztheta_initial):
    # 放置与躯干传感器默认Y轴与关节Y轴平行，放置于胸前，Z轴指向身体外
    print('放置于躯干传感器默认Y轴与关节Y轴平行，放置于胸前，Z轴指向身体外')
    RSs2E = Quat2R(q0, q1, q2, q3)
    RE2Js = np.array(
        [[np.cos(ztheta_initial), np.sin(ztheta_initial), 0], [-np.sin(ztheta_initial), np.cos(ztheta_initial), 0],
         [0, 0, 1]])
    RSs2Js = np.dot(RE2Js, RSs2E)
    return RSs2Js


def Norm_Cordinate():
    print("请将两个IMU放置为同一姿态")
    time.sleep(2)
    temp = np.zeros([1, 32])
    i = 0
    while i < Initialize_Num:
        Get_Quat(Quat)
        for j in range(32):
            temp[i, j] = Quat[j]
        print("The %d time initialization:" % (i + 1))
        # print(temp)
        # Print_Quat_Info()
        Max_Dif = np.max(temp, axis=0, keepdims=True) - np.min(temp, axis=0, keepdims=True)
        assert (Max_Dif.shape == (1, 32))
        if ((Max_Dif > Initialize_Threshold).any()):
            i = 0
            print(Max_Dif > Initialize_Threshold)
            print("Initialization Failure!Please stay still!")
            temp = np.zeros([1, 32])
            continue
        time.sleep(0.5)
        temp = np.row_stack((temp, np.zeros([1, 32])))
        i = i + 1

    temp = np.delete(temp, -1, 0)
    average = np.average(temp, axis=0)
    average = np.reshape(average, (1, 32))
    RSs2Es = Quat2R(average[0][0], average[0][1], average[0][2], average[0][3])
    RSu2Eu = Quat2R(average[0][4], average[0][5], average[0][6], average[0][7])
    REu2Es = np.dot(RSs2Es, np.linalg.inv(RSu2Eu))
    return REu2Es

def Norm_Cordinate(callback):
    print("请将3个IMU放置为同一姿态")
    time.sleep(2)
    temp = np.zeros([1, 32])
    temp_ = np.zeros([1,4])
    i = 0
    while i < Initialize_Num:
        Get_Quat(Quat)
        xsens_Data = Read_Xsens_Quaternion(callback)
        for j in range(32):
            temp[i, j] = Quat[j]
        for k in range(4):
            temp_[i, k] = xsens_Data[k]
        print("The %d time initialization:" % (i + 1))
        Max_Dif = np.max(temp, axis=0, keepdims=True) - np.min(temp, axis=0, keepdims=True)
        Max_Dif_ = np.max(temp_, axis=0, keepdims=True) - np.min(temp_, axis=0, keepdims=True)
        if ((Max_Dif > Initialize_Threshold).any() or (Max_Dif_ > Initialize_Threshold).any()):
            i = 0
            print("Initialization Failure!Please stay still!")
            temp = np.zeros([1, 32])
            temp_ = np.zeros([1,4])
            continue
        time.sleep(0.5)
        temp = np.row_stack((temp, np.zeros([1, 32])))
        temp_ = np.row_stack((temp_, np.zeros([1,4])))
        i = i + 1

    temp = np.delete(temp, -1, 0)
    temp_ = np.delete(temp_, -1, 0)
    average = np.average(temp, axis=0)
    average_ = np.average(temp_, axis=0)
    average = np.reshape(average, (1, 32))
    average_ = np.reshape(average_, (1,4))
    RSs2Es = Quat2R(average[0][0], average[0][1], average[0][2], average[0][3])
    RSu2Eu = Quat2R(average[0][4], average[0][5], average[0][6], average[0][7])
    RSu_2Eu_ = Quat2R(average_[0][0], average_[0][1], average_[0][2], average_[0][3])
    REu2Es = np.dot(RSs2Es, np.linalg.inv(RSu2Eu))
    REu_2Es = np.dot(RSs2Es, np.linalg.inv(RSu_2Eu_))
    print('REu2Es:',REu2Es)
    print('REu_2Es',REu_2Es)
    return [REu2Es, REu_2Es]

def Read_Xsens_Quaternion(callback):
    '''read the next packet for sure'''
    while not(callback.packetAvailable()):
        pass
    if callback.packetAvailable():
        # Retrieve a packet
        packet = callback.getNextPacket()
        if packet.containsOrientation():
            quaternion = packet.orientationQuaternion()
            s = [quaternion[0],quaternion[1],quaternion[2],quaternion[3]]
    return s

def Record_Initial_Yaw():
    print('请0#传感器放置于胸口中央')
    time.sleep(2)
    temp = np.zeros([1, 32])
    i = 0
    while i < Initialize_Num:
        Get_Quat(Quat)
        for j in range(32):
            temp[i, j] = Quat[j]
        print("The %d time initialization:" % (i + 1))
        Max_Dif = np.max(temp, axis=0, keepdims=True) - np.min(temp, axis=0, keepdims=True)
        if ((Max_Dif > Initialize_Threshold).any()):
            i = 0
            print("Initialization Failure!Please stay still!")
            temp = np.zeros([1, 32])
            continue
        time.sleep(0.5)
        temp = np.row_stack((temp, np.zeros([1, 32])))
        i = i + 1

    temp = np.delete(temp, -1, 0)
    average = np.average(temp, axis=0)
    average = np.reshape(average, (1, 32))
    RS2Es = Quat2R(average[0][0], average[0][1], average[0][2], average[0][3])
    ztheta_initial = np.arctan2(RS2Es[1][0], RS2Es[0][0])
    print('ztheta_initial:', ztheta_initial)
    return ztheta_initial


if __name__ == '__main__':

    print('new branch for a new way of initialization:')

    while 1:
        temp = input("start the process?(Y/N):")
        if temp == 'Y' or temp == 'y':
            break

    #load the external library
    total_lib = cdll.LoadLibrary("./libtotal.so")
    Imu_Data_Decode_Init()
    Quat = (c_float * 32)()

    #test the port of the wireless imu
    Port_Num = input('PLEASE INPUT THE PORT NUMBER OF THE WIRELESS IMU MODULE(/dev/ttyUSB*):')
    with serial.Serial("/dev/ttyUSB%d" % int(Port_Num), 115200, timeout=0.2) as ser:
        print("Serial Port OK!")
    ser.close()
    Flag_Serial_Read = True

    print("Creating XsControl object...")
    control = xda.XsControl_construct()
    assert (control is not 0)

    xdaVersion = xda.XsVersion()
    xda.xdaVersion(xdaVersion)
    print("Using XDA version %s" % xdaVersion.toXsString())

    try:
        print("Scanning for devices...")
        portInfoArray = xda.XsScanner_scanPorts()

        # Find an MTi device
        mtPort = xda.XsPortInfo()
        for i in range(portInfoArray.size()):
            if portInfoArray[i].deviceId().isMti() or portInfoArray[i].deviceId().isMtig():
                mtPort = portInfoArray[i]
                break

        if mtPort.empty():
            raise RuntimeError("No MTi device found. Aborting.")

        did = mtPort.deviceId()
        print("Found a device with:")
        print(" Device ID: %s" % did.toXsString())
        print(" Port name: %s" % mtPort.portName())

        print("Opening port...")
        if not control.openPort(mtPort.portName(), mtPort.baudrate()):
            raise RuntimeError("Could not open port. Aborting.")

        # Get the device object
        device = control.device(did)
        assert (device is not 0)

        print("Device: %s, with ID: %s opened." % (device.productCode(), device.deviceId().toXsString()))

        # Create and attach callback handler to device
        callback = XdaCallback()
        device.addCallbackHandler(callback)

        # Put the device into configuration mode before configuring the device
        print("Putting device into configuration mode...")
        if not device.gotoConfig():
            raise RuntimeError("Could not put device into configuration mode. Aborting.")

        print("Configuring the device...")
        configArray = xda.XsOutputConfigurationArray()
        configArray.push_back(xda.XsOutputConfiguration(xda.XDI_PacketCounter, 0))
        configArray.push_back(xda.XsOutputConfiguration(xda.XDI_SampleTimeFine, 0))

        if device.deviceId().isImu():
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_DeltaV, 0))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_DeltaQ, 0))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_MagneticField, 0))
        elif device.deviceId().isVru() or device.deviceId().isAhrs():
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_Quaternion, 0))
        elif device.deviceId().isGnss():
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_Quaternion, 0))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_LatLon, 0))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_AltitudeEllipsoid, 0))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_VelocityXYZ, 0))
        else:
            raise RuntimeError("Unknown device while configuring. Aborting.")

        if not device.setOutputConfiguration(configArray):
            raise RuntimeError("Could not configure the device. Aborting.")

        print("Putting device into measurement mode...")
        if not device.gotoMeasurement():
            raise RuntimeError("Could not put device into measurement mode. Aborting.")

    except RuntimeError as error:
        print(error)
    except:
        print("An unknown fatal error has occured. Aborting.")
    else:
        print("Successful initialization.")

    # 打开串口线程，持续读数
    try:
        ts = threading.Thread(target=Serial_Read, name='Serial_Read_Thread')

    except:
        raise EOFError
    ts.start()  # start reading the data from the Serai port

    temp = input("统一IMU参考系?(Y/N)")
    if temp == 'Y' or temp == 'y':
        #REu2Es = Norm_Cordinate()
        [REu2Es, REu_2Es] = Norm_Cordinate(callback)
    else:
        REu2Es = np.eye(3)
        REu_2Es = np.eye(3)

    while 1:
        temp = input("记录当前躯体航向角？（Y/N)")
        if temp == 'Y' or temp == 'y':
            break
    ztheta_initial = Record_Initial_Yaw()

    while 1:
        temp = input("Set the initial position?(Y/N)")
        if temp == 'Y' or temp == 'y':
            break
    #Quat_Relative_Zero_Point = Set_Initial_Pos()  # 获得初始状态值
    [Quat_Relative_Zero_Point, Quat_Relative_Zero_Point_] = Set_Initial_Pos(callback)
    RSs2Js = cal_RSsJs(Quat_Relative_Zero_Point[0, 0], Quat_Relative_Zero_Point[0, 1], Quat_Relative_Zero_Point[0, 2],
                       Quat_Relative_Zero_Point[0, 3],ztheta_initial)
    # 是否记录数据
    Flag_Data_Record = input("Record the data?(Y/N)")
    if Flag_Data_Record == 'Y' or Flag_Data_Record == 'y':
        Flag_Data_Record = True
        # 获取当前时间作为文件名，建立空白.csv文档
        Current_Time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        os.mknod('%s.csv' % Current_Time)
    else:
        Flag_Data_Record = False

    time_start = time.time()

    num = 0

    while (1):
        #read the data from the imu and the xsens
        Get_Quat(Quat)
        xsens_Data = Read_Xsens_Quaternion(callback)

        Rot_Mat_u2s = Cur_Quat2Relative_R(Quat_Relative_Zero_Point, (np.asarray(Quat)).reshape((1, -1)), REu2Es, RSs2Js)
        upper_o = Get_Limb_Pos(Rot_Mat_u2s)
        [xtheta_temp, ytheta_temp, ztheta_temp] = Get_Euler_Angle(Rot_Mat_u2s)


        Rot_Mat_u_2s = Cur_Quat2Relative_R(Quat_Relative_Zero_Point_, (np.array([Quat[0:4],xsens_Data])).reshape((1, -1)), REu_2Es, RSs2Js)
        upper_o_ = Get_Limb_Pos(Rot_Mat_u_2s)
        [xtheta_temp_, ytheta_temp_, ztheta_temp_] = Get_Euler_Angle(Rot_Mat_u_2s)

        num += 1
        Points_Num = list(range(num))
        if Flag_Data_Record:
            df = DataFrame(
                [[xtheta_temp, ytheta_temp, ztheta_temp, upper_o[0][0, 0], upper_o[0][1, 0], upper_o[0][2, 0],
                  xtheta_temp_, ytheta_temp_, ztheta_temp_, upper_o_[0][0, 0], upper_o_[0][1, 0], upper_o_[0][2, 0]]])
            df.to_csv('%s.csv' % Current_Time, mode='a', header=False, index=False)

        time.sleep(0.01)
        xsens_Data = Read_Xsens_Quaternion(callback)
        print("\rNO%d\t\t|X:%0.3f\tY:%0.3f\tZ:%0.3f|\t\t|X_:%0.3f\tY_:%0.3f\tZ_:%0.3f|\t\t|Xtheta:%0.3f\tYtheta:%0.3f\tZtheta:%0.3f|\t"
              % (num, upper_o[0][0, 0], upper_o[0][1, 0], upper_o[0][2, 0]
                    ,upper_o_[0][0, 0], upper_o_[0][1, 0], upper_o_[0][2, 0],xtheta_temp, ytheta_temp, ztheta_temp), end='', flush=True)

        '''if len(xtheta) > 500:
            time_end = time.time()
            break'''
    print('Time:%f' % (time_end - time_start))
    Flag_Serial_Read = False




