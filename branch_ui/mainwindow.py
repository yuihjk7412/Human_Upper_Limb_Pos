import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
import time
import numpy as np
import ctypes
from ctypes import *
import serial
import threading
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from multiprocessing import Process, Queue
import os
import data_plot as dp



class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi('mainwindow.ui', self)
        #self.setFixedSize(self.sizeHint())
        self.Process_Flag = False
        self.Record_Flag = False
        self.Initialize_Num = 5
        self.Initialize_Threshold = 0.03

        self.pushButton_exit.clicked.connect(self.Exit_Process)
        self.pushButton_start.clicked.connect(self.Start_Process)
        self.pushButton_datarecord.clicked.connect(self.Start_Record)
        self.pushButton_enddata.clicked.connect(self.End_Record)
        self.pushButton_ini.clicked.connect(self.Set_Initial_Pos)
        # 定时器接收数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.data_receive)

        self.Port_Num = input('PLEASE INPUT THE PORT NUMBER(/dev/ttyUSB*):')

        # 获取当前时间
        self.Current_Time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

        self.Quat = (c_float * 32)()
        self.total_lib = cdll.LoadLibrary("./libtotal.so")
        self.total_lib.imu_data_decode_init()
        self.Limb_Length = 200; #the length between the origin of the platform and the base
        self.Upper_Limb_Origin = np.array([[0], [0], [0], [1]])
        self.Quat_Relative_Zero_Point = np.array([[0,0,0,0,0,0,0,0]])
        self.ini_time = 0

    def Exit_Process(self):
        self.Process_Flag = False
        self.timer.stop()
        app.exit(0)

    def Start_Process(self):
        with serial.Serial("/dev/ttyUSB%d" % int(self.Port_Num), 115200, timeout=0.2) as self.ser:
            print("Serial Port OK!")
        self.ser.close()
        self.Process_Flag = True
        print("Process Started")
        self.timer.start(1)
        #self.Quat_Relative_Zero_Point = self.Set_Initial_Pos()

    def data_receive(self):
        # Do the Serial read and Data collecting HERE
        with serial.Serial("/dev/ttyUSB%d"%int(self.Port_Num), 115200) as ser:
            time.sleep(0.01)
            num = ser.in_waiting
            buf = ser.read(num)

        if num:
            for i in range(num):
                self.total_lib.Packet_Decode(buf[i])
        self.total_lib.get_quat(byref(self.Quat))
        Rot_Mat_u2s = self.Cur_Quat2Relative_R(self.Quat_Relative_Zero_Point, (np.asarray(self.Quat)).reshape((1, -1)))
        upper_o = self.Get_Limb_Pos(Rot_Mat_u2s)
        [xtheta_temp, ytheta_temp, ztheta_temp] = self.Get_Euler_Angle(Rot_Mat_u2s)
        #data recording
        if self.Record_Flag:
            df = DataFrame([[xtheta_temp, ytheta_temp, ztheta_temp, upper_o[0][0, 0], upper_o[0][1, 0], upper_o[0][2, 0]]])
            df.to_csv('%s.csv' % self.Current_Time, mode='a', header=False, index=False)
        self.textEdit_xtheta.setText(str(xtheta_temp))
        self.textEdit_ytheta.setText(str(ytheta_temp))
        self.textEdit_ztheta.setText(str(ztheta_temp))
        self.textEdit_xpos.setText(str(upper_o[0][0, 0]))
        self.textEdit_ypos.setText(str(upper_o[0][1, 0]))
        self.textEdit_zpos.setText(str(upper_o[0][2, 0]))

        #QApplication.processEvents()

    def Set_Initial_Pos(self):
        temp = np.zeros([1, 32])
        print("Initialization Processing")
        i = 0
        while i < self.Initialize_Num:
            with serial.Serial("/dev/ttyUSB%d" % int(self.Port_Num), 115200) as ser:
                time.sleep(0.01)
                num = ser.in_waiting
                buf = ser.read(num)

            if num:
                for i_n in range(num):
                    self.total_lib.Packet_Decode(buf[i_n])
            self.total_lib.get_quat(byref(self.Quat))
            print(np.array(self.Quat))
            for j in range(32):
                temp[i, j] = self.Quat[j]
            print("The %d time initialization:" % (i + 1))
            Max_Dif = np.max(temp, axis=0, keepdims=True) - np.min(temp, axis=0, keepdims=True)
            assert (Max_Dif.shape == (1, 32))
            if ((Max_Dif > self.Initialize_Threshold).any()):
                i = 0
                print(Max_Dif > self.Initialize_Threshold)
                print("Initialization Failure!Please stay still!")
                temp = np.zeros([1, 32])
                continue
            time.sleep(0.5)
            temp = np.row_stack((temp, np.zeros([1, 32])))
            i = i + 1

        temp = np.delete(temp, -1, 0)
        average = np.average(temp, axis=0)
        average = np.reshape(average, (1, 32))
        print("Initialization Completed!")
        print('The 1 IMU initial Quat is:', average[0][0:4])
        print('The 2 IMU initial Quat is:', average[0][4:8])
        self.Quat_Relative_Zero_Point = average
        self.pushButton_ini.setEnabled(False)
        return average

    def Quat2R(self,q0, q1, q2, q3):
        '''

        :param q0:
        :param q1:
        :param q2:
        :param q3:
        :return:
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

    def Cur_Quat2Relative_R(self,Relative_Zero_Points, Current_Points,Total_Arr_Length = 32):
        RSs2Js = np.eye(3)
        RSs02e = self.Quat2R(Relative_Zero_Points[0, 0], Relative_Zero_Points[0, 1], Relative_Zero_Points[0, 2],
                        Relative_Zero_Points[0, 3])
        RSu02e = self.Quat2R(Relative_Zero_Points[0, 4], Relative_Zero_Points[0, 5], Relative_Zero_Points[0, 6],
                        Relative_Zero_Points[0, 7])
        RSs2e = self.Quat2R(Current_Points[0, 0], Current_Points[0, 1], Current_Points[0, 2], Current_Points[0, 3])
        RSu2e = self.Quat2R(Current_Points[0, 4], Current_Points[0, 5], Current_Points[0, 6], Current_Points[0, 7])
        RSs2Ss0 = np.dot(np.linalg.inv(RSs02e), RSs2e)
        RSu2Su0 = np.dot(np.linalg.inv(RSu02e), RSu2e)

        RJs2Js0 = np.dot(RSs2Js, RSs2Ss0)
        RJs2Js0 = np.dot(RJs2Js0, np.linalg.inv(RSs2Js))

        RSu2Ju = np.dot(RSs2Js, np.linalg.inv(RSs02e))
        RSu2Ju = np.dot(RSu2Ju, RSu02e)
        RJu2Ju0 = np.dot(RSu2Ju, RSu2Su0)
        RJu2Ju0 = np.dot(RJu2Ju0, np.linalg.inv(RSu2Ju))

        RJu2Js = np.dot(np.linalg.inv(RJs2Js0), RJu2Ju0)

        return RJu2Js

    def Get_Limb_Pos(self,Rot_Mat):
        assert (Rot_Mat.shape == (3, 3))
        Trans = np.column_stack((Rot_Mat, np.zeros([3, 1])))
        Trans = np.row_stack((Trans, np.zeros((1, 4))))
        Linear_Trans = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -self.Limb_Length], [0, 0, 0, 1]])
        Trans = np.dot(Trans, Linear_Trans)
        u_o = np.dot(Trans, self.Upper_Limb_Origin)
        return [u_o]

    def Get_Euler_Angle(self,Rot_Mat):
        xtheta = np.degrees(np.arcsin(-Rot_Mat[1][2]))
        ytheta = np.degrees(np.arctan(Rot_Mat[0][2] / Rot_Mat[2][2]))
        ztheta = np.degrees(np.arctan(Rot_Mat[1][0] / Rot_Mat[1][1]))
        return [xtheta, ytheta, ztheta]

    def Start_Record(self):
        os.mknod('%s.csv' % self.Current_Time)
        self.Record_Flag = True

    def End_Record(self):
        self.Record_Flag = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
