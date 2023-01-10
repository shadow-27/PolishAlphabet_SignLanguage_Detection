import os
import cv2
import math
import numpy as np
import copy
import itertools
import csv
from collections import Counter
from collections import deque
from utils import CvFpsCalc
from matplotlib import pyplot as plt
import sys
from model.keypoint_classifier.keypoint_classifier import *

def linear_approx_x(x_1,x_2):
    points_x = []
    if(x_1>=0 and x_2>=0):
        if(x_1 > x_2):
            points_x.append(x_2 + ((x_1 - x_2) * 2 /3))
            points_x.append(x_2 + ((x_1 - x_2) * 1 /3))
            return points_x
        else:
            points_x.append(x_1 + ((x_2 - x_1) * 1 /3))
            points_x.append(x_1 + ((x_2 - x_1) * 2 /3))
            return points_x

    if(x_1>=0 and x_2<=0):
        points_x.append(x_2 + ((x_1 + abs(x_2)) * 2 /3))
        points_x.append(x_2 + ((x_1 + abs(x_2)) * 1 /3))
        return points_x

    if(x_1<=0 and x_2>=0):
        points_x.append(x_1 + ((x_2 + abs(x_1)) * 1 /3))
        points_x.append(x_1 + ((x_2 + abs(x_1)) * 2 /3))
        return points_x
    
    if(x_1<=0 and x_2<=0):
        if(x_1 > x_2):
            points_x.append(x_1 + ((x_2)-(x_1)) * 1/3)
            points_x.append(x_1 + ((x_2)-(x_1)) * 2/3)
            return points_x
        else:
            points_x.append(x_1 + ((x_2)-(x_1)) * 1/3)
            points_x.append(x_1 + ((x_2)-(x_1)) * 2/3)
            return points_x
    
def linear_approx_y(y_1,y_2):
    points_y = []
    if(y_1>=0 and y_2>=0):
        if(y_1 > y_2):
            points_y.append(y_2 + ((y_1 - y_2) * 2 /3))
            points_y.append(y_2 + ((y_1 - y_2) * 1 /3))
            return points_y
        else:
            points_y.append(y_1 + ((y_2 - y_1) * 1 /3))
            points_y.append(x_1 + ((y_2 - y_1) * 2 /3))
            return points_y

    if(y_1>=0 and y_2<=0):
        points_y.append(y_2 + ((y_1 + abs(y_2)) * 2 /3))
        points_y.append(y_2 + ((y_1 + abs(y_2)) * 1 /3))
        return points_y

    if(y_1<=0 and y_2>=0):
        points_y.append(y_1 + ((y_2 + abs(y_1)) * 1 /3))
        points_y.append(y_1 + ((y_2 + abs(y_1)) * 2 /3))
        return points_y
    
    if(y_1<=0 and y_2<=0):
        if(y_1 > y_2):
            points_y.append(y_1 + ((y_2)-(y_1)) * 1/3)
            points_y.append(y_1 + ((y_2)-(y_1)) * 2/3)
            return points_y
        else:
            points_y.append(y_1 + ((y_2)-(y_1)) * 1/3)
            points_y.append(y_1 + ((y_2)-(y_1)) * 2/3)
            return points_y
    
# filling empty frames linearly
def fill_coordinates(relative_list_in_fnc):
    list_x=[0,0,0,0,0,0,0]
    list_y=[0,0,0,0,0,0,0]
    if(len(relative_list_in_fnc)==0):
        return list_x
    if(len(relative_list_in_fnc)==1):
        list_x[6]=relative_list_in_fnc[0][0]
        list_y[6]=relative_list_in_fnc[0][1]

        # filling 0 values

        list_x[5]= list_x[6] * 6 / 7
        list_y[5]= list_y[6] * 6 / 7

        list_x[4]=list_x[6] * 5 / 7
        list_y[4]=list_y[6] * 5 / 7

        list_x[3]=list_x[6] * 4 / 7
        list_y[3]=list_y[6] * 4 / 7

        list_x[2]=list_x[6] * 3 / 7
        list_y[2]=list_y[6] * 3 / 7

        list_x[1]=list_x[6] * 2 / 7
        list_y[1]=list_y[6] * 2 / 7

        list_x[0]=list_x[6] * 1 / 7
        list_y[0]=list_y[6] * 1 / 7
    
    if(len(relative_list_in_fnc)==2):
        list_x[6]=relative_list_in_fnc[1][0]
        list_y[6]=relative_list_in_fnc[1][1]

        list_x[3]=relative_list_in_fnc[0][0]
        list_y[3]=relative_list_in_fnc[0][1]

        # filling 0 values
        # The only case with if:

        # list_x[5]=abs((list_x[6] - list_x[3])) + (list_x[6] - list_x[3]) * 2 / 3 
        # list_y[5]=abs((list_y[6] - list_y[3])) + (list_y[6] - list_y[3]) * 2 / 3 

        # list_x[4]=abs((list_x[6] - list_x[3])) + (list_x[6] - list_x[3]) * 1 / 3
        # list_y[4]=abs((list_y[6] - list_y[3])) + (list_y[6] - list_y[3]) * 1 / 3 

        coords_x = linear_approx_x(list_x[6],list_x[3])
        coords_y = linear_approx_x(list_y[6],list_y[3])

        list_x[5]= coords_x[0]
        list_y[5]= coords_y[0]

        list_x[4]= coords_x[1]
        list_y[4]= coords_y[1]

        list_x[2]=(list_x[3]) * 3 / 4
        list_y[2]=(list_y[3]) * 3 / 4

        list_x[1]=(list_x[3]) * 2 / 4
        list_y[1]=(list_y[3]) * 2 / 4

        list_x[0]=(list_x[3]) * 1 / 4
        list_y[0]=(list_y[3]) * 1 / 4
    
    if(len(relative_list_in_fnc)==3):
        list_x[6]=relative_list_in_fnc[2][0]
        list_y[6]=relative_list_in_fnc[2][1]

        list_x[4]=relative_list_in_fnc[1][0]
        list_y[4]=relative_list_in_fnc[1][1]

        list_x[2]=relative_list_in_fnc[0][0]
        list_y[2]=relative_list_in_fnc[0][1]

        # filling 0 values

        list_x[5]=(list_x[6] + list_x[4]) / 2
        list_y[5]=(list_y[6] + list_y[4]) / 2

        list_x[3]=(list_x[4] + list_x[2]) / 2
        list_y[3]=(list_y[4] + list_y[2]) / 2

        list_x[1]=(list_x[2]) * 2 / 3
        list_y[1]=(list_y[2]) * 2 / 3

        list_x[0]=(list_x[2]) * 1 / 3
        list_y[0]=(list_y[2]) * 1 / 3
    
    if(len(relative_list_in_fnc)==4):
        list_x[6]=relative_list_in_fnc[3][0]
        list_y[6]=relative_list_in_fnc[3][1]

        list_x[4]=relative_list_in_fnc[2][0]
        list_y[4]=relative_list_in_fnc[2][1]

        list_x[3]=relative_list_in_fnc[1][0]
        list_y[3]=relative_list_in_fnc[1][1]

        list_x[1]=relative_list_in_fnc[0][0]
        list_y[1]=relative_list_in_fnc[0][1]

        # filling 0 values

        list_x[5]=(list_x[6] + list_x[4]) / 2
        list_y[5]=(list_y[6] + list_y[4]) / 2

        list_x[2]=(list_x[3] + list_x[1]) / 2
        list_y[2]=(list_y[3] + list_y[1]) / 2

        list_x[0]=(list_x[1]) / 2
        list_y[0]=(list_y[1]) / 2

    if(len(relative_list_in_fnc)==5):
        list_x[6]=relative_list_in_fnc[4][0]
        list_y[6]=relative_list_in_fnc[4][1]

        list_x[4]=relative_list_in_fnc[3][0]
        list_y[4]=relative_list_in_fnc[3][1]

        list_x[3]=relative_list_in_fnc[2][0]
        list_y[3]=relative_list_in_fnc[2][1]

        list_x[1]=relative_list_in_fnc[1][0]
        list_y[1]=relative_list_in_fnc[1][1]

        list_x[0]=relative_list_in_fnc[0][0]
        list_y[0]=relative_list_in_fnc[0][1]

        # filling 0 values

        list_x[5]=(list_x[6] + list_x[4]) / 2
        list_y[5]=(list_y[6] + list_y[4]) / 2

        list_x[2]=(list_x[3] + list_x[1]) / 2
        list_y[2]=(list_y[3] + list_y[1]) / 2
    
    if(len(relative_list_in_fnc)==6):
        list_x[6]=relative_list_in_fnc[5][0]
        list_y[6]=relative_list_in_fnc[5][1]

        list_x[5]=relative_list_in_fnc[4][0]
        list_y[5]=relative_list_in_fnc[4][1]

        list_x[4]=relative_list_in_fnc[3][0]
        list_y[4]=relative_list_in_fnc[3][1]

        list_x[3]=relative_list_in_fnc[2][0]
        list_y[3]=relative_list_in_fnc[2][1]

        list_x[1]=relative_list_in_fnc[1][0]
        list_y[1]=relative_list_in_fnc[1][1]

        list_x[0]=relative_list_in_fnc[0][0]
        list_y[0]=relative_list_in_fnc[0][1]

        # filling 0 values

        list_x[2]=(list_x[3] + list_x[1]) / 2
        list_y[2]=(list_y[3] + list_y[1]) / 2
    
    if(len(relative_list_in_fnc)==7):
        list_x[6]=relative_list_in_fnc[6][0]
        list_y[6]=relative_list_in_fnc[6][1]
        
        list_x[5]=relative_list_in_fnc[5][0]
        list_y[5]=relative_list_in_fnc[5][1]

        list_x[4]=relative_list_in_fnc[4][0]
        list_y[4]=relative_list_in_fnc[4][1]

        list_x[3]=relative_list_in_fnc[3][0]
        list_y[3]=relative_list_in_fnc[3][1]

        list_x[2]=relative_list_in_fnc[2][0]
        list_y[2]=relative_list_in_fnc[2][1]

        list_x[1]=relative_list_in_fnc[1][0]
        list_y[1]=relative_list_in_fnc[1][1]

        list_x[0]=relative_list_in_fnc[0][0]
        list_y[0]=relative_list_in_fnc[0][1]

    return list_x,list_y

# function to store 0's in coordinates list on good places. Every list, no matter how many frames read has to have the same length. 0 will not be read anyway.
# indexes to store how we will pick frames according to how many are they
def define_indexing(length):
    indexes_7 = [ [6],
                [3,6],
                [2,4,6],
                [1,3,4,6],
                [0,1,3,4,6],
                [0,1,3,4,5,6],
                [0,1,2,3,4,5,6]]
    indexes_6 = [ [5],
                [3,5],
                [1,3,5],
                [1,2,4,5],
                [0,1,3,4,5],
                [0,1,2,3,4,5]]
    indexes_5 = [ [4],
                [2,4],
                [1,2,4],
                [0,1,3,4],
                [0,1,2,3,4]]
    indexes_4 = [ [3],
                [1,3],
                [0,1,3],
                [0,1,2,3]]
    indexes_3 = [ [2],
                [1,2],
                [0,1,2]]
    indexes_2 = [ [1],
                [0,1]]
    indexes_1 = [ [0]]

    if(length==1):
        return indexes_1
    if(length==2):
        return indexes_2
    if(length==3):
        return indexes_3
    if(length==4):
        return indexes_4
    if(length==5):
        return indexes_5
    if(length==6):
        return indexes_6
    if(length==7):
        return indexes_7

def relative_change(x_1,y_1,x_2,y_2,distance):
    if(x_2==999 or y_2==999):
        return 0,0
    new_x = (x_1-x_2)/distance
    new_y = (y_1-y_2)/distance
    
    return new_x,new_y