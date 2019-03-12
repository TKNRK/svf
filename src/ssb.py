import logging
from ctypes import *

from constants import *
from svf.profile import Profile


Profile.set_profile(DATASET_NAME, "example.log")
number_of_nodes, number_of_edges, dim_hd = Profile.get_profile()

NUMBER_OF_BUFFERS = 9

SSB_LAYOUT = 0
SSB_PROJECTION = 1
SSB_POSITION = 2
SSB_PICKER = 3
SSB_FILTER = 4
SSB_FLAGS = 5
SSB_ISDRAWN = 6
SSB_EDGELIST = 7
SSB_ISOLATION = 8


def call_buffer(binding=0):
    if binding==0: return SSB_Layout()
    elif binding == 1: return SSB_Projection()
    elif binding == 2: return SSB_Position()
    elif binding == 3: return SSB_Picker()
    elif binding == 4: return SSB_Filter()
    elif binding == 5: return SSB_Flags()
    elif binding == 6: return SSB_isDrawn()
    elif binding == 7: return SSB_Edgelist()
    elif binding == 8: return SSB_Isolation()
    else:
        logging.error("Invalid SSB call")
        raise Exception()


# 高次元空間の頂点の座標
class SSB_Layout(Structure):
    _fields_ = [('layout_hd', c_float * (number_of_nodes * dim_hd))]


# 射影ベクトル
class SSB_Projection(Structure):
    _fields_ = [('projection', c_float * (dim_hd * 2))]


# 射影空間の頂点の座標
class SSB_Position(Structure):
    _fields_ = [('positions', c_float * (number_of_nodes * 3))]


# ドラッグ検知のためのSSB
class SSB_Picker(Structure):
    _fields_ = [('clicked_x', c_uint), ('clicked_y', c_uint),
                ('pick_z', c_float),   ('pick_lock', c_int),
                ('pick_id', c_int)]


# フィルタイングのための中心性を乗せたSSB
class SSB_Filter(Structure):
    _fields_ = [('score', c_float * number_of_nodes)]


# フィルタイングのための中心性を乗せたSSB
class SSB_Flags(Structure):
    _fields_ = [('vertexColor', c_int * number_of_nodes)]


# フィルタリングの結果，各頂点が描かれているかの状態を載せたSSB
class SSB_isDrawn(Structure):
    _fields_ = [('isDrawn', c_int * number_of_nodes)]


# エッジリストを載せたSSB
class SSB_Edgelist(Structure):
    _fields_ = [('edges', c_uint * (number_of_edges * 2))]


# フィルタリングの結果，隣接頂点のいなくなったかどうかの状態を載せたSSB
class SSB_Isolation(Structure):
    _fields_ = [('isIsolated', c_int * number_of_nodes)]
