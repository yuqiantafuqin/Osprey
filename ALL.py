# -*- coding:utf-8 -*-
from Nan import Nan
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import datetime
import os, sys


class ALL():
    '''
    全品种回测, 存储的是按照交易数绘制的pnl图像, 速度会快很多
    '''
    def __init__(self, signal, saved_base_path):
        self.signal = signal
        self.instrument_stat_dict = {}
        self.saved_base_path = saved_base_path
        self.error_list = {}

    def run_once(self, instrument):
        self.signal.signal_config['Benchmark'] = instrument
        self.signal.signal_config['data_set'] = 'all'

        signal_name = self.signal.signal_config['Name']
        saved_path = os.path.join(self.saved_base_path, signal_name+ '_' + instrument +'.png')
        print signal_name, '+', instrument
        Na = Nan(self.signal)
        Na.run(data_mode='mongo', print_info=False, print_signal_info=True, print_summary=False, process_bar=True, plot1=True, plot2=False, saved_path=saved_path, summary=True, summary2=False)
        self.instrument_stat_dict[instrument] = Na.stat_df.copy()


    def run(self):
        instrument_id_list = []
        from pymongo import MongoClient
        client = MongoClient('192.168.1.99', 27017)
        db = client['future_data']
        cursor = db['data_all'].find({}, {'instrument_id':1})  
        for j in cursor: 
            instrument_id_list.append(j['instrument_id'])
        print '*'*30, 'instrument_id', '*'*30
        print instrument_id_list
        print '*'*70

        for i in instrument_id_list:
            try:
            print '\n', '*'*30, i, '*'*30
            self.run_once(i)
            except Exception, e:
                print '******************error', i, e, '************************'
                self.error_list[i] = e
                continue

    def get_error_list():
        return self.error_list



sys.path.append(r'D:\MaikeKongfu\signal\shipan')  
import c_c_c_c_l_h_BE_20170602 as signal    
s1 = signal.signal()
from ALL import ALL
all = ALL(signal=s1, saved_base_path='D:\MaikeKongfu\signal_image')
all.run()
