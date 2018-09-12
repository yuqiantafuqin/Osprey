# coding=utf-8
import libwingchunbl

import kungfu.longfist.longfist_constants as lf
import kungfu.longfist.longfist_structs as lf_structs
import kungfu.longfist.longfist_utils as lf_utils

import functools
import ctypes
import datetime, time


import sys
import json, yaml, os
import time, datetime
import numpy as np
import pandas as pd 
from pandas import Timedelta
from prettytable import PrettyTable
from collections import deque, defaultdict
from MinNan import MinNan

def tree():
    return defaultdict(tree)


class FirstStrategy(libwingchunbl.WcStrategy):
    # 类说明函数  ===========================================================================================================================================
    def get_version(self):
        return '~'*10 + 'version 0.1 updated in 2017/05/19' + '~'*10

    def get_name(self):
        return 'FirstStrategy'

    def __init__(self):
        self.tickers = ['rb1710']
        self.instrument_id = self.tickers[0]
        self.last_traded_price = {}
        self.open_prices = {}
        self.limit_price = {}
        self.order_volume = 1
        self.cur_positions = {ticker: 0 for ticker in self.tickers}
        self.order_suspended = {ticker: False for ticker in self.tickers}
        self.is_running = False

        libwingchunbl.WcStrategy.__init__(self, 'FIRST_STRATEGY', -1)
        # if raw_input('check version:\n %s \nt or f:' %self.get_version()).upper() != 'T':
        #     raise Exception('version wrong!')
        #self.info("heihei")
        self.launch_time = time.strftime('%Y%m%d_%H%M%S')  #启动时间

    # constraint函数 ===========================================================================================================================================
    def init_risk_constraints(self):
        self.cancel_order_cumsum = 0
        self.place_order_cumsum = 0
        self.max_place_order_cumsum = 400
        self.max_cancel_order_cumsum = 400
        self.lock_position_max_position = 100
        self.max_holding_position = 100
        self.order_record_for_risk = deque(maxlen=100)

    def risk_constraints(self):
        if len(self.order_record_for_risk) >= 10 and (self.order_record_for_risk[-1] - self.order_record_for_risk[-10]) <= 1:
            print self.order_record_for_risk[-10:]
            raise Exception('Warning, short period order action too frequently!!')
        if self.place_order_cumsum >= self.max_place_order_cumsum:
            raise Exception('Warning, place_order_cumsum:' + str(self.place_order_cumsum) + ' above the threshold:' + str(self.max_place_order_cumsum))
        if self.cancel_order_cumsum >= self.max_cancel_order_cumsum:
            raise Exception('Warning, cancel_order_cumsum:' + str(self.cancel_order_cumsum) + ' above the threshold:' + str(self.max_cancel_order_cumsum))
        if max(self.investor_position_ctp[self.instrument_id]['1'], self.investor_position_ctp[self.instrument_id]['0']) >= self.max_holding_position:
            raise Exception('Warning, holding position nums above the risk level: %s' %self.max_holding_position)

    def init_misc(self):
        self.subscribe(self.get_source(), self.tickers)
        self.req_position(self.get_source())
        self.configure_path()
        self.manage_portfolio()
        self.init_risk_constraints()
        self.connect_mongo_client()
        self.activate_signal()
        self.prepare_signal_data()
        self.warm_signal()
        self.prepare_trade()

        print '8888888888888888888888888888888888888888888888888888888888888888'
        print '88                     SIC * PARVIS * MAGNA!                  88'
        print '8888888888888888888888888888888888888888888888888888888888888888'


    # 流程函数 交易前===========================================================================================================================================
    def configure_path(self):
        #print self.instrument_id
        #self.instrument = 'rb'
        self.one_jump = 1
        self.base_config = {"root_dir": os.path.join("/shared/strategy/"),
            "tranx_record_dir": "/shared/strategy/tranx_record/",
            'save_data_dir':"/shared/strategy/data_record/",
            'program_record_dir': "/shared/strategy/program_record/"}

    def manage_portfolio(self):
        self.fund_name = 'HongFei_2'

    def activate_signal(self):
        base_path = '/shared/strategy/signal/shipan'
        signal_name = ['s_c_h_20170601','s_c_h_b_20170601','c_c_h_l_l_h_BE_20170602', 'c_c_h_l_l_h_20170601' ,'c_c_h_l_c_c_20170601', 'c_c_c_c_l_h_BE_20170602'] #'bourne_identity_01',
                        #'bourne_identity_03', 'bourne_identity_05','break_1min', 'kingkeltner_ts']
        self.strategies = {}
        for i in signal_name:
            self.strategies[i.split('.py')[0].strip()] = MinNan(signal_name=i, base_path=base_path, logger=self.info, config_dict=self.base_config)
            self.strategies[i.split('.py')[0].strip()].signal_name = i

    def connect_mongo_client(self):
        from pymongo import MongoClient
        client = MongoClient('127.0.0.1', 27017)
        self.mongo_db = client['TranxRecord']
        if raw_input('pls check fund:\n %s \nt or f:' %self.fund_name).upper() != 'T':
            raise Exception('fund name error')
        self.mongo_db_tranx_record = self.mongo_db[self.fund_name]   #fund name

    def prepare_signal_data(self):
        self.schedule = self.get_trading_time()
        self.info('[trading_schedule] ' + str(self.schedule))
        for s in self.strategies.keys():
            self.strategies[s].data_prepare(r'/shared/strategy/history_data_1min/rb1710_1min.csv')
            #print self.info('-'*30 + str(self.strategies[s].history_data_df['dates'].values[-1]))

    def warm_signal(self):
        # 预热信号, 在交易前
        wgt = {}
        for s in self.strategies.keys():
            print 'warm signal: ', s
            self.strategies[s].warm_signal(200)

    def prepare_trade(self):
        for s in self.strategies.keys():
            self.strategies[s].prepare_trade()
        self.tick_time_list = deque(maxlen=100)     # 记录所有收到tick的时间
        self.tick_price_list = deque(maxlen=100)    # tick的price
        self.tranx_record_dict = {}  # 一张大表记录所有交易
        self.trade_queue = {}
        self.offset_str_to_flag_dict = {'Open': lf.LfOffsetFlagType.Open, 'CloseYesterday': lf.LfOffsetFlagType.CloseYesterday, 'CloseToday': lf.LfOffsetFlagType.CloseToday}
        self.offset_flag_to_str_dict = {lf.LfOffsetFlagType.Open:'Open', lf.LfOffsetFlagType.CloseYesterday:'CloseYesterday', lf.LfOffsetFlagType.CloseToday:'CloseToday'}


    def on_market_data_level1(self, data, source, rcv_time):
        if data.InstrumentID == self.instrument_id and self.in_trading_time(data.UpdateTime):
            self.last_tick = {}
            self.last_tick['update_time'] = data.UpdateTime
            self.last_tick['trading_day'] = data.TradingDay
            self.last_tick['last_price'] = data.LastPrice
            self.last_tick['volume'] = data.Volume
            
            self.tick_price_list.append(data.LastPrice)
            self.tick_time_list.append(data.UpdateTime)
            if len(self.tick_time_list) <=1 : return
            strategies_order_dict = tree()
            process = {0:"\\", 1:"-", 2:"/", 3:'|'}
            sys.stdout.write("\r%s %s %s" %(self.instrument_id, data.UpdateTime, process[int(data.UpdateTime[-2:])%4]))
            sys.stdout.flush()

            # minNan逻辑
            for s_name in self.strategies:
                self.strategies[s_name].update_kline(self.tick_time_list, self.last_tick)
                self.strategies[s_name].work(self.tick_time_list, self.last_tick)

            if self.position_get is not True: return  #直到查到持仓, 才继续

            # 遍历所有策略, 返回订单字典
            for s_name in self.strategies:
                one_strategies_order_dict = self.strategies[s_name].trade_order(self.tick_time_list, self.tick_price_list, self.last_tick)
               # print s_name, one_strategies_order_dict
                if len(one_strategies_order_dict) > 0:
                    strategies_order_dict[s_name].update(one_strategies_order_dict)
                    strategies_order_dict[s_name]['signal_name'] = s_name
                    strategies_order_dict[s_name]['trigger_price'] = data.LastPrice
                    strategies_order_dict[s_name]['trigger_time'] = data.UpdateTime
                    strategies_order_dict[s_name]['trigger_date'] = data.TradingDay
                    strategies_order_dict[s_name]['tick_since_entry'] = 0 
                    self.info('\n[new order] ' + s_name + ' ' + str(strategies_order_dict[s_name]))

            # 自成交
            self.trade_self(strategies_order_dict)

            # 处理订单, 更新trade_queue
            self.process_offset(strategies_order_dict)

            # 执行订单
            self.trade_order()
            




    def trade_self(self, strategies_order_dict):
        if len(strategies_order_dict) == 0: return
        # 自成交
        position = {}
        for s_name in strategies_order_dict:
            if 'size' in  strategies_order_dict[s_name] and 'direction' in strategies_order_dict[s_name]:
                position[s_name] = strategies_order_dict[s_name]['size'] * strategies_order_dict[s_name]['direction']
        if len(position) == 0: return 
        position_arr = np.array(position.values())
        if np.sum(position_arr) > 0:
            # 多大于空
            short_position = np.sum(position_arr[position_arr < 0])
            for s_name in position:
                if position[s_name] < 0:  # 空平
                    position[s_name] = 0
                    continue
                elif position[s_name] > 0 and position[s_name] >= abs(short_position):
                    position[s_name] = position[s_name] - abs(short_position)
                    short_position = 0 
                elif position[s_name] > 0 and position[s_name] < abs(short_position):
                    short_position += position[s_name]
                    position[s_name] = 0
                elif position[s_name] == 0:
                    position[s_name] = 0
                else:
                    print position, s_name, position[s_name]
                    raise Exception('trade_self() long position beyond short position!')

        elif np.sum(position_arr) < 0:
            # 空大于多
            long_position = np.sum(position_arr[position_arr > 0])
            for s_name in position:
                if position[s_name] > 0:   #多平
                    position[s_name] = 0
                    continue
                elif position[s_name] < 0 and abs(position[s_name]) > long_position:
                    position[s_name] = position[s_name] + long_position
                    long_position = 0 
                elif position[s_name] < 0 and abs(position[s_name]) <= long_position:
                    long_position -= abs(position[s_name])
                    position[s_name] = 0
                elif position[s_name] == 0:
                    position[s_name] = 0
                else:
                    print position, s_name, position[s_name]
                    raise Exception('trade_self() short position beyond long position!')

        elif np.sum(position_arr) == 0:
            # 多空相等
            for s_name in position:
                position[s_name] = 0
        
        for s_name in position:
            strategies_order_dict[s_name]['ExecSize'] = abs(position[s_name])
            if strategies_order_dict[s_name]['ExecSize'] == 0:
                strategies_order_dict[s_name]['deal'] = 'Yes'
                if strategies_order_dict[s_name]['action'] == 'open':
                    self.strategies[s_name].tranx_order['Open deal'] = 'Yes'
                    self.strategies[s_name].tranx_order['Open time'] = self.last_tick['trading_day'] + ' ' + self.last_tick['update_time']
                    self.strategies[s_name].tranx_order['Open price'] = self.last_tick['last_price']
                    self.strategies[s_name].tranx_order['Close deal'] = 'No'
                elif strategies_order_dict[s_name]['action'] == 'close':
                    self.strategies[s_name].tranx_order['Close deal'] = 'Yes'
                    self.strategies[s_name].tranx_order['Close time'] = self.last_tick['trading_day'] + ' ' + self.last_tick['update_time']
                    self.strategies[s_name].tranx_order['Close price'] = self.last_tick['last_price']
            elif strategies_order_dict[s_name]['ExecSize'] > strategies_order_dict[s_name]['size']:
                raise Exception('trade_self error')
        print '[trade_self]' + str(strategies_order_dict)
        print '[trade_queue]' + str(self.trade_queue)

    
    def process_offset(self, strategies_order_dict):
        if len(strategies_order_dict) == 0: return
        # 有订单需要执行
        for s_name in strategies_order_dict:
            #if 'ExecSize' in strategies_order_dict[s_name]: 
            print s_name, strategies_order_dict[s_name]['ExecSize'] 
        volume = np.sum([strategies_order_dict[s_name]['ExecSize'] for s_name in strategies_order_dict])
        if volume == 0: return
        position = self.investor_position_ctp[self.instrument_id].copy()   #复制一个仓位
        order_direction = np.sign(np.sum([strategies_order_dict[s_name]['ExecSize'] * strategies_order_dict[s_name]['direction'] for s_name in strategies_order_dict]))   #已处理完自成交. 这里只有多或者空一个方向
        for i in self.trade_queue: # 考虑未成交对于offset的影响
            #if len(self.trade_queue[i]) == 0: break     
            if self.trade_queue[i]['offset_flag'] == 'CloseYesterday' and self.trade_queue[i]['direction'] == -1 and self.trade_queue[i]['deal'] == 'No':
                position['0_Yd'] = position['0_Yd'] - self.trade_queue[i]['ExecSize']
                position['0'] = position['0'] - self.trade_queue[i]['ExecSize']
            elif self.trade_queue[i]['offset_flag'] == 'CloseYesterday' and self.trade_queue[i]['direction'] == 1 and self.trade_queue[i]['deal'] == 'No':
                position['1_Yd'] = position['1_Yd'] - self.trade_queue[i]['ExecSize']
                position['1'] = position['1'] - self.trade_queue[i]['ExecSize']
            elif self.trade_queue[i]['offset_flag'] == 'CloseToday' and self.trade_queue[i]['direction'] == -1 and self.tranx_queue[i]['deal'] == 'No':
                position['0'] = position['0'] - self.trade_queue[i]['ExecSize']
            elif self.trade_queue[i]['offset_flag'] == 'CloseToday' and self.trade_queue[i]['direction'] == 1 and self.tranx_queue[i]['deal'] == 'No':
                position['1'] = position['1'] - self.trade_queue[i]['ExecSize']
            elif self.trade_queue[i]['offset_flag'] == 'Open' and self.trade_queue[i]['direction'] == -1 and self.tranx_queue[i]['deal'] == 'No':
                position['1'] = position['1'] + self.trade_queue[i]['ExecSize']
            elif self.trade_queue[i]['offset_flag'] == 'Open' and self.trade_queue[i]['direction'] == 1 and self.tranx_queue[i]['deal'] == 'No':
                position['0'] = position['0'] + self.trade_queue[i]['ExecSize']
            else:
                raise Exception('process_offset error')
        if self.lock_mode:
            print volume, position
            direction = 0 if order_direction>0 else 1
            lock_flag, lock_offset_flag_dict = self.lock_position(volume, position, direction)
            no_lock_flag, no_lock_offset_flag_dict = self.no_lock_position(volume, position, direction)
            if lock_flag:
                offset_flag_dict =  lock_offset_flag_dict
            else:
                offset_flag_dict = no_lock_offset_flag_dict
        else:
            no_lock_flag, no_lock_offset_flag_dict = self.no_lock_position(volume, position, direction)
            offset_flag_dict = no_lock_offset_flag_dict
        print 'offset_flag_dict', offset_flag_dict

        for s_name in strategies_order_dict:
            size = strategies_order_dict[s_name]['ExecSize']
            for i in offset_flag_dict:
                if size >= offset_flag_dict[i]:
                    strategies_order_dict[s_name]['offset_flag'][i]['volume'] = offset_flag_dict[i]
                    size -= offset_flag_dict[i]
                    offset_flag_dict[i] = 0
                elif size < offset_flag_dict[i]:
                    strategies_order_dict[s_name]['offset_flag'][i]['volume'] = size
                    offset_flag_dict[i] -= size
                    size = 0
                
                trade_volume = strategies_order_dict[s_name]['offset_flag'][i]['volume']
                if trade_volume > 0:
                    trade_id = '[signal]'+ s_name + '[offset]' + i + '[id]' + str(time.time())
                    #strategies_order_dict[s_name]['offset_flag'][i]['order_ref'] = trade_id
                    self.trade_queue[trade_id] = {'offset_flag':i,
                                                'action': strategies_order_dict[s_name]['action'],
                                                'signal': s_name,
                                                'volume': trade_volume,
                                                'ExecSize': trade_volume,
                                                'direction': strategies_order_dict[s_name]['direction'],
                                                'trigger_date': strategies_order_dict[s_name]['trigger_date'],
                                                'signal_name':  strategies_order_dict[s_name]['signal_name'],
                                                'trigger_price': strategies_order_dict[s_name]['trigger_price'],
                                                'trigger_time':  strategies_order_dict[s_name]['trigger_time'], 
                                                'comment':  strategies_order_dict[s_name]['comment'],
                                                'tick_since_entry': 0,
                                                'deal': 'No'} 
        print '[trade_queue]' + str(self.trade_queue)

    

    def trade_order(self):
        # 为了避免下单优化中, 离散状态的订单发生自成交, 这里所有订单都是由行情驱动
        # 按顺序将self.trade_queue下出去
        if len(self.trade_queue) == 0: return
        for i in self.trade_queue:
            order_dict = self.trade_queue[i]
            if order_dict['deal'] == 'Yes' or order_dict.get('AllTraded', 'N') == 'Y': continue
            if order_dict['deal'] == 'No' and order_dict['tick_since_entry'] == 0:
                self.order_dict_exec(order_dict, mode='limit')
            elif order_dict.get('OrderCanceled', 'Nan') == 'N': #没有撤单, 才去撤单
                request_id = order_dict['request_id']
                place_price = order_dict['place_price']
                if (pd.to_datetime(self.last_tick['update_time']) - pd.to_datetime(order_dict['trigger_time'])) >= Timedelta('0 days 00:00:10'):
                    self.color_log('info', 'r', 'trigger time:' + order_dict['trigger_time'] + ' now:' + self.last_tick['update_time'] + ' exceed 20 tick time, use market order')
                    self.cancel_order_exec(request_id)
                elif abs(self.last_tick['last_price'] - place_price) <= 1*self.one_jump:
                    self.info('limit order price below last price 1 jump, limit order keeps!')
                elif abs(self.last_tick['last_price'] - place_price) >= 2*self.one_jump:
                    self.info('limit order price above last price 2 jump, limit order cancel!')
                    self.cancel_order_exec(request_id)
            elif order_dict.get('OrderCanceled', 'Nan') == 'Y':
                if abs(self.last_tick['last_price'] - order_dict['trigger_price']) >= 5 *self.one_jump:
                    self.color_log('info', 'r', 'current price above trigger price 5 jump')
                    order_dict_exec(order_dict, mode='market')
                elif pd.to_datetime(self.last_tick['update_time']) - pd.to_datetime(order_dict['trigger_time']) >= Timedelta('0 days 00:00:10'):
                    self.color_log('info', 'r', 'trigger time:' + order_dict['trigger_time'] + ' now:' + self.last_tick['update_time'] + ' exceed 10 tick time, use market order')
                    self.order_dict_exec(order_dict, mode='market')
                else:
                    self.order_dict_exec(order_dict, mode='limit')
            order_dict['tick_since_entry'] += 1






    # 逻辑函数 交易中 ===========================================================================================================================================
    def lock_position(self, volume, position, order_direction):
        '''
        昨仓为0, 优先开同向仓, 而不是反向平仓, 这样开仓手续费低, 但是无限开今仓, 达到风控上线, 就使用无锁模式
        if position_Yd is 0, open position as priority, but close position in reverse direction
        '''
        sim_position = position.copy()
        offset_flag_dict = {}
        offset_flag_dict['Open'], offset_flag_dict['CloseToday'], offset_flag_dict['CloseYesterday'] = 0, 0, 0
        if order_direction == lf.LfDirectionType.Buy:
            # if order_direction = '0' to buy, then offset = 0 when we don't have short position
            if position[lf.LfDirectionType.Sell] > 0:
                if volume <= position[lf.LfDirectionType.Sell + '_Yd']:
                    # CloseYesterday
                    offset_flag_dict['CloseYesterday'] = volume

                elif volume > position[lf.LfDirectionType.Sell + '_Yd'] and position[lf.LfDirectionType.Sell + '_Yd'] > 0:
                    offset_flag_dict['CloseYesterday'] = position[lf.LfDirectionType.Sell + '_Yd']
                    offset_flag_dict['Open'] = volume - position[lf.LfDirectionType.Sell + '_Yd']

                elif position[lf.LfDirectionType.Sell + '_Yd'] == 0:  #position_yd=0 but position_today>0, keep position_today
                    offset_flag_dict['Open'] = volume
                else:
                    raise Exception('lock_position lf.LfDirectionType.Buy coner case')
            else:
                offset_flag_dict['Open'] = volume
        else:
            # if order_direction = '1' to short, then offset = 0 when we don't have long position
            if position[lf.LfDirectionType.Buy] > 0:
                if volume <= position[lf.LfDirectionType.Buy + '_Yd']:
                    # CloseYesterday
                    offset_flag_dict['CloseYesterday'] = volume

                elif volume > position[lf.LfDirectionType.Buy + '_Yd'] and position[lf.LfDirectionType.Buy + '_Yd'] > 0: 
                    offset_flag_dict['CloseYesterday'] = position[lf.LfDirectionType.Buy + '_Yd']
                    offset_flag_dict['Open'] = volume - position[lf.LfDirectionType.Buy + '_Yd' ]

                elif position[lf.LfDirectionType.Buy + '_Yd'] == 0:
                    offset_flag_dict['Open'] = volume 

                else:
                    raise Exception('lock_position lf.LfDirectionType.Sell coner case')
            else:
                offset_flag_dict['Open'] = volume

        for flag in offset_flag_dict.keys():
            if flag == 'Open':
                if order_direction == '0': sim_position[lf.LfDirectionType.Buy] += offset_flag_dict[flag] 
                if order_direction == '1': sim_position[lf.LfDirectionType.Sell] += offset_flag_dict[flag]
            elif flag == 'CloseYesterday':
                if order_direction == '0': sim_position[lf.LfDirectionType.Buy] -= offset_flag_dict[flag] 
                if order_direction == '1': sim_position[lf.LfDirectionType.Sell] -= offset_flag_dict[flag]
                if order_direction == '0': sim_position[lf.LfDirectionType.Buy + '_Yd'] -= offset_flag_dict[flag]
                if order_direction == '1': sim_position[lf.LfDirectionType.Sell + '_Yd'] -= offset_flag_dict[flag]
            elif flag == 'CloseToday':
                if offset_flag_dict[flag] != 0:
                    raise Exception('offset_flag coner case', offset_flag_dict[flag])
                else:
                    pass

        threshold = max(sim_position[lf.LfDirectionType.Sell], sim_position[lf.LfDirectionType.Buy])
        if threshold > self.lock_position_max_position:
            return False, offset_flag_dict
        else:
            return True, offset_flag_dict

    
    def no_lock_position(self, volume, position, order_direction):
        '''
        昨仓为0, 优先平反向今仓, 而不是开今仓, 手续费会高, 但是风控有仓位上线
        if position_Yd is 0, close today as priority due to sum position up_limit
        '''
        sim_position = position.copy()
        offset_flag_dict = {}
        offset_flag_dict['Open'], offset_flag_dict['CloseToday'], offset_flag_dict['CloseYesterday'] = 0, 0, 0
        if order_direction == lf.LfDirectionType.Buy:
            # if order_direction = '0' to buy, then offset = 0 when we don't have short position
            if volume <= position[lf.LfDirectionType.Sell + '_Yd']:
                # CloseYesterday
                offset_flag_dict['CloseYesterday'] = volume

            elif volume > position[lf.LfDirectionType.Sell + '_Yd'] and position[lf.LfDirectionType.Sell + '_Yd'] > 0:
                offset_flag_dict['CloseYesterday'] = position[lf.LfDirectionType.Sell + '_Yd']
                today_position = position[lf.LfDirectionType.Sell] - position[lf.LfDirectionType.Sell + '_Yd']
                rest_volume = volume - position[lf.LfDirectionType.Sell + '_Yd']
                if rest_volume > today_position:
                    offset_flag_dict['CloseToday'] = today_position
                    offset_flag_dict['Open'] = rest_volume - today_position
                else:
                    offset_flag_dict['CloseToday'] = rest_volume

            elif position[lf.LfDirectionType.Sell + '_Yd'] == 0 and position[lf.LfDirectionType.Sell] > 0:
                today_position = position[lf.LfDirectionType.Sell] 
                if volume > today_position:
                    offset_flag_dict['CloseToday'] = today_position
                    offset_flag_dict['Open'] = volume - today_position
                else:
                    offset_flag_dict['CloseToday'] = volume

            elif position[lf.LfDirectionType.Sell] == 0:
                offset_flag_dict['Open'] = volume

            else:
                raise Exception('no_lock_position lf.LfDirectionType.Buy')
        else:
            # if order_direction = '1' to short, then offset = 0 when we don't have long position
            if volume <= position[lf.LfDirectionType.Buy + '_Yd']:
                # CloseYesterday
                offset_flag_dict['CloseYesterday'] = volume

            elif volume > position[lf.LfDirectionType.Buy + '_Yd'] and position[lf.LfDirectionType.Buy + '_Yd'] > 0:
                offset_flag_dict['CloseYesterday'] = position[lf.LfDirectionType.Buy + '_Yd']
                today_position = position[lf.LfDirectionType.Buy] - position[lf.LfDirectionType.Buy + '_Yd']
                rest_volume = volume - position[lf.LfDirectionType.Buy + '_Yd']
                if rest_volume > today_position:
                    offset_flag_dict['CloseToday'] = today_position
                    offset_flag_dict['Open'] = rest_volume - today_position
                else:
                    offset_flag_dict['CloseToday'] = rest_volume

            elif position[lf.LfDirectionType.Buy + '_Yd'] == 0 and position[lf.LfDirectionType.Buy] > 0:
                today_position = position[lf.LfDirectionType.Buy] 
                if volume > today_position:
                    offset_flag_dict['CloseToday'] = today_position
                    offset_flag_dict['Open'] = volume - today_position
                else:
                    offset_flag_dict['CloseToday'] = volume

            elif position[lf.LfDirectionType.Buy] == 0:
                offset_flag_dict['Open'] = volume

            else:
                raise Exception('no_lock_position lf.LfDirectionType.Sell')

        for flag in offset_flag_dict.keys():
            if flag == 'Open':
                if order_direction == '0': sim_position[lf.LfDirectionType.Buy] += offset_flag_dict[flag] 
                if order_direction == '1': sim_position[lf.LfDirectionType.Sell] += offset_flag_dict[flag]
            elif flag == 'CloseYesterday':
                if order_direction == '0': sim_position[lf.LfDirectionType.Buy] -= offset_flag_dict[flag] 
                if order_direction == '1': sim_position[lf.LfDirectionType.Sell] -= offset_flag_dict[flag]
                if order_direction == '0': sim_position[lf.LfDirectionType.Buy + '_Yd'] -= offset_flag_dict[flag]
                if order_direction == '1': sim_position[lf.LfDirectionType.Sell + '_Yd'] -= offset_flag_dict[flag]
            elif flag == 'CloseToday':
                if order_direction == '0': sim_position[lf.LfDirectionType.Buy] -= offset_flag_dict[flag] 
                if order_direction == '1': sim_position[lf.LfDirectionType.Sell] -= offset_flag_dict[flag]
            else:
                raise Exception('offset_flag coner case', flag)
        return True, offset_flag_dict
        

    def on_rtn_order(self, data, request_id, source, rcv_time):
        ticker = data.InstrumentID
        direction = data.Direction
        volume = data.VolumeTotalOriginal
        volume_traded = data.VolumeTraded
        offset_flag = data.OffsetFlag
        #request_id = data.RequestID
        #order_ref = data.OrderRef
        self.debug('[ORDER] (s){} (tm){} (rid){} (lid){} (ticker){} (v_tot){} (v_traded){} (v_remain){} (d){} (ofs){} (status){}'.format(
            source, rcv_time, request_id, data.OrderRef, ticker, data.VolumeTotalOriginal, data.VolumeTraded, data.VolumeTotal, data.Direction, data.OffsetFlag, data.OrderStatus))
        for i in self.trade_queue:
            if self.trade_queue[i]['request_id'] == request_id:
                trade_id = i          

        if request_id not in self.tranx_record_dict.keys(): 
            self.tranx_record_dict[request_id] = {}
            self.tranx_record_dict[request_id]['RequestID'] = request_id
            self.tranx_record_dict[request_id]['InstrumentID'] = ticker
            self.tranx_record_dict[request_id]['mongo_id'] = str(time.time())
            self.tranx_record_dict[request_id]['server_time'] = time.strftime('%Y-%m-%d %H:%M:%S') 
            self.tranx_record_dict[request_id]['OffsetFlag'] = offset_flag
            self.tranx_record_dict[request_id]['Direction'] = direction
            self.tranx_record_dict[request_id]['place_volume'] = volume
            self.tranx_record_dict[request_id]['place_price'] = float(data.LimitPrice)
            self.tranx_record_dict[request_id]['curr_price'] = self.last_tick['last_price']
            self.tranx_record_dict[request_id]['OrderPriceType'] = data.OrderPriceType
            self.tranx_record_dict[request_id]['OrderRef'] = data.OrderRef
            self.tranx_record_dict[request_id]['error'] = 0
            self.tranx_record_dict[request_id]['TradeTime'] = ''
            self.tranx_record_dict[request_id]['Volume'] = ''
            self.tranx_record_dict[request_id]['TradeCost'] = ''
            self.tranx_record_dict[request_id]['Amount'] = ''
            self.tranx_record_dict[request_id]['slippage'] = 0
            self.tranx_record_dict[request_id]['InsertTime'] = self.last_tick['trading_day'] + ' ' + self.last_tick['update_time']
            self.tranx_record_dict[request_id]['UserID'] = data.UserID
            self.tranx_record_dict[request_id]['ParticipantID'] = data.ParticipantID
            self.tranx_record_dict[request_id]['InvestorID'] = data.InvestorID
            self.tranx_record_dict[request_id]['UserID'] = data.UserID           
            self.tranx_record_dict[request_id]['ExchangeID'] = data.ExchangeID

        if data.OrderStatus == lf.LfOrderStatusType.AllTraded:
            self.color_log('info', 'b', '[OrderStatus] AllTraded ' + ' request_id:' + str(request_id))
            self.tranx_record_dict[request_id]['AllTraded'] = 'Y'

        elif data.OrderStatus == lf.LfOrderStatusType.Canceled:
            self.color_log('info', 'b', '[OrderStatus] OrderCanceled ' + ' request_id:' + str(request_id))
            self.tranx_record_dict[request_id]['Canceled'] = 'Y'
            self.trade_queue[trade_id]['OrderCanceled'] = 'Y'

        elif data.OrderStatus == lf.LfOrderStatusType.PartTradedQueueing:
            self.color_log('info', 'b', '[OrderStatus] PartTradeQueueing '+ ' request_id:' + str(request_id))
            self.tranx_record_dict[request_id]['PartTradedQueueing'] = 'Y'

        elif data.OrderStatus == lf.LfOrderStatusType.NoTradeQueueing:
            self.color_log('info', 'b', '[OrderStatus] NoTradeQueueing ' + ' request_id:' + str(request_id))
            self.tranx_record_dict[request_id]['NoTradeQueueing'] = 'Y'

        elif data.OrderStatus == lf.LfOrderStatusType.PartTradedNotQueueing:
            self.color_log('info', 'b', '[OrderStatus] PartTradedNotQueueing  Part order has been canceled!' + ' request_id:' + str(request_id))
            self.tranx_record_dict[request_id]['PartTradedNotQueueing'] = 'Y'

        elif data.OrderStatus == lf.LfOrderStatusType.NoTradeNotQueueing:
            self.color_log('info', 'b', '[OrderStatus] NoTradeNotQueueing ' + ' request_id:' + str(request_id))
            self.tranx_record_dict[request_id]['NoTradeNotQueueing'] = 'Y'
        else:
            print 'OrderStatus exception OrderStatus:' + data.OrderStatus


        

    def on_rtn_trade(self, data, request_id, source, rcv_time):
        ticker = data.InstrumentID
        direction = data.Direction
        volume = data.Volume
        price = data.Price
        offset_flag = data.OffsetFlag
        amount = price * volume

        self.debug('[TRADE] (s){} (tm){} (rid){} (lid){} (ticker){} (p){} (v){} (d){} (ofs){}'.format(
            source, rcv_time, request_id, data.OrderRef, ticker, data.Price, data.Volume, data.Direction, data.OffsetFlag))

        if offset_flag == lf.LfOffsetFlagType.Open:
            self.investor_position_ctp[ticker][direction] += volume
        else:
            oppsite_direction = str(1 - int(direction))
            self.investor_position_ctp[ticker][oppsite_direction] -= volume
            if offset_flag == lf.LfOffsetFlagType.CloseYesterday:
                self.investor_position_ctp[ticker][oppsite_direction + '_Yd'] -= volume

        for i in self.trade_queue:
            if self.trade_queue[i]['request_id'] == request_id:
                order_dict = self.trade_queue[i]
                order_dict['volume'] -= volume
                trigger_price = order_dict['trigger_price']
                if order_dict['volume'] == 0:
                    order_dict['deal'] = 'Yes'
                    s_name = order_dict['signal_name']
                    if order_dict['action'] == 'open':
                        print self.strategies[s_name], 'Open deal = Yes'
                        self.strategies[s_name].tranx_order['Open deal'] = 'Yes'
                        self.strategies[s_name].tranx_order['Open price'] = price
                        self.strategies[s_name].tranx_order['Open time'] = self.last_tick['trading_day'] + ' ' + data.TradeTime
                        self.strategies[s_name].tranx_order['Close deal'] = 'No'
                    elif order_dict['action'] == 'close':
                        print self.strategies[s_name], 'Close deal = Yes'
                        self.strategies[s_name].tranx_order['Close deal'] = 'Yes'
                        self.strategies[s_name].tranx_order['Close price'] = price
                        self.strategies[s_name].tranx_order['Close time'] = self.last_tick['trading_day'] +  ' ' + data.TradeTime

        self.info(self.instrument_id + ' ' + str(self.investor_position_ctp[self.instrument_id]))
        self.tranx_record_dict[request_id]['position'] = str(self.investor_position_ctp[self.instrument_id])
        self.tranx_record_dict[request_id]['TradeTime'] += '[' + data.TradingDay + ' ' + data.TradeTime + ']'
        self.tranx_record_dict[request_id]['Volume'] += '[' + str(volume) + ']'
        self.tranx_record_dict[request_id]['TradeCost'] += '[' + str(price) + ']'
        self.tranx_record_dict[request_id]['Amount'] += '[' + str(amount) + ']'
        self.tranx_record_dict[request_id]['trigger_price'] = trigger_price
        self.tranx_record_dict[request_id]['trigger_time'] = order_dict['trigger_time']
        self.tranx_record_dict[request_id]['slippage'] += volume * (price - trigger_price) * (1 if int(direction) == 0 else -1)
        self.tranx_record_dict[request_id]['script'] = self.get_name()
        self.mongo_db_tranx_record.update_one({'mongo_id':self.tranx_record_dict[request_id]['mongo_id']}, {"$set":self.tranx_record_dict[request_id]}, upsert=True)
        self.debug('tranx_record has been inserted into mongoDB: TranxRecord.' + self.fund_name)
        return


    def on_order_err(self, data, error_id, error_msg, request_id, source, rcv_time):
        self.error('error founded!')
        ticker = data.InstrumentID
        self.order_suspended[ticker] = False
        self.error('[ORDER REJECTED] (rid){} (source){} (errId){} (errMsg){}'.format(request_id, source, error_id, error_msg))


    # 下单优化 交易中 ===========================================================================================================================================
    def order_dict_exec(self, order_dict, mode):
        # 执行单个offset flag方向的订单
        if self.last_tick['update_time'] in ['15:00:00', '14:59:59', '14:59:58', '14:59:57', '22:59:57', '22:59:58', '22:59:59', '23:00:00', '23:29:59', '23:29:58', '23:30:30']:
            mode = 'market'
        offset_flag = order_dict['offset_flag']
        volume = order_dict['volume']
        direction = '0' if order_dict['direction']>0 else '1'
        if mode == 'limit':
            limit_price = self.last_tick['last_price'] - self.one_jump * (1 if direction == '0' else -1)
            order_dict['place_price'] = limit_price
            order_dict['request_id'] = self.limit_order_exec(direction, self.offset_str_to_flag_dict[offset_flag], limit_price, volume)
        elif mode == 'market':
            order_dict['request_id'] = self.market_order_exec(direction, self.offset_str_to_flag_dict[offset_flag], volume)
        else:
            raise Exception('catch an exception case in Rules_101_single_order_exec()')
        order_dict['OrderCanceled'] = 'N'
        #self.error('order_dict_exec!')



    def limit_order_exec(self, direction, offset, limit_price, volume):
        request_id = self.insert_order(source=self.get_source(), ticker=self.instrument_id, volume=volume, direction=direction, offset=offset, price_type=lf.LfOrderPriceTypeType.LimitPrice, limit_price=limit_price, time_condition=lf.LfTimeConditionType.GFD)
        self.color_log('info', 'red', '[INSERT] [LIMIT] ' + ' ' + self.instrument_id + ' direct:' + str(direction) + ' offset:' + str(offset) + ' volume:' + str(
                volume) + ' limit_price:' + str(limit_price) + ' current price:' + str(self.last_tick['last_price']) + ' request_id:' + str(request_id))
        return request_id

    def market_order_exec(self, direction, offset, volume):
        request_id = self.insert_order(source=self.get_source(), ticker=self.instrument_id, volume=volume, direction=direction, offset=offset, price_type=lf.LfOrderPriceTypeType.AnyPrice, time_condition=lf.LfTimeConditionType.GFD)
        self.color_log('info', 'red', '[INSERT] [MARKET] ' + self.instrument_id + ' direct:' + str(direction) + ' offset:' + str(offset) + ' volume:' + str(
        volume) + ' current price:' + str(self.last_tick['last_price']) + ' request_id:' + str(request_id))
        return request_id

    def cancel_order_exec(self, request_id):
        self.color_log('info', 'r', '[INSERT] [Cancel] ' + ' request_id:' + str(request_id) + ' ')
        self.cancel_order(self.get_source(), request_id)


    # ----------------
    # basic utililites
    # ----------------
    def on_data(self, frame, msg_type, request_id, source, rcv_time):
        if msg_type == lf.MsgTypes.MD:
            data = lf_utils.dcast_frame(frame)
            self.on_market_data_level1(data, source, rcv_time)
        elif msg_type == lf.MsgTypes.RTN_ORDER:
            data = lf_utils.dcast_frame(frame)
            self.on_rtn_order(data, request_id, source, rcv_time)
        elif msg_type == lf.MsgTypes.RTN_TRADE:
            data = lf_utils.dcast_frame(frame)
            self.on_rtn_trade(data, request_id, source, rcv_time)
        elif msg_type == lf.MsgTypes.ORDER:
            if frame.error_id() != 0:
                # our order is blocked...
                data = lf_utils.dcast_frame(frame)
                self.on_order_err(data, frame.error_id(), frame.error_msg(), request_id, source, rcv_time)

    def on_investor_position(self, wc_positions, source):
        self.print_position(wc_positions, source)
        for ticker in self.tickers:
            self.cur_positions[ticker] = wc_positions.get_long_tot(ticker)
        self.info('[pos] updated to :' + ', '.join(map(lambda x:x[0] + ':' + str(x[1]), self.cur_positions.items())))
        self.is_running = True
        self.set_pos_map(wc_positions, source)
        self.investor_position_ctp = {self.instrument_id:{'0': int(self.get_pos_long_tot(self.instrument_id, self.get_source())),
                                    '1': int(self.get_pos_short_tot(self.instrument_id, self.get_source())),
                                    '0_Yd': int(self.get_pos_long_yd(self.instrument_id, self.get_source())),
                                    '1_Yd': int(self.get_pos_short_yd(self.instrument_id, self.get_source()))}}
        self.info('[position] ' + str(self.investor_position_ctp))
        self.position_get = True


    def insert_order(self, source, ticker, volume, direction, offset,
                     price_type=lf.LfOrderPriceTypeType.AnyPrice,
                     time_condition=lf.LfTimeConditionType.IOC,
                     limit_price=0,
                     min_volume=1,
                     volume_condition=lf.LfVolumeConditionType.AV,
                     force_close_reason=lf.LfForceCloseReasonType.NotForceClose,
                     contingent_condition=lf.LfContingentConditionType.Immediately,
                     hedge_flag=lf.LfHedgeFlagType.Speculation):
        order = lf_structs.LFInputOrderField()
        order.InstrumentID = ticker
        order.Volume = volume
        order.TimeCondition = time_condition
        order.VolumeCondition = volume_condition
        order.LimitPrice = limit_price
        order.Direction = direction
        order.OffsetFlag = offset
        order.OrderPriceType = price_type
        order.ForceCloseReason = force_close_reason
        order.ContingentCondition = contingent_condition
        order.HedgeFlag = hedge_flag
        order.MinVolume = min_volume

        rid = self.get_rid()
        self.req_insert_order(ctypes.addressof(order), rid, source)
        return rid

    def cancel_order(self, source, order_rid):
        rid = self.get_rid()
        self.req_cancel_order(source, order_rid, rid)
        return rid

    def req_position(self, source):
        rid = bl.get_rid()
        bl.req_investor_position(rid, source)
        return rid

    def subscribe(self, source, tickers, market=None):
        markets = []
        if market is None:
            markets = map(lambda x:'', tickers)
        elif isinstance(market, basestring):
            markets = map(lambda x:market, tickers)
        elif isinstance(market, list):
            markets = market
        self.req_subscribe_market_data(tickers, markets, source)

    def in_trading_time(self, tm_str):
        for (start, end) in self.get_trading_time():
            if tm_str >= start and tm_str <= end:
                return True
        return False

    def get_trading_time(self):
        # 品种交易时间段
        if self.instrument_id[:2] == 'rb':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:00:00')
            ]
            self.market_order_mode = 'pseudo'
            self.exchange_id = 'SHME'
            self.lock_mode = True

        elif self.instrument_id[0] == 'i':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:30:00')   # cautious!
            ]
            self.market_order_mode = 'true'
            self.exchange_id = 'DCE'
            self.lock_mode = False

        elif self.instrument_id[:2] == 'hc':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:00:00')
            ]
            self.market_order_mode = 'pseudo'
            self.exchange_id = 'SHME'
            self.lock_mode = True

        elif self.instrument_id[:2] == 'ZC':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:30:00')   # cautious!
            ]
            self.market_order_mode = 'true'
            self.exchange_id = 'CZCE'
            self.lock_mode = False

        elif self.instrument_id[:2] == 'SM':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
            ]
            self.market_order_mode = 'true'
            self.exchange_id = 'CZCE'
            self.lock_mode = True

        elif self.instrument_id[0] == 'v':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
            ]
            self.market_order_mode = 'true'
            self.exchange_id = 'DCE'
            self.lock_mode = False

        elif self.instrument_id[:2] == 'cs':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
            ]
            self.market_order_mode = 'true'
            self.exchange_id = 'DCE'
            self.lock_mode = False
        elif self.instrument_id[:2] == 'cu':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '24:00:00'),
                ('00:00:00', '01:00:00')]
            self.exchange_id = 'SHME'
            self.lock_mode = True 
        else:
            raise Exception('no matched instrument_id')
        return self.schedule

    def get_source(self):
        return lf.SOURCE.CTP

    def get_tds(self):
        return [self.get_source()]

    def get_l1_mds(self):
        return [self.get_source()]


    def color_log(self, log_type, color, log_str):
        print 'color_log'
        color_end = '\033[0m'
        color_start = '\033[1;37;40m'  #default white
        if color.lower() == 'red' or color.lower() == 'r':
            color_start = '\033[1;31;40m'
        elif color.lower() == 'green' or color.lower() == 'g':
            color_start = '\033[1;32;40m'
        elif color.lower() == 'golden' or color.lower() == 'gold' or color.lower() == 'yellow' or color.lower() == 'y':
            color_start = '\033[1;33;40m'
        elif color.lower() == 'blue' or color.lower() == 'b':
            color_start = '\033[1;34;40m'
        elif color.lower() == 'purple' or color.lower() == 'p':
            color_start = '\033[1;35;40m'
        if log_type == 'info':
            self.info(color_start + log_str + color_end)
        elif log_type == 'debug':
            self.debug(color_start + log_str + color_end)
        else:
            raise Exception('Wrong color_log() usage!')


if __name__ == '__main__':
    bl = FirstStrategy()
    bl.init()
    bl.run()
