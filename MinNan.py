# coding=utf-8
import numpy as np
import pandas as pd 
from pandas import Timedelta
import datetime
import time, sys
import json, yaml, os
from collections import deque
from Nan import Nan

# import sys 
# reload(sys) 
# sys.setdefaultencoding('utf-8')

# 采集时间 
global execution_time_record_dict
execution_time_record_dict = {}

def execution_time_record(func):
    def wrapper(*args, **kw):
        t0 = pd.to_datetime(datetime.datetime.now())
        result = func(*args, **kw)
        #t1 = time.time()
        t1 = pd.to_datetime(datetime.datetime.now())
        delta = str(t1-t0)
        global execution_time_record_dict 
        if func.func_name not in execution_time_record_dict.keys():
            execution_time_record_dict[func.func_name] = [delta]
        else:
            execution_time_record_dict[func.func_name].append(delta)
        return result
    return wrapper


class MinNan(Nan):
    def __init__(self, signal_name, base_path, logger, config_dict):
        self.logger = logger
        self.signal_name = signal_name
        self.base_path = base_path
        self.config = config_dict
        self.last_tick_volume = 0
        self.load_signal()
        self.base_config_prepare()
        self.shipan_config_prepare()
        self.get_market_time_schedule()
        self.robust_test_mode = False  #must False

    def load_signal(self):
        import imp
        if self.signal_name + '.py' in os.listdir(self.base_path):
            signal_file = os.path.join(self.base_path, self.signal_name + '.py')
            import compileall   
            compileall.compile_file(signal_file)
            signal_class = imp.load_source('signal_class', signal_file)
            os.remove(signal_file)
        elif self.signal_name + '.pyc' in os.listdir(self.base_path):
            signal_file = os.path.join(self.base_path, self.signal_name + '.pyc')
            signal_class = imp.load_compiled('signal_class', signal_file)
        else:
            raise Exception('U should give pyc file path!' + signal_file)
        self.signal = signal_class.signal()


    def shipan_config_prepare(self):
        if 'min' in self.time_frame:    
            self.min = int(self.time_frame.split('min')[0])
        self.slippoint = 0
        self.commission = 0
        self.stamp = 0


    def data_prepare(self, history_data_file):
        self.history_data_df = pd.read_csv(history_data_file,
            names=['dates', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'ClosePrice', 'Volume', 'Position'])
        # if end is not None:
        #     end = pd.to_datetime(end)
        #     index_ = np.where(pd.to_datetime(self.history_data_df['dates'].values) < end)[0]
        #     self.history_data_df = self.history_data_df.iloc[index_, :]
            #print self.history_data_df.tail()
            #print self.signal_name, self.history_data_df.shape
        if self.history_data_df.shape[0] < self.min * (self.maxlookback):
            print_info = "%s data length doesn't meet %s" % (self.config["history_data_file"], self.min * (self.maxlookback))
            raise Exception(print_info)
        else:
            self.history_data_df = self.history_data_df.iloc[self.min * (-self.maxlookback - 300):]

        # create min data
        self.future_data_dict = {}
        self.future_data_dict['Close'], self.future_data_dict['Open'], \
        self.future_data_dict['Low'], self.future_data_dict['High'], \
        self.future_data_dict['Init'], self.future_data_dict['Volume'], \
        self.future_data_dict['Update'] = [], [], [], [], [], [], []#deque(maxlen=10000), deque(maxlen=10000), deque(maxlen=10000), deque(maxlen=10000), deque(maxlen=10000), deque(maxlen=10000), deque(maxlen=10000)
        min = self.min
        moving_min = 0 
        for i in range(self.history_data_df.shape[0] - (min - 1)):
            date = self.history_data_df["dates"].values[i]
            if int(date[-2:]) % min == moving_min:
                self.future_data_dict['Close'].append(self.history_data_df["ClosePrice"].values[i + (min -1)])
                self.future_data_dict['Open'].append(self.history_data_df["OpenPrice"].values[i])
                self.future_data_dict['High'].append(
                    np.max(self.history_data_df["HighestPrice"].values[i:i + min]))
                self.future_data_dict['Low'].append(
                    np.min(self.history_data_df["LowestPrice"][i:i + min]))
                self.future_data_dict['Volume'].append(
                    np.sum(self.history_data_df["Volume"].values[i:i + min]))
                self.future_data_dict['Init'].append(
                    str(pd.to_datetime(self.history_data_df["dates"].values[i] + ':00')))
                self.future_data_dict['Update'].append(
                    str(pd.to_datetime(self.history_data_df["dates"].values[i + (min - 1)] + ':59')))

        for i in range(self.history_data_df.shape[0] - (min - 1), self.history_data_df.shape[0]):
            date = self.history_data_df["dates"].values[i]
            if int(date[-2:]) % min == moving_min:
                self.future_data_dict['Close'].append(self.history_data_df["ClosePrice"].values[-1])
                self.future_data_dict['Open'].append(self.history_data_df["OpenPrice"].values[i])
                self.future_data_dict['High'].append(
                    np.max(self.history_data_df["HighestPrice"].values[i:]))
                self.future_data_dict['Low'].append(np.min(self.history_data_df["LowestPrice"][i:]))
                self.future_data_dict['Volume'].append(
                    np.sum(self.history_data_df["Volume"].values[i:]))
                self.future_data_dict['Init'].append(
                    str(pd.to_datetime(self.history_data_df["dates"].values[i] + ':00')))
                self.future_data_dict['Update'].append(
                    str(pd.to_datetime(self.history_data_df["dates"].values[-1] + ':59')))


    def signal_info_screen(self, print_signal_info):
        # 交易前 显示信号信息
        direct = {1: "Long Only", 0: "Long & Short", -1: "Short only"}
        tmp = np.column_stack([self.cfg["Name"], self.cfg["Benchmark"], self.maxlookback,
                               self.raw_data_start, self.raw_data_end, self.cfg['Data']['DailyBarNum'],
                               self.cfg['Tcost']['SlipPoint'], self.cfg['Tcost']['Commission'], self.cfg['Tcost']['Stamp'], self.pointValue, self.priceScale
                               ])
        self.info_df = pd.DataFrame(tmp, index=[""], columns=np.array(["SignalName", "InstrumentID", "MaxLookback","PointValue", "PriceScale"]))
        if print_signal_info is False: return
        print self.info_df.loc[:,["SignalName", "InstrumentID", "MaxLookback", 'PointValue', 'PriceScale']].T



    def prebatch(self, ClosePrice, OpenPrice, HighestPrice, LowestPrice, Volume, dates, update_dates):
        self.ClosePrice = ClosePrice * 1.
        self.OpenPrice = OpenPrice * 1.
        self.HighestPrice = HighestPrice * 1.
        self.LowestPrice = LowestPrice * 1.
        self.Volume = Volume * 1.
        self.dates = dates
        self.update_dates = update_dates

        self.signal.ClosePrice = self.ClosePrice.copy() 
        self.signal.OpenPrice = self.OpenPrice.copy()
        self.signal.HighestPrice = self.HighestPrice.copy()
        self.signal.LowestPrice = self.LowestPrice.copy()
        self.signal.dates = self.dates.copy()
        self.signal.Volume = self.Volume.copy()



    def warm_signal(self, lookback=50):
        self.tranx_order = {}
        for i in range(lookback):
            ClosePrice = np.array(self.future_data_dict['Close'][-self.maxlookback + (i-lookback):i-lookback]) 
            OpenPrice=np.array(self.future_data_dict['Open'][-self.maxlookback + (i-lookback):i-lookback]) 
            HighestPrice=np.array(self.future_data_dict['High'][-self.maxlookback + (i-lookback):i-lookback]) 
            LowestPrice=np.array(self.future_data_dict['Low'][-self.maxlookback + (i-lookback):i-lookback]) 
            Volume=np.array(self.future_data_dict['Volume'][-self.maxlookback + (i-lookback):i-lookback]) 
            dates =  np.array(self.future_data_dict['Init'][-self.maxlookback + (i-lookback):i-lookback])
            update = np.array(self.future_data_dict['Update'][-self.maxlookback + (i-lookback):i-lookback])

            self.prebatch(ClosePrice, OpenPrice, HighestPrice, LowestPrice, Volume, dates, update)
            self.last_bar_init_time = dates[-1]
            self.signal.prebatch()
            self.signal.generate(len(ClosePrice), self)
            # self.logger.info(self.signal_name + ' date ' + str(dates[-1]) + ' update ' + str(update[-1]) +\
            # " C: " + str(ClosePrice[-1]) + ' O: ' + str(OpenPrice[-1]) +' H: ' + str(HighestPrice[-1]) + " L: " + str(LowestPrice[-1]) + " V: " + str(Volume[-1]))
            # print str(self.tranx_order)



    def work(self, tick_time_list, last_tick):
        # 回测逻辑和实盘逻辑一致
        # 实盘逻辑 (过程态 + 结果态)
        #                 di -1                                di                               di + 1
        # |----------------------------------|-----------------------------------|--------------------------------|
        #                              check order di-1                       check order di
        #                              clear di-1                             clear di
        #                              roll for di                            roll for di + 1
        #            trade_order                          trade_order                         trade_order 
        #                              process_order                        process_order
        # 上一个bar结束, 交易系统返回上一个bar的交易成功状态, 接着清算上一个bar
        # 下一个bar开始前, 策略会被运行, 止盈止损线和平仓指令会被更新, 订单信息会发出
        # 简之, check_order --> clear --> roll --> 订单信息发出 --> 交易 -->处理已成交订单
        # 在固定时间点上, 保存订单记录
        # 对于时间, 举例: 59的第一个tick就会触发下面条件, 但是对于trade_order里面的
        if last_tick['update_time'] in self.signal.trigger_time_list and tick_time_list[-2] != tick_time_list[-1]:
            #print 'prebatch...'
            self.prebatch(np.array(self.future_data_dict['Close']), np.array(self.future_data_dict['Open']), np.array(self.future_data_dict['High']),
                            np.array(self.future_data_dict['Low']), np.array(self.future_data_dict['Volume']), np.array(self.future_data_dict['Init']), np.array(self.future_data_dict['Update']))
            di = len(self.ClosePrice)
            self.last_bar_init_time = self.dates[di-1]
            self.check_order(di-1)     #结果态
            #print 'clear...'
            self.clear(di-1)           #结果态
            #print 'signal.prebatch()'
            self.signal.prebatch()
            #print 'date', self.dates[di-1]
            self.roll(di)              #结果态
            #print 'roll over', str(self.tranx_order)

        if (tick_time_list[-1][-2:] == '30' and tick_time_list[-2][-2:] == '29') or tick_time_list[-1] == '15:00:00':
           self.save_everything()
            #self.save_time_record()


    def trade_order(self, tick_time_list, tick_price_list, last_tick, di=-1):
        # 过程态, 对应回测系统的process_order的一部分
        # 判断逻辑:订单动作和订单状态, 这两个条件
        # 交易系统实时调用此函数, 来监控目前tick价格是否满足开仓平仓条件, 一旦订单发出, 所在的bar的期限内, monitor_order就会关闭, 直到下一个bar更新self.tranx_order
        if len(self.tranx_order) >0:
            # 未持仓
            if self.tranx_order['Open deal'] == 'No' and self.tranx_order['Open exec'] == 'No':
                #限价单
                if self.tranx_order['Open type'] == 'Stop' and self.tranx_order['Target Open price'] == last_tick['last_price']:
                    self.tranx_order['Target Open time'] = self.update_dates[di]
                    self.tranx_order['Open exec'] = 'Yes'
                    self.tranx_order['Trigger Open time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Open price'] = last_tick['last_price']
                    return {'size': self.Size, 'direction': self.tranx_order['direction'], 'action': 'open', 'Target Open price': self.tranx_order['Target Open price'], 'comment': 'Open using Stop'}
                # 收盘市价单 or 开盘市价单
                elif (self.tranx_order['Open type'] == 'OpenPrice' or self.tranx_order['Open type'] =='ClosePrice') and last_tick['update_time'][-8:] in self.signal.trigger_time_list:
                    self.tranx_order['Target Open price'] = self.ClosePrice[di]
                    self.tranx_order['Target Open time'] = self.update_dates[di]
                    self.tranx_order['Trigger Open price'] = last_tick['last_price']
                    self.tranx_order['Trigger Open time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Open exec'] = 'Yes'
                    return {'size': self.Size, 'direction': self.tranx_order['direction'], 'action': 'open', 'comment': 'Open using ClosePrice or OpenPrice'}
                else:
                    return {}

            # 已持仓
            elif self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Close deal'] == 'No' and self.tranx_order['Close exec'] == 'No':
                # 平仓的优先级: 开盘价 > 反向开单 > 止损 > 止盈
                # =======================================
                # Exit 市价平仓使用开盘价
                if 'Close type' in self.tranx_order.keys() and (self.tranx_order['Close type'] == 'OpenPrice' or self.tranx_order['Close type'] == 'ClosePrice') and last_tick['update_time'][-8:] in self.signal.trigger_time_list and tick_time_list[-1] != tick_time_list[-2]:
                    self.tranx_order['Comment'] = 'Exit'
                    self.tranx_order['Target Close price'] =  self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.update_dates[di]
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Close using OpenPrice'}

                # =======================================
                # 反向开单, 直接开双倍订单, 市价平仓使用开盘价, 反向开仓也使用开盘价
                elif 'Reverse type' in self.tranx_order.keys() and self.tranx_order['Reverse type'] == 'OpenPrice' and last_tick['update_time'][-8:] in self.signal.trigger_time_list and tick_time_list[-1] != tick_time_list[-2]:
                    self.tranx_order['Target Close price'] =  self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.update_dates[di]
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Comment'] = 'Reverse'
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Reverse using OpenPrice'}

                # =======================================
                # 止盈止损
                # 判断订单的多头止损
                if self.tranx_order['LongSL'] == last_tick['last_price']:
                    self.tranx_order['Comment'] = 'Long SL'
                    self.tranx_order['Target Close price'] =  self.tranx_order['LongSL']
                    self.tranx_order['Target Close time'] = self.update_dates[di]
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Long StopLoss'}
                elif last_tick['update_time'][-8:] in self.signal.trigger_time_list and tick_time_list[-1] != tick_time_list[-2] and self.tranx_order['LongSL'] > self.HighestPrice[di]:
                    # 跳空的情况, 这根bar结束了, 才知道是否跳空
                    self.tranx_order['Comment'] = 'Long Gap SL'
                    self.tranx_order['Target Close price'] =  self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.update_dates[di]
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Long Gap StopLoss'}
                     
                # 判断订单的空头止损
                if self.tranx_order['ShortSL'] == last_tick['last_price']:
                    self.tranx_order['Comment'] = 'Short SL'
                    self.tranx_order['Target Close price'] = self.tranx_order['ShortSL']
                    self.tranx_order['Target Close time'] = self.update_dates[di]
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Short StopLoss'}
                elif last_tick['update_time'][-8:] in self.signal.trigger_time_list and self.tranx_order['ShortSL'] > 0 and self.tranx_order['ShortSL'] < self.LowestPrice[di]:
                    # 跳空
                    self.tranx_order['Comment'] = 'Short Gap SL'
                    self.tranx_order['Target Close price'] = self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.update_dates[di]  
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Short Gap StopLoss'}

                # 判断订单的多头止盈
                if self.tranx_order['LongPT'] == last_tick['last_price']:
                    self.tranx_order['Comment'] = 'Long PT'
                    self.tranx_order['Target Close price'] = self.tranx_order['LongPT']
                    self.tranx_order['Target Close time'] = self.update_dates[di]  
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Long ProfitTarget'}
                elif last_tick['update_time'][-8:] in self.signal.trigger_time_list and self.tranx_order['LongPT'] > 0 and self.tranx_order['LongPT'] < self.LowestPrice[di]:
                    # 跳空
                    self.tranx_order['Comment'] = 'Long Gap PT'
                    self.tranx_order['Target Close price'] = self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.update_dates[di]
                    self.tranx_order['Trigger Close time'] = last_tick['trading_day'] + ' ' + last_tick['update_time']
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Comment'] = 'Gap PT'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Long Gap ProfitTarget'}

                # 判断订单的空头止盈
                if self.tranx_order['ShortPT'] == last_tick['last_price']:
                    self.tranx_order['Comment'] = 'Short PT'
                    self.tranx_order['Target Close price'] = self.tranx_order['ShortPT']
                    self.tranx_order['Target Close time'] = self.update_dates[di]  
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] == 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Short ProfitTarget'}
                elif last_tick['update_time'][-8:] in self.signal.trigger_time_list and self.tranx_order['ShortPT'] > self.HighestPrice[di]:
                    # 跳空
                    self.tranx_order['Comment'] = 'Short Gap PT'
                    self.tranx_order['Target Close price'] = self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.update_dates[di]
                    self.tranx_order['Trigger Close price'] = last_tick['last_price']
                    self.tranx_order['Close exec'] == 'Yes'
                    return {'size': self.Size, 'direction': -1 * self.tranx_order['direction'], 'action': 'close', 'comment': 'Short Gap ProfitTarget'}
                return {}

            # 已平仓
            elif self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Close deal'] == 'Yes' and self.tranx_order['Close exec'] == 'Yes':
                return {}
             
            # 等待成交   
            else:
                return {}
        else:
            return {}



    # ====================================================================================================================================================================
    # 以下处理数据
    # ====================================================================================================================================================================
    def save_everything(self):
        df = pd.DataFrame()
        df['Init'] = self.future_data_dict['Init']
        df['Update'] = self.future_data_dict['Update']
        df['Open'] = self.future_data_dict['Open']
        df['High'] = self.future_data_dict['High']
        df['Low'] = self.future_data_dict['Low']
        df['Close'] = self.future_data_dict['Close']
        df['Volume'] = self.future_data_dict['Volume']
        if len(os.path.split(self.config['save_data_dir'])[0]) > 1 and not os.path.exists(os.path.split(self.config['save_data_dir'])[0]):
            os.mkdir(os.path.split(self.config['save_data_dir'])[0])
        if not os.path.exists(self.config['save_data_dir']):
            os.mkdir(self.config['save_data_dir'])
        df.to_csv(os.path.join(self.config['save_data_dir'], self.signal_name + '_' + self.instrument_id + '.csv'))
            #print 'data has been saved to .csv!', os.path.join(self.config['save_data_dir'], self.config["data_type"] + '.csv')
        self.tranx_record_df.to_csv(os.path.join(self.config['tranx_record_dir'], self.signal_name + '_' + self.instrument_id + '.csv'))
        #print self.signal_name, ' tranx_record has been saved'

    def save_time_record(self):
        # 保存运行时间
        global execution_time_record_dict
        json_path = os.path.join(self.config["program_record_dir"], self.signal_name + '_' + self.instrument_id +'_time_record.json')
        with open(json_path, 'w') as f:
            d = json.dumps(execution_time_record_dict)
            f.write(d)


    def update_kline(self, tick_time_list, last_tick):
        date = last_tick['trading_day']
        update_time = date + ' ' + last_tick['update_time']
        if tick_time_list[-1] >= '21:00:00' and tick_time_list[-2]<'21:00:00':
            self.last_tick_volume = 0
            self.logger('[new tradeday] ' + last_tick['update_time'] + '== 21:00:00, begin new tradeday, self.last_tick_volume init to 0')

        if str(pd.to_datetime(last_tick['update_time']) - Timedelta('0 days 00:00:01'))[-8:] in self.signal.trigger_time_list and tick_time_list[-1] != tick_time_list[-2]:
            self.future_data_dict['Close'].append(last_tick['last_price'])
            self.future_data_dict['High'].append(last_tick['last_price'])
            self.future_data_dict['Low'].append(last_tick['last_price'])
            self.future_data_dict['Open'].append(last_tick['last_price'])
            self.future_data_dict['Init'].append(update_time)
            self.future_data_dict['Update'].append(update_time)
            self.future_data_dict['Volume'].append(last_tick['volume'] - self.last_tick_volume)

        self.future_data_dict['Close'][-1] = last_tick['last_price']
        self.future_data_dict['High'][-1] = last_tick['last_price'] if self.future_data_dict['High'][-1] < last_tick[
            'last_price'] else self.future_data_dict['High'][-1]
        self.future_data_dict['Low'][-1] = last_tick['last_price'] if self.future_data_dict['Low'][-1] > last_tick[
            'last_price'] else self.future_data_dict['Low'][-1]
        self.future_data_dict['Update'][-1] = update_time
        vol_chg = last_tick['volume'] - self.last_tick_volume
        self.future_data_dict['Volume'][-1] = self.future_data_dict['Volume'][-1] + vol_chg
        self.last_tick_volume = last_tick['volume']


    def unit_keywords(self, data, mode='simple'):
        CTP_keywords = {'UpdateTime':'update_time', 
                    'AskPrice1':'ask_price1',
                    'AskVolume1': 'ask_volume1',
                    'BidPrice1':'bid_price1',
                    'BidVolume1':'bid_volume1',
                    'InstrumentID': 'instrument_id',
                    'LastPrice':'last_price',
                    'OpenInterest':'open_interest',
                    'Volume':'volume',
                    'TurnOver':'turnover',
                    'ExchangeID':'exchange_id',
                    'ExchangeInstID':'exchange_inst_id',
                    'nano_time':'nano_time'}

        SHFE_keywords = {'update_time':'update_time', 
            'ask_price1':'ask_price1',
            'ask_volume1': 'ask_volume1',
            'bid_price1':'bid_price1',
            'bid_volume1':'bid_volume1',
            'instrument_id': 'instrument_id',
            'last_price':'last_price',
            'open_interest':'open_interest',
            'volume':'volume',
            'turnover':'turnover',
            'exchange_id':'exchange_id',
            'exchange_inst_id':'exchange_inst_id',
            'nano_time':'nano_time'}

        keywords = {}
        keywords.update(CTP_keywords)
        keywords.update(SHFE_keywords)

        res = {}
        keywords_keys = keywords.keys()
        if mode is 'all':
            for i in data:
                if i in keywords_keys:
                    res[keywords[i]] = data[i]
                else:
                    res[i] = data[i]
        elif mode is 'simple':
            for i in data:
                if i in keywords_keys:
                    res[keywords[i]] = data[i]
        return  res 


    def get_subscribed_msg_types(self):
        msg_type_list = [
            ninja.MsgType.MSG_CTP_L1MD, 
        ]
        return msg_type_list


    def get_market_time_schedule(self):
        # for RB only
        if self.instrument_id[:2] == 'rb':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:00:00')
            ]
        elif self.instrument_id[0] == 'i':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:30:00')   # cautious!
            ]
        elif self.instrument_id[:2] == 'hc':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:00:00')
            ]
        elif self.instrument_id[:2] == 'ZC':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
                ('21:00:00', '23:30:00')   # cautious!
            ]
        elif self.instrument_id[:2] == 'SM':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
            ]
        elif self.instrument_id[0] == 'v':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
            ]
        elif self.instrument_id[:2] == 'cs':
            self.schedule = [
                ('09:00:00', '10:15:00'),
                ('10:30:00', '11:30:00'),
                ('13:30:00', '15:00:00'),
            ]
        return self.schedule


    def is_in_schedule(self, time_str):
        for (start, end) in self.schedule:
            if time_str <= end and time_str >= start:
                return 1
            else:
                continue
        return 0



    # def load_reader_group_min(self, yumi):
    #     tick_time_record = []
    #     last_time_str = self.future_data_dict['Update'][-1]
    #     cur_date = last_time_str.split(' ')[0]
    #     last_time_str = str(pd.to_datetime(last_time_str) + Timedelta('0 days 00:01:00'))
    #     last_time = int(time.mktime(time.strptime(last_time_str, '%Y-%m-%d %H:%M:%S')) * 1e9)

    #     self.info(self.signal_name + ' loading history data form reader... start from '+ last_time_str)

    #     yumi.init_reader(last_time)
    #     frame = yumi.reader.getNextEntry()
    #     first = 0
    #     while frame is not None:
    #         msg_type = frame.msg_type()

    #         if msg_type == ninja.MsgType.MSG_TIME_STAMP:
    #             yumi.time_info = frame.get_data()

    #         if msg_type in self.get_subscribed_msg_types():
    #             data = frame.get_data()
    #             data = self.unit_keywords(data)
    #             if self.is_in_schedule(data['update_time']) and data['instrument_id'] == self.instrument_id:
    #                 #print data['update_time']
    #                 if first == 0:
    #                     first = 1
    #                     tick_time_record.append(data['update_time'])
    #                     date = datetime.datetime.fromtimestamp(data['nano_time'] / 1000000000).strftime('%Y-%m-%d')
    #                     self.logger.info("first reader data start from %s" % (date + ' ' + data['update_time']))
    #                     # if pd.to_datetime(date + ' ' + data['update_time']) - pd.to_datetime(last_time_str) \
    #                     #       > Timedelta('0 days 00:01:00'):
    #                     #   raise Exception('history data end with %s' % last_time_str)
    #                 if first == 1:
    #                     tick_time_record.append(data['update_time'])
    #                     self.update_kline(tick_time_record, data)
    #                 #df = pd.DataFrame(self.future_data_dict[0])
    #                 #df.to_csv('nanhua.csv', index=False)
    #                 #self.on_market_data_recv(ninja.MsgType.MSG_NH_L1MD, data)

    #         frame = yumi.reader.getNextEntry()
    #     #print 'data update to', data['update_time']
    #     self.logger.info('tick data updated successfully!')



    # def load_tick(self, yumi, mode='db'):
    #     if mode == 'db':
    #         self.load_mongodb_group_min()
    #         self.load_reader_group_min(yumi)
    #     elif mode == 'yumi':
    #         self.load_reader_group_min(yumi)

    # def load_mongodb_group_min(self):
    #     last_time_str = str(self.future_data_dict['Update'][-1])
    #     self.logger.info(self.signal_name + ' loading history data form mongoDB... start from ' + last_time_str)
    #     from pymongo import MongoClient
    #     from pymongo import DESCENDING
    #     client = MongoClient('127.0.0.1', 27017)
    #     db_name = 'FUTURE'
    #     db = client[db_name]
    #     # print db_name, self.instrument_id
    #     cur = db[self.instrument_id].find({"date": {'$gt':last_time_str}}).sort("date") 
    #     count = 0
    #     for data in cur:
    #         self.update_kline(data)