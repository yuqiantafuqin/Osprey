# -*- coding:utf-8 -*-
# author: xhcao
import numpy as np 
import pandas as pd 
import sys, os
import operators 

'''
对于单个函数的问题, 请看单个函数的注释
名词解释
* pips 点位 对应到不同品种, price = pips * minMove(指最小变动), 目的是对于含有定点止损的策略, 更够不需要更改就适应不同minMove的品种, 外部调用可以使用 Pips()
* minMove 品种价格的最小变动
* priceScale 品种价格的精度 比如1, 0.1, 0.01等等

输出
* 统计结果 & 对应的绘图一 按照逐笔交易, 横坐标轴为交易次数
* 统计结果 & 对应的绘图二 按照回测时间, 横坐标为时间
（两者结果略有差别在于最后一笔交易的记录，不会影响其他回测结果。）
* 开平仓位置绘图, 调用trade_plot()

功能
* 鲁棒性测试
  优先级顺序：随机开始位置 > 随机历史价格 > 随机输入参数 > 跳过订单 > shuffle（重新排列） > 极端正收益剔除（负收益保留）
  为了方便进行参数鲁棒性测试， 所有的含参数的，供外部调用的函数都设计了鲁棒性开关
  结果以绘图的形式，设置参数参考函数注释

'''
print 'version 1.06 20170617 beta'

# =========================================
# 计算用的函数
def MAX(x, y):
    return x*(x>=y) + y*(x<y)

def MIN(x, y):
    return x*(x<=y) + y*(x>y)

def REF(x, period):
    if type(period) is int or type(period) is float or type(period) is long:
        return delay(x, period)
    elif type(period) is np.ndarray:
        index_ = np.arange(len(x))-np.nan_to_num(period)
        return x[index_.astype('int')]
    else:
        raise Exception(e)

def ABS(x):
    return np.abs(x)

def ATR(ClosePrice, HighestPrice, LowestPrice, N): 
    #NUMERIC N(14,2,360,1)
    TR = MAX(MAX((HighestPrice-LowestPrice),ABS(REF(ClosePrice,1)-HighestPrice)),ABS(REF(ClosePrice,1)-LowestPrice))
    import talib
    ATR = talib.MA(TR, N)
    return ATR

def delay(x, period):
    """
    delay() value of x d days ago
    """
    res = np.zeros(x.shape) * np.nan
    res[period:] = x[:-period]
    return res

def Rsquared(y):
    from scipy.stats import linregress
    """ Return R^2 where x and y are array-like."""
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2


# ==========================================
# 工具
# 记录运行时间 
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


class Nan():
    def __init__(self, signal):
        self.signal = signal
        print self.signal.signal_config['Name'] + ' is created!'
        self.robust_test_mode = False
        self.base_config_prepare()
        self.backtest_config_prepare()

    def base_config_prepare(self):
        self.cfg = self.signal.signal_config
        self.signal_name = self.cfg['Name']
        self.instrument_id = self.cfg['Benchmark']
        self.maxlookback = self.cfg['Macro']['Maxlookback']
        self.time_frame = self.cfg['Data']['TimeFrame']
        self.Size = int(self.cfg['Money Management']['Fixed Size'])
        if 'PointValue' in self.cfg['Data']:
            self.pointValue  = self.cfg['Data']['PointValue']
        fpath = os.path.dirname(os.path.realpath(__file__))
        instrument_info_df = pd.read_csv(os.path.join(fpath, 'future_instrument_info.csv'))   #品种信息
        self.instrument_info_dict = {}
        (m, n) = instrument_info_df.shape
        for i in range(m):
            instrument = instrument_info_df['Instrument'].values[i]
            self.instrument_info_dict[instrument] = {}
            for j in range(n):
                self.instrument_info_dict[instrument][instrument_info_df.columns.values[j]] = instrument_info_df.iloc[i, j]
        self.instrument = self.instrument_id.split('_')[0]
        for j in self.instrument_info_dict:
            if j in self.instrument_id:
                self.instrument = j
                self.pointValue = self.instrument_info_dict[j]['PointValue']   # 每个点多少$
                self.tickSize = self.instrument_info_dict[j]['MinMove']    # 最小变动单位(包含了价格精度)

        # 参数
        if getattr(self.signal, 'param_dict', None) is not None:
            for param in self.signal.param_dict:
                setattr(self.signal, param, self.signal.param_dict[param]['Default'])



    def backtest_config_prepare(self):
        self.daily_bar_num = int(self.cfg['Data']['DailyBarNum'])   #用于算年化Sharp, 但是每年的交易时间差距较大, 参考意义不大
        self.data_set = self.cfg['data_set']
        self.start = str(self.cfg['Macro']['Start'])
        self.end = str(self.cfg['Macro']['End'])
        self.capital = float(self.cfg['Macro']['Capital'])
        self.data_path = self.cfg['Data']['Inpath']

        if 'Tcost' in self.cfg and 'SlipPoint' in self.cfg['Tcost']:
            self.slippoint = float(self.cfg['Tcost']['SlipPoint'])
        if 'Tcost' in self.cfg and 'Commission' in self.cfg['Tcost']:
            self.commission = float(self.cfg['Tcost']['Commission'])
        else:
            for j in self.instrument_info_dict:
                if j in self.instrument_id:
                    self.instrument = j
                    self.slippoint = self.instrument_info_dict[j]['SlipPoint']   # 每笔滑点(滑多少个点位)
                    self.commission = self.instrument_info_dict[j]['Commission']    # 交易所佣金

        if 'Tcost' in self.cfg and 'Stamp' in self.cfg['Tcost']: raise Exception('Should delete Stamp')
        #print self.slippoint
        #raise Exception()

    def data_prepare(self, mode='mongo', start_point=0, print_signal_info=True):
        # mongo or csv
        if mode == 'mongo':
            if print_signal_info: print 'data source: mongoDB ... '
            self.mongodb_collection = 'data_' + self.instrument_id.split('_')[1]
            from pymongo import MongoClient
            client = MongoClient('192.168.1.99', 27017)
            db = client['future_data']
            cursor = db['data_all'].find({'instrument_id':self.instrument_id}, {'ClosePrice':1, "HighestPrice":1, "OpenPrice":1, "LowestPrice":1, "Volume":1, "dates":1})    
            mongo_dict = cursor.next()
            self.raw_df = pd.DataFrame()
            for k in mongo_dict:
                if k == '_id': continue
                self.raw_df[k] = mongo_dict[k]
            self.raw_df.index = self.raw_df['dates']
            raw_dates = self.raw_df.index.values
        elif mode == 'csv':
            print 'data source: csv ... '
            self.raw_df = pd.read_csv(self.data_path, names=['OpenPrice', 'HighestPrice', 'LowestPrice', 'ClosePrice', 'Volume', 'Position'], header=0)
            raw_dates = self.raw_df.index.values

        self.raw_data_start = raw_dates[0]
        self.raw_data_end = raw_dates[-1]
        if print_signal_info is True:
            print >> sys.stdout, '\nraw data date: ', self.raw_data_start, ' - ', self.raw_data_end
        if self.data_set == 'is':
            start = pd.to_datetime(self.start)
            end = pd.to_datetime(self.end)
            print end
            #raw_dates_parse = np.array([dt_parser.parse(i) for i in raw_dates])
            raw_dates_index = pd.to_datetime(raw_dates.astype(str))
            if start < pd.to_datetime(raw_dates[self.maxlookback]):
                #print >> sys.stdout, "warning! Start date besides data length! pls use -A for all sample"
                self.sidx = self.maxlookback
            else:
                self.sidx = np.where(raw_dates_index <= start)[0][-1]
            if end > pd.to_datetime(raw_dates[-1]):
                self.eidx = raw_dates.shape[0] - 1
                #raise Exception("end date besides data length!")
            else:
                self.eidx = np.where(raw_dates_index >= end)[0][0]

        elif self.data_set == 'all':
            # 考虑每个品种当日成交量5万手为成交活跃, 作为起始日
            if self.instrument in self.instrument_info_dict:
                start = pd.to_datetime(self.instrument_info_dict[self.instrument]['BackTestStartDate'])
                raw_dates_index = pd.to_datetime(raw_dates.astype(str))
                if start < pd.to_datetime(raw_dates[self.maxlookback]):
                    self.sidx = self.maxlookback
                else:
                    self.sidx = np.where(raw_dates_index <= start)[0][-1]
            else:
                self.sidx = self.maxlookback
            self.eidx = raw_dates.shape[0] - 1

        self.sidx = self.sidx - self.maxlookback + start_point
        self.backtest_len = self.eidx - self.sidx
        self.lookback_start = raw_dates[self.sidx]
        self.lookback_end = raw_dates[self.sidx+self.maxlookback]
        self.s_date = raw_dates[self.sidx + self.maxlookback]
        self.e_date = raw_dates[self.eidx]
        if print_signal_info is True:
            print >> sys.stdout, 'lookback date: ', self.lookback_start, ' - ', self.lookback_end
            print >> sys.stdout, 'backtest date: ', self.s_date, ' - ', self.e_date
            self.signal_info_screen(print_signal_info)
        self.ClosePrice = self.raw_df['ClosePrice'].values[self.sidx:self.eidx] * 1.
        self.OpenPrice = self.raw_df['OpenPrice'].values[self.sidx:self.eidx] * 1.
        self.HighestPrice = self.raw_df['HighestPrice'].values[self.sidx:self.eidx] * 1.
        self.LowestPrice = self.raw_df['LowestPrice'].values[self.sidx:self.eidx] * 1.
        self.Volume = self.raw_df['Volume'].values[self.sidx:self.eidx] * 1.
        self.dates = self.raw_df.index.values[self.sidx:self.eidx] 


    def signal_info_screen(self, print_signal_info=True):
        # output signal cfg to screen
        direct = {1: "Long Only", 0: "Long & Short", -1: "Short only"}
        tmp = np.column_stack([self.cfg["Name"], self.cfg["Benchmark"], self.s_date, self.e_date,
                               self.backtest_len - self.maxlookback, self.maxlookback,
                               self.lookback_start, self.lookback_end,
                               direct[self.cfg["Direction"]],
                               self.raw_data_start, self.raw_data_end, 
                               self.slippoint, self.commission, self.pointValue, self.tickSize
                               ])
        self.info_df = pd.DataFrame(tmp, index=[""], columns=np.array(["SignalName", "InstrumentID", 
                                                                  "BacktestStart", "BacktestEnd",
                                                                  "BacktestDateLength", "MaxLookback",
                                                                  "LookbackStart", "LookbackEnd",
                                                                  "Direction", "RawDataStart", 
                                                                  "RawDataEnd", "SlipPoint", "Commission",
                                                                              "PointValue", 'MinMove']))
        if print_signal_info is False: return
        print self.info_df.loc[:,["SignalName", "InstrumentID", "BacktestDateLength", "MaxLookback",
                                                                  "Direction", 'PointValue', 'MinMove', 'SlipPoint', "Commission"]].T

    def prebatch(self, print_info=True):
        self.signal.ClosePrice = self.ClosePrice.copy()
        self.signal.OpenPrice = self.OpenPrice.copy()
        self.signal.HighestPrice = self.HighestPrice.copy()
        self.signal.LowestPrice = self.LowestPrice.copy()
        self.signal.dates = self.dates.copy()
        self.signal.Volume = self.Volume.copy()

        if print_info is True:
            print >> sys.stdout, "\nStep1: Prebatch is running..."
        self.signal.prebatch()


    def prepare_trade(self):
        self.tranx_order = {}
        self.signal.tranx_order = self.tranx_order
        self.tranx_record_df = pd.DataFrame()
        self.tranx_order_num = 0

        self.rolling_direction = {}
        self.rolling_pnl = {}
        self.rolling_net_pnl = {}
        self.rolling_dates = {}


    def work(self, process_bar=True, print_info=True):
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
        if print_info is True:
            print >> sys.stdout, "\nStep2: Data ready! Starting roll...\n"
        from tqdm import tqdm
        for di in tqdm(xrange(self.maxlookback, self.dates.shape[0], 1), disable=(process_bar is False)):
            self.last_bar_init_time = self.dates[di-1]
            self.check_order(di-1)       # 结算前检查订单
            self.clear(di-1)             # 结算上一轮订单
            self.roll(di)              # 新一轮的回测, 产生订单
            order_dict = self.trade_order(di)   # 虚拟盘交易
            self.process_order(order_dict, di)   # 处理交易结果


    def check_order(self, di): 
        # 清算前, 核对订单
        if len(self.tranx_order) >0:
            # 未持仓
            if self.tranx_order['Open deal'] == 'No':
                self.tranx_order['bars'] += 1
            # 过期Stop订单
            if self.tranx_order['Open deal'] == 'No' and self.tranx_order['Open type'] == 'Stop' and self.tranx_order['bars'] >= self.tranx_order['expired']:
                self.tranx_order['Comment'] = 'Expired'
                self.tranx_order['Close time'] = self.dates[di]
                self.tranx_order['Open time'] = self.dates[di]
                self.tranx_order['Close price'] = self.tranx_order['Target Open price']
                self.tranx_order['Open price'] = self.tranx_order['Target Open price']
                self.tranx_order = {}  #订单关闭
                return 


    def clear(self, di):
        self.rolling_dates[self.dates[di]] = self.dates[di]
        if self.dates[di] not in self.rolling_pnl: self.rolling_pnl[self.dates[di]] = 0
        if self.dates[di] not in self.rolling_net_pnl: self.rolling_net_pnl[self.dates[di]] = 0
        # 空仓
        if len(self.tranx_order) == 0:
            self.rolling_pnl[self.dates[di]] = 0
            self.rolling_net_pnl[self.dates[di]] = 0
            self.rolling_direction[self.dates[di]] = 0
        # 未成交
        elif self.tranx_order['Open deal'] == 'No':
            self.rolling_direction[self.dates[di]] = 0
            self.rolling_pnl[self.dates[di]] = 0  
            self.rolling_net_pnl[self.dates[di]] = 0
            self.tranx_order['actual_profit'] = 0
        # 开仓
        elif self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Bars_Since_Entry'] == 0:
            self.rolling_direction[self.dates[di]] = self.tranx_order['direction']
            self.rolling_pnl[self.dates[di]] += self.pointValue  * self.tranx_order['direction'] * (self.ClosePrice[di] - self.tranx_order['Open price'])
            self.rolling_net_pnl[self.dates[di]] += self.pointValue  * self.tranx_order['direction'] * (self.ClosePrice[di] - self.tranx_order['Open price']) - self.slippoint * self.pointValue  - self.commission * self.pointValue * self.ClosePrice[di]    
            self.tranx_order['actual_profit'] = self.tranx_order['direction'] * (self.ClosePrice[di] - self.tranx_order['Open price'])
            # 累计收益
            self.last_bar_profit = self.rolling_pnl[self.dates[di]]/self.pointValue   #最近的一个bar的收益
            self.tranx_order['Bars_Since_Entry'] += 1
        # 平仓
        elif self.tranx_order['Close deal'] == 'Yes':
            self.tranx_calcula()
            self.tranx_order['actual_profit'] = self.tranx_order['direction'] * (self.tranx_order['Close price'] - self.tranx_order['Open price'])
            self.rolling_direction[self.dates[di]] = 0
            self.rolling_pnl[self.dates[di]] += self.pointValue  * self.tranx_order['direction'] * (self.tranx_order['Close price'] - self.ClosePrice[di-1])
            self.rolling_net_pnl[self.dates[di]] += self.pointValue  * self.tranx_order['direction'] * (self.tranx_order['Close price'] - self.ClosePrice[di-1]) - self.slippoint * self.pointValue  - self.commission * self.pointValue * self.ClosePrice[di]
            self.tranx_order = {} # init
        # 持仓
        elif self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Bars_Since_Entry'] > 0:
            self.rolling_direction[self.dates[di]] = self.tranx_order['direction']
            self.rolling_pnl[self.dates[di]] = self.pointValue  * self.tranx_order['direction'] * (self.ClosePrice[di] - self.ClosePrice[di-1])
            self.rolling_net_pnl[self.dates[di]] = self.pointValue  * self.tranx_order['direction'] * (self.ClosePrice[di] - self.ClosePrice[di-1])
            # 累计收益
            self.tranx_order['actual_profit'] = self.tranx_order['direction'] * (self.ClosePrice[di] - self.tranx_order['Open price'])
            self.last_bar_profit = self.rolling_pnl[self.dates[di]]/self.pointValue   #最近的一个bar的收益
            self.tranx_order['Bars_Since_Entry'] += 1
        else:
            print 'Close deal', self.tranx_order['Close deal']
            print 'Bars_Since_Entry', self.tranx_order['Bars_Since_Entry']
            print 'Open deal', self.tranx_order['Open deal']
            raise Exception('corner case for clear()')



    def tranx_calcula(self):
        # 每个bar结算
        self.tranx_order_num += 1
        self.tranx_order['Type'] = 'Long' if self.tranx_order['direction'] == 1 else 'Short'
        self.tranx_order['Size'] = self.Size
        self.tranx_order['Symbol'] = self.instrument_id
        self.tranx_order['Timeframe'] = self.time_frame
        self.tranx_order['P/L in pips'] = self.tranx_order['direction'] * (self.tranx_order['Close price'] - self.tranx_order['Open price'])
        self.tranx_order['P/L in money'] = self.pointValue   * self.tranx_order['P/L in pips']
        self.tranx_order['Time in trade'] = str(pd.to_datetime(self.tranx_order['Close time']) - pd.to_datetime(self.tranx_order['Open time']))
        tmp = pd.DataFrame(self.tranx_order, index=[self.tranx_order_num])
        tmp['Order'] = self.tranx_order_num
        self.tranx_record_df = self.tranx_record_df.append(tmp)



    def roll(self, di):
        # 处理信号
        self.signal.generate(di, self)
        # 已持仓的话, 更新订单
        if len(self.tranx_order) >0 and self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Close deal'] == 'No':
            # =======================================
            # 更新定点止盈止损
            if self.tranx_order['Bars_Since_Entry'] == 1:
                if self.tranx_order['LongSL_pips'] != 0: self.tranx_order['LongSL'] = self.tranx_order['Target Open price'] - self.tranx_order['LongSL_pips']
                if self.tranx_order['ShortSL_pips'] != 0: self.tranx_order['ShortSL']= self.tranx_order['Target Open price'] + self.tranx_order['ShortSL_pips']
                if self.tranx_order['LongPT_pips'] != 0: self.tranx_order['LongPT'] = self.tranx_order['Target Open price'] + self.tranx_order['LongPT_pips']
                if self.tranx_order['ShortPT_pips'] != 0: self.tranx_order['ShortPT'] = self.tranx_order['Target Open price'] - self.tranx_order['ShortPT_pips'] 

            # =======================================
            # 更新其他
            # self.tranx_order['Bars_Since_Entry'] += 1 
            # =======================================
            # 更新多头止损线
            if self.tranx_order['LongSL'] != 0 and self.tranx_order['LongSL_trailing'] != 0:
                self.tranx_order['LongSL'] = max(self.tranx_order['LongSL'], self.tranx_order['LongSL_trailing'])
            elif self.tranx_order['LongSL'] == 0 and self.tranx_order['LongSL_trailing'] != 0:
                self.tranx_order['LongSL'] = self.tranx_order['LongSL_trailing']

            # 更新空头止损线
            if self.tranx_order['ShortSL'] != 0 and self.tranx_order['ShortSL_trailing'] != 0:
                self.tranx_order['ShortSL'] = min(self.tranx_order['ShortSL'], self.tranx_order['ShortSL_trailing'])
            elif self.tranx_order['ShortSL'] == 0 and self.tranx_order['ShortSL_trailing'] != 0:
                self.tranx_order['ShortSL'] = self.tranx_order['ShortSL_trailing']

            # 更新多头止盈线
            if self.tranx_order['LongPT'] != 0 and self.tranx_order['LongPT_trailing'] != 0:
                self.tranx_order['LongPT'] = min(self.tranx_order['LongPT'], self.tranx_order['LongPT_trailing'])
            elif self.tranx_order['LongPT'] == 0 and self.tranx_order['LongPT_trailing'] != 0:
                self.tranx_order['LongPT'] = self.tranx_order['LongPT_trailing']
            # 更新空头止盈线
            if self.tranx_order['ShortPT'] != 0 and self.tranx_order['ShortPT_trailing'] != 0:
                self.tranx_order['ShortPT'] = max(self.tranx_order['ShortPT'], self.tranx_order['ShortPT_trailing'])
            elif self.tranx_order['ShortPT'] == 0 and self.tranx_order['ShortPT_trailing'] != 0:
                self.tranx_order['ShortPT'] = self.tranx_order['ShortPT_trailing']


    def trade_order(self, di):
        # 对应实盘交易, 返回成交价格和成交时间
        if len(self.tranx_order) >0:
            # 未持仓
            if self.tranx_order['Open deal'] == 'No' and self.tranx_order['Open exec'] == 'No':
                #限价单
                if self.tranx_order['Open type'] == 'Stop' and self.tranx_order['Target Open price'] >= self.LowestPrice[di] and self.tranx_order['Target Open price'] <= self.HighestPrice[di]:
                    self.tranx_order['Target Open time'] = self.dates[di]
                    self.tranx_order['Open exec'] = 'Yes'
                    return {'price': self.tranx_order['Target Open price'], 'time': self.dates[di]}
                # 开盘市价单
                elif self.tranx_order['Open type'] == 'OpenPrice':
                    self.tranx_order['Target Open price'] = self.OpenPrice[di]
                    self.tranx_order['Target Open time'] = self.dates[di]
                    self.tranx_order['Open exec'] = 'Yes'
                    return {'price': self.OpenPrice[di], 'time': self.dates[di]}
                # 收盘市价单
                elif self.tranx_order['Open type'] =='ClosePrice':
                    self.tranx_order['Target Open price'] = self.ClosePrice[di]
                    self.tranx_order['Target Open time'] = self.dates[di]
                    self.tranx_order['Open exec'] = 'Yes'
                    return {'price': self.ClosePrice[di], 'time': self.dates[di]}
                else:
                    return {}
            # 已持仓
            elif self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Close deal'] == 'No' and self.tranx_order['Close exec'] == 'No':
                # 平仓的优先级: 开盘价 > 反向开单 > 止损 > 止盈
                # =======================================
                # Exit 市价平仓使用开盘价
                if 'Close type' in self.tranx_order.keys() and self.tranx_order['Close type'] == 'OpenPrice':
                    self.tranx_order['Comment'] = 'Exit'
                    self.tranx_order['Target Close price'] =  self.OpenPrice[di]
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'price': self.OpenPrice[di], 'time': self.dates[di]}

                # =======================================
                # 反向开单
                if 'Reverse type' in self.tranx_order.keys() and self.tranx_order['Reverse type'] == 'OpenPrice':
                    self.tranx_order['Target Close price'] =  self.OpenPrice[di]
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Comment'] = 'Reverse'
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'price': self.OpenPrice[di], 'time': self.dates[di]}

                # =======================================
                # 判断订单的多头止损
                if self.tranx_order['LongSL'] >= self.LowestPrice[di] and self.tranx_order['LongSL'] <= self.HighestPrice[di]:
                    self.tranx_order['Comment'] = 'Long SL'
                    self.tranx_order['Target Close price'] =  self.tranx_order['LongSL']
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'
                    return  {'price': self.tranx_order['LongSL'], 'time': self.dates[di]}
                elif self.tranx_order['LongSL'] > self.HighestPrice[di]:
                    # 跳空的情况
                    self.tranx_order['Comment'] = 'Long Gap SL'
                    self.tranx_order['Target Close price'] =  self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'price': self.ClosePrice[di], 'time': self.dates[di]}
                # 判断订单的空头止损
                if self.tranx_order['ShortSL'] >= self.LowestPrice[di] and self.tranx_order['ShortSL'] <= self.HighestPrice[di]:
                    self.tranx_order['Comment'] = 'Short SL'
                    self.tranx_order['Target Close price'] = self.tranx_order['ShortSL']
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'price': self.tranx_order['ShortSL'], 'time': self.dates[di]}
                elif self.tranx_order['ShortSL'] > 0 and self.tranx_order['ShortSL'] < self.LowestPrice[di]:
                    # 跳空
                    self.tranx_order['Comment'] = 'Short Gap SL'
                    self.tranx_order['Target Close price'] = self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'  
                    return {'price': self.ClosePrice[di], 'time': self.dates[di]}

                # 判断订单的多头止盈
                if self.tranx_order['LongPT'] >= self.LowestPrice[di] and self.tranx_order['LongPT'] <= self.HighestPrice[di]:
                    self.tranx_order['Comment'] = 'Long PT'
                    self.tranx_order['Target Close price'] = self.tranx_order['LongPT']
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'  
                    return {'price': self.tranx_order['LongPT'], 'time': self.dates[di]}
                elif self.tranx_order['LongPT'] > 0 and self.tranx_order['LongPT'] < self.LowestPrice[di]:
                    # 跳空
                    self.tranx_order['Comment'] = 'Long Gap PT'
                    self.tranx_order['Target Close price'] = self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'    
                    return {'price': self.ClosePrice[di], 'time': self.dates[di]}
                # 判断订单的空头止盈
                if self.tranx_order['ShortPT'] >= self.LowestPrice[di] and self.tranx_order['ShortPT'] <= self.HighestPrice[di]:
                    self.tranx_order['Comment'] = 'Short PT'
                    self.tranx_order['Target Close price'] = self.tranx_order['ShortPT']
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'  
                    return {'price': self.tranx_order['ShortPT'], 'time': self.dates[di]}
                elif self.tranx_order['ShortPT'] > self.HighestPrice[di]:
                    # 跳空
                    self.tranx_order['Comment'] = 'Short Gap PT'
                    self.tranx_order['Target Close price'] = self.ClosePrice[di]
                    self.tranx_order['Target Close time'] = self.dates[di]
                    self.tranx_order['Close exec'] = 'Yes'
                    return {'price': self.ClosePrice[di], 'time': self.dates[di]}
                return {}
            else:
                raise Exception('roll() corner case')
        else:
            return {}



    def process_order(self, order_dict, di):
        if order_dict == {}: return   # 无成交的交易
        if len(self.tranx_order) >0:
            # 未持仓
            if self.tranx_order['Open deal'] == 'No':
                #限价单
                self.tranx_order['Open price'] = order_dict['price']
                self.tranx_order['Open time'] = order_dict['time']
                self.tranx_order['Open deal'] = 'Yes'
                self.tranx_order['Close deal'] = 'No'
            # 已持仓
            elif self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Close deal'] == 'No':
                self.tranx_order['Close price'] = order_dict['price']
                self.tranx_order['Close time'] = order_dict['time']
                self.tranx_order['Close deal'] = 'Yes'
                if 'Reverse type' in self.tranx_order.keys() and self.tranx_order['Reverse type'] == 'OpenPrice':
                    Reverse_direction = self.tranx_order['direction'] * -1
                    self.clear(di)
                    if Reverse_direction == 1: 
                        self.Buy_on_Open()
                    else:
                        self.Sell_on_Open()
                    self.process_order(order_dict, di)
                    #print order_dict, self.tranx_order
                    self.tranx_order['Reverse deal'] = 'Yes'
                    return
            else:
                raise Exception('roll() corner case')




    def stat_Ddp(self, cpnl):
        rmax = np.array([np.nanmax(cpnl[:i + 1]) for i in xrange(cpnl.shape[0])])
        d = cpnl - rmax
        max_dd_time = 0
        dd_time = 0
        for i in xrange(d.shape[0]):
            if d[i] >= 0:
                if dd_time >= max_dd_time:
                    max_dd_time = dd_time
                dd_time = 0
            else:
                dd_time += 1
        if dd_time >= max_dd_time:
            max_dd_time = dd_time
        return max_dd_time

    def stat_drawdown(self, cpnl):
        dd = cpnl - np.array([np.nanmax(cpnl[:i + 1]) for i in xrange(cpnl.shape[0])])
        return dd

    # 内部函数
    def random_param_ratio_distrib(self):
        return np.random.uniform(-self.random_history_data_range, self.random_history_data_range)

    def PriceScale(self, num):
        # 输入为float长度不定的价格, 转化为能交易的价格, 最小变动价格为0.2, 则127.323 ---> 127.4
        tmp = num%self.tickSize/self.tickSize  
        tmp2 = num//self.tickSize
        if tmp >= 0.5: tmp2 += 1 
        return tmp2 * self.tickSize

    def PipsScale(self, num):
        return self.tickSize * int(num)  # 同样止损点位, 来适应不同品种


    def Init_tranx_order(self, trade_type, direction, trade_price=None):
        self.tranx_order = {}  
        self.tranx_order['Open Init time'] = self.last_bar_init_time
        self.tranx_order['Open type'] = trade_type
        self.tranx_order['Open deal'] = 'No'
        self.tranx_order['Open exec'] = 'No'
        self.tranx_order['Close deal'] = 'No'
        self.tranx_order['Close exec'] = 'No'
        self.tranx_order['bars'] = 0
        self.tranx_order['LongSL_pips'] = 0
        self.tranx_order['LongPT_pips'] = 0
        self.tranx_order['ShortPT_pips'] = 0
        self.tranx_order['ShortSL_pips'] = 0
        # self.tranx_order['LongSL_point'] = 0
        # self.tranx_order['ShortSL_point'] = 0
        # self.tranx_order['LongPT_point'] = 0
        # self.tranx_order['ShortPT_point'] = 0
        self.tranx_order['LongSL_trailing'] = 0
        self.tranx_order['ShortSL_trailing'] = 0
        self.tranx_order['LongPT_trailing'] = 0
        self.tranx_order['ShortPT_trailing'] = 0
        self.tranx_order['LongSL'] = 0
        self.tranx_order['LongPT'] = 0
        self.tranx_order['ShortPT'] = 0
        self.tranx_order['ShortSL'] = 0
        self.tranx_order['direction'] = direction
        self.tranx_order['Bars_Since_Entry'] = 0
        self.tranx_order['actual_profit'] = 0

        if trade_type == 'Stop':
            self.tranx_order['expired'] = 1e10  #default inf
            self.tranx_order['Target Open price'] = trade_price
        elif trade_type != 'Stop' and trade_price is not None:
            raise Exception('Order type error! Only stop/limit order has specific price')

    # ========================================== 以下面向外部使用 ==========================================
    # --- * 以下回测 * ----
    def run(self, data_mode='mongo', print_info=True, print_signal_info=True, plot=True, process_bar=True, plot1=True, plot2=True, saved_path=None, summary1=True, summary2=True, IS_end_date=None):
        if plot is False: plot1, plot2 = False, False
        self.data_prepare(print_signal_info=print_signal_info, mode=data_mode)
        self.prebatch(print_info=print_info)
        self.prepare_trade()
        self.work(print_info=print_info, process_bar=process_bar)
        self.summary(print_summary=summary1)
        self.pnl_plot(plot=plot1, saved_path=saved_path)
        self.summary2(print_summary=summary2)
        self.pnl_plot2(plot=plot2, saved_path=saved_path, IS_end_date=IS_end_date)


    def summary(self, print_summary=True):
        # 总的交易结算
        #print self.tranx_record_df.T
        if len(self.tranx_record_df) == 0:
            self.tranx_record_df['Cummulative P/L in pips'], self.tranx_record_df['Cummulative P/L in money'], self.tranx_record_df['Net P/L in money'], self.tranx_record_df['Cummulative Net P/L in money'] = 0, 0, 0, 0
            self.stat_df = pd.DataFrame(index=[''])
            self.stat_df['InstrumentId'] = self.instrument_id
            self.stat_df['Original profit $'] = 0
            self.stat_df['Net profit $'] = 0
            self.stat_df['# trades'] = 0
            self.stat_df['Stability'] = 0
            self.stat_df['DD bar nums'] = 0
            self.stat_df['Max DD $'] = 0
            self.stat_df['Ret/DD Ratio'] = 0
            self.stat_df['Win/Loss Ratio'] = 0
            self.stat_df['Buy profit'] = 0
            self.stat_df['Sell profit'] = 0
            self.stat_df['Avg win $'] = 0
            self.stat_df['Avg loss $'] = 0
            self.stat_df['Avg holding bars'] = 0
            self.stat_df['Avg holding long bars'] = 0
            self.stat_df['Avg holding short bars'] = 0
            self.stat_df['Avg holding win bars'] = 0
            self.stat_df['Avg holding loss bars'] = 0
            return
        self.tranx_record_df['Cummulative P/L in pips'] = np.cumsum(self.tranx_record_df['P/L in pips'])
        #self.tranx_record_df['Cummulative P/L'] = np.cumsum(self.tranx_record_df['Profit/Loss'])
        self.tranx_record_df['Cummulative P/L in money'] = np.cumsum(self.tranx_record_df['P/L in money'].values)
        # consider slippoint + brokerage fee
        #print self.tranx_record_df['Close price'].values*self.pointValue
        #print self.slippoint*self.pointValue + self.commission*self.tranx_record_df['Close price'].values*self.pointValue  
        self.tranx_record_df['Net P/L in money'] = self.tranx_record_df['P/L in money'].values - 2*self.slippoint*self.pointValue - 2*self.commission*self.tranx_record_df['Close price'].values*self.pointValue     
        self.tranx_record_df['Cummulative Net P/L in money'] = np.cumsum(self.tranx_record_df['Net P/L in money'])
        self.tranx_record_df = self.tranx_record_df[['Symbol', 'Timeframe', 'Type', 'Open time', 'Open price', 'Size', 'Close time', \
                                                'Close price', 'P/L in money', 'Cummulative P/L in money', 'Net P/L in money',\
                                                'Cummulative Net P/L in money', 'P/L in pips', 'Cummulative P/L in pips', 'Comment','LongPT', 'LongSL', 'ShortPT', 'ShortSL', 'Time in trade', 'Bars_Since_Entry']]
        self.tranx_record_df.index.name = 'Order'


        self.stat_df = pd.DataFrame(index=[''])
        self.stat_df['InstrumentId'] = self.instrument_id
        pnl = self.tranx_record_df['P/L in money'].values
        net_pnl = self.tranx_record_df['Net P/L in money'].values
        c_net_cpnl = self.tranx_record_df['Cummulative Net P/L in money'].values
        self.stat_df['Original profit $'] =  self.tranx_record_df['Cummulative P/L in money'].values[-1] 
        self.stat_df['Net profit $'] = c_net_cpnl[-1]
        self.stat_df['# trades'] = c_net_cpnl.shape[0]
        self.stat_df['Stability'] = round(Rsquared(c_net_cpnl), 2)
        self.stat_df['DD bar nums'] = self.stat_Ddp(c_net_cpnl)
        self.draw_down = self.stat_drawdown(c_net_cpnl)
        self.stat_df['Max DD $'] = np.max(np.abs(self.draw_down))
        self.stat_df['Ret/DD Ratio'] = round(self.stat_df['Net profit $']/self.stat_df['Max DD $'], 1)
        self.stat_df['Win/Loss Ratio'] = round(abs(np.sum(net_pnl[net_pnl>0])/np.sum(net_pnl[net_pnl<0])), 1)
        self.stat_df['Buy profit'] = np.sum(net_pnl[self.tranx_record_df['Type'].values=='Long'])
        self.stat_df['Sell profit'] = np.sum(net_pnl[self.tranx_record_df['Type'].values=='Short'])
        self.stat_df['Avg win $'] = round(np.mean(net_pnl[net_pnl>0]), 1)
        self.stat_df['Avg loss $'] = round(np.mean(net_pnl[net_pnl<0]), 1)
        self.stat_df['Avg holding bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values), 1)
        self.stat_df['Avg holding long bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Type'].values=='Long']), 1)
        self.stat_df['Avg holding short bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Type'].values=='Short']), 1)
        self.stat_df['Avg holding win bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Net P/L in money'].values>0]), 1)
        self.stat_df['Avg holding loss bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Net P/L in money'].values<0]), 1)       
        #self.stat_df['Symmetry'] = str(self.stat_df['Buy profit'].values[0]/(self.stat_df['Buy profit'].values[0] + self.stat_df['Sell profit'].values[0]*100) + '%'

        if print_summary==False: return
        print 'Summary (tranx axis):'
        from prettytable import PrettyTable 
        x = PrettyTable(list(self.stat_df.columns.values[:5]))
        x.add_row(list(self.stat_df.iloc[0,:].values[0:5]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[5:9]))
        x.add_row(list(self.stat_df.iloc[0,:].values[5:9]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[9:13]))
        x.add_row(list(self.stat_df.iloc[0,:].values[9:13]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[13:16]))
        x.add_row(list(self.stat_df.iloc[0,:].values[13:16]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[16:]))
        x.add_row(list(self.stat_df.iloc[0,:].values[16:]))
        print x

    def summary2(self, print_summary=True):
        if len(self.tranx_record_df) == 0:
            self.tranx_record_df['Cummulative P/L in pips'], self.tranx_record_df['Cummulative P/L in money'], self.tranx_record_df['Net P/L in money'], self.tranx_record_df['Cummulative Net P/L in money'] = 0, 0, 0, 0
            self.stat_df = pd.DataFrame(index=[''])
            self.stat_df['InstrumentId'] = self.instrument_id
            self.stat_df['Original profit $'] = 0
            self.stat_df['Net profit $'] = 0
            self.stat_df['# trades'] = 0
            self.stat_df['Stability'] = 0
            self.stat_df['DD bar nums'] = 0
            self.stat_df['Max DD $'] = 0
            self.stat_df['Ret/DD Ratio'] = 0
            self.stat_df['Win/Loss Ratio'] = 0
            self.stat_df['Sharpe Ratio'] = 0
            return
        self.stat_df = pd.DataFrame(index=[''])
        self.stat_df['InstrumentId'] = self.instrument_id
        self.rolling_dates = np.sort(self.rolling_dates.keys())
        self.rolling_pnl = np.array([self.rolling_pnl[i] for i in self.rolling_dates])
        self.rolling_net_pnl = np.array([self.rolling_net_pnl[i] for i in self.rolling_dates]) 
        self.rolling_direction = np.array([self.rolling_direction[i] for i in self.rolling_dates]) 
        self.cpnl = np.cumsum(self.rolling_pnl)
        self.net_cpnl = np.cumsum(self.rolling_net_pnl)
        self.stat_df['Original profit $'] =  self.cpnl[-1] 
        self.stat_df['Net profit $'] = self.net_cpnl[-1]

        daily_pnl = operators.ts_sum(self.rolling_net_pnl, int(self.daily_bar_num))[np.arange(0, len(self.rolling_net_pnl), int(self.daily_bar_num))]
        print daily_pnl
        self.stat_df['Sharpe Ratio'] = np.nanmean(daily_pnl)/np.nanstd(daily_pnl) * np.sqrt(252)
        self.stat_df['# trades'] = self.tranx_record_df.shape[0]
        self.stat_df['Stability'] = round(Rsquared(self.net_cpnl), 2)
        self.stat_df['DD period'] = self.stat_Ddp(self.net_cpnl)
        self.draw_down = self.stat_drawdown(self.net_cpnl)
        self.stat_df['Max DD $'] = np.max(np.abs(self.draw_down))
        self.stat_df['Ret/DD Ratio'] = round(self.stat_df['Net profit $']/self.stat_df['Max DD $'], 1)

        self.stat_df['Win/Loss Ratio'] = round(abs(np.sum(self.rolling_net_pnl[self.rolling_net_pnl>0])/np.sum(self.rolling_net_pnl[self.rolling_net_pnl<0])), 1)
        #print self.rolling_pnl, self.rolling_direction
        self.stat_df['Buy profit'] = np.sum(self.rolling_net_pnl[self.rolling_direction==1])
        self.stat_df['Sell profit'] = np.sum(self.rolling_net_pnl[self.rolling_direction==-1])
        self.stat_df['Avg win $'] = round(np.mean(self.rolling_net_pnl[self.rolling_direction==1]), 1)
        self.stat_df['Avg loss $'] = round(np.mean(self.rolling_net_pnl[self.rolling_direction==-1]), 1)

        self.stat_df['Avg holding bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values), 1)
        self.stat_df['Avg holding long bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Type'].values=='Long']), 1)
        self.stat_df['Avg holding short bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Type'].values=='Short']), 1)
        self.stat_df['Avg holding win bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Net P/L in money'].values>0]), 1)
        self.stat_df['Avg holding loss bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Net P/L in money'].values<0]), 1)     

        if print_summary==False: return
        print 'Summary (time axis):'
        from prettytable import PrettyTable 
        x = PrettyTable(list(self.stat_df.columns.values[:5]))
        x.add_row(list(self.stat_df.iloc[0,:].values[0:5]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[5:9]))
        x.add_row(list(self.stat_df.iloc[0,:].values[5:9]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[9:13]))
        x.add_row(list(self.stat_df.iloc[0,:].values[9:13]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[13:16]))
        x.add_row(list(self.stat_df.iloc[0,:].values[13:16]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[16:]))
        x.add_row(list(self.stat_df.iloc[0,:].values[16:]))
        print x


    def pnl_plot(self, plot=True, web_plot=False, saved_path=None):
        if plot is False: return
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16,12))
        rect1 = [0.05, 0.225, 0.9, 0.8]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
        rect2 = [0.05, 0.025, 0.9, 0.17]
        sub1 = plt.axes(rect1)
        sub2 = plt.axes(rect2)

        plt.sca(sub1)
        # original line 
        plt.plot(self.tranx_record_df['Cummulative P/L in money'].values, label='Original')
        plt.plot(self.tranx_record_df['Cummulative Net P/L in money'].values, label='net')
        # direction line
        x= np.arange(self.tranx_record_df.shape[0])
        #print self.tranx_record_df['Cummulative Net P/L in money']
        y = self.tranx_record_df['Cummulative Net P/L in money'].values[-1] * x/(self.tranx_record_df.shape[0]-1)

        plt.plot(y)
        dates_str = self.tranx_record_df['Close time'].values.astype(str)

        if '.' in dates_str[0]:
            split_str = '.'
        elif '/' in dates_str[0]:
            split_str = '/'
        else:
            split_str = '-'
            #raise Exception('cannot recognise date string!')
        dates_short_str = np.array([i.split(split_str)[0] for i in dates_str])  # x轴为交易等距, 非时间等距
        if len(np.unique(dates_short_str))<3:   # 优化x坐标轴显示
            dates_short_str = np.array([i.split(split_str)[0] + split_str + i.split(split_str)[1] for i in dates_str])             
        space = np.array([np.where(dates_short_str==i)[0][0] for i in np.unique(dates_short_str)])
        #dates_str = np.array([i[:7] for i in dates_str])  # 直接显示时间
        #space = np.int32(np.linspace(0, dates_str.shape[0] - 1, 9))
        plt.xticks(space, dates_str[space])
        plt.ylabel("Profit $")
        plt.legend(loc='upper left')
        plt.grid()
        plt.title(self.signal_name + ' ' + self.instrument_id + ' # ' + str(self.tranx_record_df.shape[0]) + ' trades')
        
        plt.sca(sub2)
        plt.fill_between(np.arange(self.draw_down.shape[0]), np.zeros_like(self.draw_down), self.draw_down, color='red', alpha=0.5)
        plt.ylabel("Drawdown $")
        plt.grid()
        if web_plot:
            import mpld3
            mpld3.show()
        elif saved_path is not None:
            plt.savefig(saved_path)
            return 
        plt.show()



    def pnl_plot2(self, plot=True, saved_path=None, IS_end_date=None):
        if plot is False: return
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16,8))
        rect1 = [0.05, 0.225, 0.9, 0.8]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
        rect2 = [0.05, 0.025, 0.9, 0.17]
        sub1 = plt.axes(rect1)
        sub2 = plt.axes(rect2)

        plt.sca(sub1)
        # original line 
        plt.plot(self.cpnl, label='Original')
        plt.plot(self.net_cpnl, label='net')
        # direction line
        #x= np.arange(self.cpnl.shape[0])
        #y = self.net_cpnl[-1] * x/(self.net_cpnl.shape[0]-1)
        #plt.plot(y)
        if IS_end_date is not None:
            IS_OOS_divide_line = np.where(self.rolling_dates >= IS_end_date)[0][0]
            plt.axvspan(IS_OOS_divide_line, len(x), edgecolor='red', facecolor='grey', linewidth=1.5, alpha=0.15)
        dates_str = np.array(self.rolling_dates)
        if '.' in dates_str[0]:
            split_str = '.'
        elif '/' in dates_str[0]:
            split_str = '/'
        else:
            split_str = '-'
        dates_short_str = np.array([i.split(split_str)[0] for i in dates_str])  # x轴为时间等距
        if len(np.unique(dates_short_str))<3:   # 优化x坐标轴显示
            dates_short_str = np.array([i.split(split_str)[0] + split_str + i.split(split_str)[1] for i in dates_str])             
        space = np.array([np.where(dates_short_str==i)[0][0] for i in np.unique(dates_short_str)])
        #dates_str = np.array([i[:7] for i in dates_str])  # 直接显示时间
        #space = np.int32(np.linspace(0, dates_str.shape[0] - 1, 9))
        plt.xticks(space, dates_str[space])
        plt.ylabel("Profit $")
        plt.legend(loc='upper left')
        plt.grid()
        plt.title(self.signal_name + ' ' + self.instrument_id + ' # ' + str(self.tranx_record_df.shape[0]) + ' trades')
        
        plt.sca(sub2)
        plt.fill_between(np.arange(self.draw_down.shape[0]), np.zeros_like(self.draw_down), self.draw_down, color='red', alpha=0.5)
        plt.ylabel("Drawdown $")
        plt.grid()
        if saved_path is not None:
            plt.savefig(saved_path)
            return 
        plt.show()


    def web_plot(self, path):
        # 绘制开平仓位置(买卖点), 使用mpld3可进行缩放观察
        import matplotlib.pyplot as plt
        plt.figure(figsize=(23,12))
        close = self.ClosePrice[self.maxlookback:]
        plt.plot(close)
        dates_str = np.array(self.rolling_dates)
        # if '.' in dates_str[0]:
        #     split_str = '.'
        # elif '/' in dates_str[0]:
        #     split_str = '/'
        # else:
        #     split_str = '-'
        # dates_short_str = np.array([i.split(split_str)[0] for i in dates_str])  # x轴为时间等距
        # if len(np.unique(dates_short_str))<3:   # 优化x坐标轴显示
        #     dates_short_str = np.array([i.split(split_str)[0] + split_str + i.split(split_str)[1] for i in dates_str])             
        #space = np.array([np.where(dates_short_str==i)[0][0] for i in np.unique(dates_short_str)])
        #dates_str = np.array([i[:7] for i in dates_str])  # 直接显示时间
        #space = np.int32(np.linspace(0, dates_str.shape[0] - 1, 9))
        space = np.arange(0, len(dates_str), int(len(dates_str)/10.))
        plt.xticks(space, dates_str[space])


        sell, sell_day = [], []
        buy, buy_day = [], []
        wgt = [] 
        pnl = [0]
        direction = self.rolling_direction

        print len(self.rolling_direction), len(close)
        for i in range(len(close)-1):
            if i == 0: continue
            if direction[i]>0 and (direction[i-1] == 0 or direction[i-1] <0):
                # long_open
                buy.append(close[i])
                buy_day.append(i)
                plt.plot(i, close[i],'or')
            elif direction[i]==0 and direction[i-1]>0:
                # long_close
                plt.plot(i, close[i],'or')
                buy_day.append(i)
                buy.append(close[i])
                plt.plot(np.array(buy_day)[[0,-1]], np.array(buy)[[0,-1]], '-r',linewidth=1.8)
                pnl.append(buy[-1] - buy[0])
                buy, buy_day = [], []
                
            if direction[i]<0 and (direction[i-1] == 0 or direction[i-1] >0):
                # short_open
                plt.plot(i, close[i],'og')
                sell.append(close[i])
                sell_day.append(i)
                
            elif direction[i]==0 and direction[i-1]<0:
                # show_close
                plt.plot(i, close[i],'og')
                sell_day.append(i)
                sell.append(close[i])
                plt.plot(np.array(sell_day)[[0,-1]], np.array(sell)[[0,-1]], '-g',linewidth=1.8)
                pnl.append(sell[0] - sell[-1])
                sell, sell_day = [], []

        plt.grid(linewidth=0.1)
        plt.title(self.signal_name + ' ' + self.instrument_id + ' #' + str(self.stat_df['# trades'].values) +' trades  red:long, green:short')
        print 'pls wait ...'
        plt.savefig(path)
        #plt.plot(np.cumsum(pnl))
        #i#mport mpld3
        #mpld3.show(ip='192.168.168.168.11', port=8890)



    # ---- * 以下鲁棒性测试  * ----
    def Robustness_Test(self, all_test=None,\
                        monte_carlo_nums=10, \
                        confidence=0.95, \
                        shuffle_trades=False, \
                        skip_trades=False, skip_trades_prob = 0.05,\
                        random_param=False, random_param_prob=0.10, random_param_range=0.10,\
                        random_start=False, random_start_point=1000,\
                        random_history_data=False, random_history_data_prob=0.1, random_history_data_range=0.1,\
                        outlier_remove=False, outlier_remove_ratio = 0.05):
        '''
        蒙特卡洛的次数: 循环模拟的次数
        每次循环, 测试的优先级为:随机开始位置 > 随机价格 > 随机参数 > 跳过订单 > shuffle > 极端收益拉回
        confidence range: 对模拟结果的pnl进行分布拟合, 返回original pnl vs 50% ~ 100% 置信区间的结果
        '''
        if all_test is True: 
            shuffle_trades=True
            skip_trades=True
            random_param=True
            random_start=True
            random_history_data=True
            outlier_remove=True


        # 参数显示
        from prettytable import PrettyTable
        x = PrettyTable(['Category', 'Robust Test', ' ','  ','   ','    '])
        x.align['Category'] = 'l'
        x.align['Robust Test'] = 'l'
        x.align[' '] = 'l'
        x.align['  '] = 'l'
        x.align['   '] = 'l'
        x.align['    '] = 'l'
        x.padding_width = 1
        x.add_row(['monte_carlo_nums', monte_carlo_nums,'','','',''])
        x.add_row(['confidence', confidence,'','','',''])
        x.add_row(['shuffle_trades', shuffle_trades,'','','',''])
        x.add_row(['skip_trades',skip_trades,'prob',skip_trades_prob,'',''])
        x.add_row(['random_param', random_param, 'prob',random_param_prob, 'range', random_param_range])
        x.add_row(['random_start', random_start, 'max_start_point', random_start_point,'',''])
        x.add_row(['random_history_data', random_history_data, 'prob', random_history_data_prob, 'range',random_history_data_range])
        x.add_row(['outlier_remove', outlier_remove,'remove_ratio',outlier_remove_ratio,'',''])
        print x


        # 测试开始
        print 'Robust test begins...'
        # 记录策略原始pnl
        init_net_cpnl = self.net_cpnl#self.tranx_record_df['Cummulative Net P/L in money'].values.copy()
        init_cpnl = self.cpnl #self.tranx_record_df['Cummulative P/L in money'].values.copy()
        self.robust_net_pnl, self.robust_orignal_pnl = {}, {}
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        self.robust_df = pd.DataFrame(index=[''])
        self.robust_df['P/L in money'] = self.stat_df['Original profit $']
        self.robust_df['Conf'] = 'Original'
        self.robust_df['Max DD in money'] = self.stat_df['Max DD $']

        np.random.RandomState(10)
        for i in tqdm(xrange(monte_carlo_nums)):
            if random_start:
                start_point = np.random.randint(0, random_start_point, 1)[0]
                self.data_prepare(start_point=start_point, print_signal_info=False)
                #print >> sys.stdout, 'backtest date: ', self.s_date, ' - ', self.e_date

            if random_history_data:
                atr = ATR(self.ClosePrice, self.HighestPrice, self.LowestPrice, 14)
                loc = np.random.randint(0, len(atr), int(len(atr) * random_history_data_prob))
                # % and max price change % of ATR
                atr_change = atr * random_history_data_range * np.in1d(np.arange(len(atr)), loc)
                self.ClosePrice = atr_change + self.ClosePrice
                self.HighestPrice = atr_change + self.HighestPrice
                self.LowestPrice = atr_change + self.LowestPrice
                self.OpenPrice = atr_change + self.OpenPrice

            if random_param:
                self.robust_test_mode = True
                val1 = np.array([True, False])
                prob1 = [random_param_prob, 1-random_param_prob]
                from scipy.stats import rv_discrete
                self.random_param_TF_distrib = rv_discrete(values=(val1, prob1))
                # use: self.random_param_TF_distrib.rvs()
                self.random_history_data_range = random_history_data_range
                pass

            # rolling 
            if random_start or random_history_data or random_param:
                self.prebatch(print_info=False)
                self.work(process_bar=False, print_info=False)
                self.summary(print_summary=False)
                self.summary2(print_summary=False)

            net_pnl = self.net_pnl#self.tranx_record_df['Net P/L in money'].values.copy()
            original_pnl = self.pnl#self.tranx_record_df['P/L in money'].values.copy()

            if skip_trades:
                loc = np.random.randint(0, len(net_pnl), int(len(net_pnl) * skip_trades_prob))
                net_pnl = net_pnl[~np.in1d(np.arange(len(net_pnl)), loc)]
                loc = np.random.randint(0, len(original_pnl), int(len(original_pnl) * skip_trades_prob))
                original_pnl = original_pnl[~np.in1d(np.arange(len(original_pnl)), loc)]

            if shuffle_trades:    
                np.random.shuffle(net_pnl)
                np.random.shuffle(original_pnl)

            if outlier_remove:
                import scipy.stats as st
                net_pnl = st.mstats.winsorize(net_pnl, limits=0.02, axis=0).data  # 3倍标准差拉回
                original_pnl = st.mstats.winsorize(original_pnl, limits=0.02, axis=0).data  # 3倍标准差拉回

            # plot
            self.robust_net_pnl[i] = np.cumsum(net_pnl)
            self.robust_orignal_pnl[i] = np.cumsum(original_pnl)


        plt.figure(figsize=(15,12))
        plt.plot(init_cpnl, linewidth=1)
        for i in self.robust_orignal_pnl:
            plt.plot(self.robust_orignal_pnl[i], linewidth=0.3)
        plt.title('Robust test # ' + str(monte_carlo_nums) + ' nums')
        plt.ylabel("P/L in money")
        plt.grid()
        plt.show()


        plt.figure(figsize=(15,12))
        plt.plot(init_net_cpnl, linewidth=1)
        for i in self.robust_net_pnl:
            plt.plot(self.robust_net_pnl[i], linewidth=0.3)
        plt.title('Robust test # ' + str(monte_carlo_nums) + ' nums')
        plt.ylabel("Net P/L in money")
        plt.grid()
        plt.show()


    def time(self):
        # 函数调用花费的时间
        global execution_time_record_dict
        for i in execution_time_record_dict:
            print i, np.mean(execution_time_record_dict[i]), 's'


    # --- * 以下交易函数 * ----
    def No_position_is_open(self):
        # 没有下单指令, 返回T
        # 有限价单指令, 但是没有成交, 返回T
        return len(self.tranx_order) == 0 or self.tranx_order['Open deal'] == 'No'
        #return ~(len(self.tranx_order) > 0 and self.tranx_order['Open deal'] == 'Yes')

    def Position_is_open(self):
        # 已经开仓, 返回T
        return len(self.tranx_order)>0 and self.tranx_order['Open deal'] == 'Yes'

    def Buy_at_Limit(self, price):
        # 发起限价单
        price = self.PriceScale(price) 

        if self.robust_test_mode is True:
            if self.random_param_TF_distrib.rvs():
                rand = self.random_param_ratio_distrib() 
                price   = price + self.PriceScale(rand*price)
        self.Init_tranx_order(trade_type='Stop', direction=1, trade_price=price)

    def Sell_at_Limit(self, price):
        # 发起限价单
        price = self.PriceScale(price) 

        if self.robust_test_mode is True:
            if self.random_param_TF_distrib.rvs():
                rand = self.random_param_ratio_distrib() 
                price = price + self.PriceScale(rand*price)
        self.Init_tranx_order(trade_type='Stop', direction=-1, trade_price=price)

    def Close_position_at_Open(self):
        # 市价平仓使用开盘价
        if len(self.tranx_order) > 0 and self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['Close deal'] == 'No':
            self.tranx_order['Close type'] = 'OpenPrice'
            self.tranx_order['Close Init time'] = self.last_bar_init_time
        else:
            print self.tranx_order
            raise Exception('no holding position for closing action!')

    def Reverse_position_at_Open(self):
        # 市价平仓使用开盘价, 反向开仓使用开盘价
        if len(self.tranx_order) > 0 and self.tranx_order['Open deal'] == 'Yes':
            self.tranx_order['Reverse type'] = 'OpenPrice'
            self.tranx_order['Reverse deal'] = 'No'
            self.tranx_order['Close Init time'] = self.last_bar_init_time
        else:
            #print self.tranx_order, self.tranx_order['Open deal']
            raise Exception('no holding position for reversing action!')

    def Buy_on_Open(self):
        # 发起开盘市价多单
        self.Init_tranx_order(trade_type='OpenPrice', direction=1)

    def Sell_on_Open(self):
        # 发起开盘市价空单
        self.Init_tranx_order(trade_type='OpenPrice', direction=-1)

    def Limit_Bars_valid(self, bar_nums):
        # 限价单有效期
        # 当有新的订单, 其有效期会刷新老订单的有效期, 或者可以直接理解为旧订单会被覆盖
        # 设计这个的主要目的是, 订单长期无法成交, 失去了开仓的意义。
        # 第一版设计的时候， 老的限价单挂上去， 位置就锁定了， 新的订单只能在一下情况生效： 老订单成交， 老订单expired， 这并不适合交易逻辑， 所以被修改了
        if self.robust_test_mode is True:
            if self.random_param_TF_distrib.rvs():
                rand = self.random_param_ratio_distrib() 
                bar_nums = bar_nums + int(rand * bar_nums)
        self.tranx_order['expired'] = bar_nums

    def Pips(self, pips):
        # 外部调用, 当点位和价格进行组合的时候, 需要使用
        return self.PipsScale(pips)


    def Stop_Loss_pips(self, pips):
        # 定点止损
        # 输入参数pips 是指点位
        # 对应到不同品种, price = pips * minMove(指最小变动)
        # 目的是对于含有定点止损的策略, 更够不需要更改就适应不同minMove的品种
        if self.robust_test_mode is True:
            if self.random_param_TF_distrib.rvs():
                rand = self.random_param_ratio_distrib() 
                pips = pips + int(rand * pips)

        pips = self.PipsScale(pips) 
        #多头固定点止损
        if self.tranx_order['direction'] == 1:
            self.tranx_order['LongSL_pips'] = pips
        # 空头固定点止损
        elif self.tranx_order['direction'] == -1:
            self.tranx_order['ShortSL_pips'] = pips
        else:
            raise Exception('Stop loss condition error')    

    def Profit_Target_pips(self, pips):
        if self.robust_test_mode is True:
            if self.random_param_TF_distrib.rvs():
                rand = self.random_param_ratio_distrib() 
                pips = pips + int(rand * pips)

        pips = self.PipsScale(pips) 
        # 多头固定点止盈
        if self.tranx_order['direction'] == 1:
            self.tranx_order['LongPT_pips'] = pips
        # 空头固定点止盈
        elif self.tranx_order['direction'] == -1:
            self.tranx_order['ShortPT_pips'] = pips
        else:
            raise Exception('Profit_Target condition error')    

    def Stop_Loss_trailing(self, price):
        # 多头or空头移动止损
        # 和Stop_Loss_pips的区别是 trailing传入的是价格, Stop_Loss_pips给定的是pips点位
        # 两个函数都可以实时更新
        price = self.PriceScale(price) 
        if self.robust_test_mode is True:
            if self.random_param_TF_distrib.rvs():
                rand = self.random_param_ratio_distrib() 
                price = price + self.PriceScale(rand*price)

        if price<=0:
            raise Exception('Stop_Loss_trailing, error')
        if self.tranx_order['direction'] == 1:
            self.tranx_order['LongSL_trailing'] = max(self.tranx_order['LongSL_trailing'], price)  #更新
            if self.tranx_order['LongSL_trailing'] == 0 or self.tranx_order['LongSL_trailing'] < price:
                self.tranx_order['LongSL_trailing'] = min(self.tranx_order['Target Open price'], price)
        elif self.tranx_order['direction'] == -1:
            self.tranx_order['ShortSL_trailing'] = min(self.tranx_order['ShortSL_trailing'], price)  #更新
            if self.tranx_order['ShortSL_trailing'] == 0 or self.tranx_order['ShortSL_trailing'] > price:
                self.tranx_order['ShortSL_trailing'] = max(self.tranx_order['Target Open price'], price)
        else:
            raise Exception('SL condition error')

    def Profit_Target_trailing(self, price):
        # 多头or空头移动止赢
        price = self.PriceScale(price) 
        if self.robust_test_mode is True:
            if self.random_param_TF_distrib.rvs():
                rand = self.random_param_ratio_distrib() 
                price = price + self.PriceScale(rand*price)

        if price<=0:
            raise Exception('Profit_Target_trailing, error')
        if self.tranx_order['direction'] == 1:
            if self.tranx_order['LongPT_trailing'] == 0 or self.tranx_order['LongPT_trailing'] > price:
                self.tranx_order['LongPT_trailing'] = price
        elif self.tranx_order['direction'] == -1:
            if self.tranx_order['ShortPT_trailing'] == 0 or self.tranx_order['ShortPT_trailing'] < price:
                self.tranx_order['ShortPT_trailing'] = price
        else:
            raise Exception('PT condition error')

    def Bars_Since_Entry(self):
        # 下单有效时间 按照bar的个数计算
        return self.tranx_order['Bars_Since_Entry'] 

    def Position_is_Long(self):
        # 是否多头持仓
        if len(self.tranx_order) > 0 and self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['direction'] > 0:
            return True 
        else:
            return False

    def Position_is_Short(self):
        # 是否空头持仓
        if len(self.tranx_order) > 0 and self.tranx_order['Open deal'] == 'Yes' and self.tranx_order['direction'] < 0:
            return True 
        else:
            return False

    def Move_Stop_Loss_to_Break_Even(self):
        # 止损线调整到开仓成本
        if self.tranx_order['direction'] == 1:
            #if self.tranx_order['LongSL_trailing'] == 0 or self.tranx_order['LongSL_trailing'] < self.tranx_order['Open price']:
            self.tranx_order['LongSL_trailing'] = self.tranx_order['Target Open price']
        elif self.tranx_order['direction'] == -1:
            #if self.tranx_order['ShortSL_trailing'] == 0 or self.tranx_order['ShortSL_trailing'] > self.tranx_order['Open price']:
            self.tranx_order['ShortSL_trailing'] = self.tranx_order['Target Open price']
        else:
            raise Exception('Move_Stop_Loss_to_Break_Even error')

    def Actual_profit(self):
        # 返回已开仓位的实时累计收益
        return self.tranx_order['actual_profit'] if len(self.tranx_order) > 0 else 0

    def Last_bar_profit(self):
        # 返回最近的一个bar的收益
        return self.last_bar_profit

    def EntryPrice(self):
        if len(self.tranx_order) > 0 and self.tranx_order['Open deal'] == 'Yes':
            return self.tranx_order['Target Open price']
        else:
            raise('no holding position')

