# -*- coding:utf-8 -*-
from Nan import Nan
from numpy import random
random.seed(20170605)
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import datetime

class FORCE_Optimization():
    # ---- * Simple Auto 参数寻优 * ----
    def __init__(self, signal, default_param, param_range, target_func='Ret/DD', max_result_store_nums=100):
        # Forcible optimization
        # 策略里面的参数命名规则: self.param1, self.param2 ...
        # default_param 为参数默认值的字典 default_param = {'param1': 50, 'param2': 70}, 默认参数应该包含所有参数值
        # param_range={'param1':{'Start':10, 'Stop': 90, 'Step':10, 'Type': 'int'}, 'float'...}
        # test_nums 为参数寻优测试的回测次数
        # target_func(best selection criteria): 默认'Ret/DD', 其他有NetProfit, Stability, Win/Loss
        # max_result_store_nums 最大结果存储数
        pass

class GE_Optimization():
    def __init__(self, signal, target_func='Ret/DD Ratio', max_result_store_nums=100,
        generation=100, popsize=10, chrosize=10,crossrate=1., mutationrate=0.1):
        '''
        Genetic optimization
        策略里面的参数命名规则: self.param1, self.param2 ...
        #default_param 为参数默认值的字典 default_param = {'param1': 50, 'param2': 70}, 默认参数应该包含所有参数值
        #param_range={'param1':{'Start':10, 'Stop': 90, 'Step':10, 'Type': 'int'}, 'float'...}
        test_nums 为参数寻优测试的回测次数
        target_func(best selection criteria): 默认'Ret/DD Ratio', 其他有'Net profit $', 'Stability', 'Win/Loss Ratio', 'Max DD $', '# trades'
        max_result_store_nums 最大结果存储数
        generation: 遗传代数
        popsize:    种群大小
        chrosize:   编码长度
        crossrate:  交叉率
        mutationrate: 变异率
        * 算法默认使用了单点交叉, 赌轮盘选择, 精英策略 
        * 外部使用run() 执行和 plot() 绘制收敛图
        '''
        self.signal = signal
        self.target_func = target_func
        self.max_result_store_nums = max_result_store_nums
        # 参数
        self.default_param, self.param_range = {}, {}
        for param in self.signal.param_dict:
            self.default_param[param] = self.signal.param_dict[param]['Default']
            self.param_range[param] = self.signal.param_dict[param].copy()

        self.generation = generation
        self.popsize =popsize
        self.chrosize =chrosize
        self.crossrate =1
        self.mutationrate =0.01

    # 按照指定的参数类型转换
    def convert_param_type(self, param_dict):
        for j in param_dict:
            if 'Type' not in self.param_range[j]:
                raise Exception('no type of param_dict')
            if self.param_range[j]['Type'] == 'int':
                param_dict[j] = [int(i) for i in param_dict[j]]
            elif self.param_range[j]['Type'] == 'float':
                param_dict[j] = [float(i) for i in param_dict[j]]
            else:
                raise Exception('no type of param_dict')
        return param_dict

    # 目标函数, 返回值越大, 适应性越好
    def func(self, param_dict):
        for k in self.default_param:
            setattr(self.signal, k, self.default_param[k])
        for j in param_dict:
            setattr(self.signal, j, param_dict[j])  
        Na = Nan(self.signal)
        Na.run(print_info=False, print_signal_info=False, plot=False, summary1=False, summary2=False, process_bar=False)
        return Na.stat_df


    # 计算过的就记录下来, 同样参数, 就跳过, 不用再计算一遍了
    def efficient(self, param_dict):
        for i in self.efficient_dict:
            efficient_param_dict = self.efficient_dict[i]['param']
            efficient_param_values = np.array([efficient_param_dict.get(j, np.nan) for j in param_dict.keys()])
            if np.sum(efficient_param_values - np.array(param_dict.values())) == 0:
                return self.efficient_dict[i]['stat_df'].copy()
        stat_df = self.func(param_dict)
        id_ = str(len(self.efficient_dict)+1)
        self.efficient_dict[id_]['param'] = param_dict
        self.efficient_dict[id_]['stat_df'] = stat_df
        return stat_df

    def initialpop(self):
        pop = random.randint(0,2,size =(self.popsize,self.chrosize))
        return pop


    #由于我们所求的是最大值，所有可以用函数值代替适应度
    def get_fitness(self, param_dict):     
        fitness = []
        self.signal_result_record = []
        for i in range(len(param_dict[param_dict.keys()[0]])):
            param_once_dict = {}
            for j in param_dict:
                param_once_dict[j] = param_dict[j][i]
            stat_df = self.efficient(param_once_dict)
            fitness.append(stat_df[self.target_func].values[0])
            self.signal_result_record.append(stat_df)
        return fitness


    #输入参数为上一代的种群，和上一代种群的适应度列表
    def selection(self,popsel,fitvalue):
        # 这里使用了轮盘赌选择（Roulette Wheel Selection ）
        # 如果使用排序选择算法, 需要rank值来替代fitvalue的目标函数值
        # "排序选择将使所有个体都有将会被选择。但是这样也会导致种群不容易收敛，因为最好的个体与其他个体的差别减小。"
        new_fitvalue = []
        totalfit = sum(fitvalue)
        accumulator = 0.0
        for val in fitvalue: 
            #对每一个适应度除以总适应度，然后累加，这样可以使适应度大
            #的个体获得更大的比例空间。
            new_val =(val*1.0/totalfit)            
            accumulator += new_val
            new_fitvalue.append(accumulator)            
        ms = []
        for i in xrange(self.popsize):
            #随机生成0,1之间的随机数
            ms.append(random.random()) 
        ms.sort() #对随机数进行排序
        fitin = 0
        newin = 0
        newpop = popsel
        while newin < self.popsize:
            #随机投掷，选择落入个体所占轮盘空间的个体
            if(ms[newin] < new_fitvalue[fitin]):
                newpop[newin] = popsel[fitin]
                newin = newin + 1
            else:
                fitin = fitin + 1
        #适应度大的个体会被选择的概率较大
        #使得新种群中，会有重复的较优个体
        pop = newpop
        return pop
    
    def crossover(self,pop):
        # 这里使用单点交叉
        # 对于两点交叉, 均匀交叉和算术交叉, TODO
        for i in xrange(self.popsize-1):
            #近邻个体交叉，若随机数小于交叉率
            if(random.random()<self.crossrate):
                #随机选择交叉点
                singpoint =random.randint(0,self.chrosize)
                temp1 = []
                temp2 = []
                #对个体进行切片，重组
                temp1.extend(pop[i][0:singpoint])
                temp1.extend(pop[i+1][singpoint:self.chrosize])
                temp2.extend(pop[i+1][0:singpoint])
                temp2.extend(pop[i][singpoint:self.chrosize])
                pop[i]=temp1
                pop[i+1]=temp2
        return pop 
    
    def mutation(self,pop):
        for i in xrange(self.popsize):
            #反转变异，随机数小于变异率，进行变异
            if (random.random()< self.mutationrate):
                mpoint = random.randint(0,self.chrosize-1)
                #将随机点上的基因进行反转。
                if(pop[i][mpoint]==1):
                    pop[i][mpoint] = 0
                else:
                    pop[mpoint] =1
        return pop
    

    def elitism(self,pop,popbest,nextbestfit,fitbest,nextfitvalue):
        #输入参数为上一代最优个体，变异之后的种群，
        #上一代的最优适应度，本代最优适应度。这些变量是在主函数中生成的。
        if nextbestfit-fitbest <0:  
            #满足精英策略后，找到最差个体的索引，进行替换。
            pop_worst =nextfitvalue.index(min(nextfitvalue))
            for j in pop:
                pop[j][pop_worst] = popbest[j]
        return pop
    
    
    #对十进制进行转换到求解空间中的数值
    def get_declist(self,chroms, xrangemax, xrangemin):
        step =(xrangemax - xrangemin)/float(2**self.chrosize-1)
        chroms_declist =[]
        for i in xrange(self.popsize):
            chrom_dec =xrangemin+step*self.chromtodec(chroms[i])  
            chroms_declist.append(chrom_dec)
        return chroms_declist
    
    
     #将二进制数组转化为十进制
    def chromtodec(self,chrom):
        m = 1
        r = 0
        for i in xrange(self.chrosize):
            r = r + m * chrom[i]
            m = m * 2
        return r


    def before_optimization(self, IS_start_date, IS_end_date, OOS_date):
        if self.watch_mode: print '************************ Before optimization ************************'
        if IS_start_date is not None and IS_end_date is not None and OOS_date is not None:
            if self.watch_mode: print 'IS start', IS_start_date, 'IS end', IS_end_date, 'OOS_date', OOS_date 
            self.signal.signal_config['Macro']['Start'] = IS_start_date
            self.signal.signal_config['Macro']['End'] = OOS_date
        for k in self.default_param:
            setattr(self.signal, k, self.default_param[k])
            print 'signal use ', k, getattr(self.signal, k)
        Na = Nan(self.signal)
        Na.run(print_info=False, print_signal_info=self.watch_mode, summary1=False, process_bar=False, plot1=False, plot2=self.watch_mode, IS_end_date=IS_end_date)


    def after_optimization(self, IS_start_date, IS_end_date, OOS_date):
        if self.watch_mode: print '************************ After optimization ************************'
        if IS_start_date is not None and IS_end_date is not None and OOS_date is not None:
            if self.watch_mode: print 'IS start', IS_start_date, 'IS end', IS_end_date, 'OOS_date', OOS_date 
            self.signal.signal_config['Macro']['Start'] = IS_start_date
            self.signal.signal_config['Macro']['End'] = OOS_date
        param_dict = self.param_best_record[-1]
        print 'default param:', self.default_param
        print 'best param:', param_dict
        for j in param_dict:
            setattr(self.signal, j, param_dict[j])  
            print 'signal use ', j, getattr(self.signal, j)
        Na = Nan(self.signal)
        Na.run(print_info=False, print_signal_info=False, summary1=False, process_bar=False, plot1=False, plot2=self.watch_mode, IS_end_date=IS_end_date)
        optimization_record = {'stat_df': Na.stat_df,
                                'net_cpnl': Na.net_cpnl, 
                                'cpnl':Na.cpnl,
                                'tranx_record_df': Na.tranx_record_df,
                                'rolling_direction': Na.rolling_direction,
                                'rolling_dates': Na.rolling_dates}
        return optimization_record


    def during_optimization(self, speed_mode=True, elite_mode=True, watch_mode=False, IS_start_date=None, IS_end_date=None):
        print '************************ During optimization ************************'
        if IS_start_date is not None and IS_end_date is not None:
            print 'IS start', IS_start_date, 'IS end', IS_end_date
            self.signal.signal_config['Macro']['Start'] = IS_start_date
            self.signal.signal_config['Macro']['End'] = IS_end_date

        import collections
        def tree():
            return collections.defaultdict(tree)
        self.efficient_dict = tree()
            
        pop = {}
        time_record = []
        self.fit_components_record = {'Ret/DD Ratio':[],'Net profit $':[], 'Stability':[], 'Win/Loss Ratio':[], 'Max DD $':[], '# trades':[]}

        for j in self.param_range:
            pop[j] =self.initialpop()  #种群初始化

        declist, nextdeclist = {}, {}
        self.fit_best_record, self.pop_best_record, self.param_best_record = [], [], [] #每代最优个体对应的适应值, 二进制, 十进制
        popbest, parambest = {}, {}
        from tqdm import tqdm
        self.interation = 1
        for i in tqdm(xrange(self.generation), desc='iteration total ' + str(self.generation)):
            time_record.append(pd.to_datetime(datetime.datetime.now()))
            if len(time_record) == 2:
                finish_time = self.generation*np.mean(np.diff(time_record))
                print 'finish_time:', str(finish_time), 'target_time', str(pd.to_datetime(datetime.datetime.now()) + finish_time)[:19]
            self.interation = i+1
            #在遗传代数内进行迭代
            for j in self.param_range:
                declist[j] =self.get_declist(pop[j], self.param_range[j]['Start'], self.param_range[j]['Stop'])#解码
            declist = self.convert_param_type(declist)
            fitvalue = self.get_fitness(declist)#适应度函数
            loc = fitvalue.index(max(fitvalue))#最高适应度的位置
            #选择适应度函数最高个体
            for j in self.param_range:
                popbest[j] = pop[j][loc]
                #对popbest进行深复制，以为后面精英选择做准备
                popbest[j] =copy.deepcopy(popbest[j])
                parambest[j]= declist[j][loc]
                #最高适应度
            fitbest = max(fitvalue)
            #if watch_mode is True: print declist, 'best', parambest
            #保存每代最高适应度值
            self.fit_best_record.append(fitbest)
            self.pop_best_record.append(popbest.copy())
            self.param_best_record.append(parambest.copy())
            # 记录需要观察的其他指标
            for compon in self.fit_components_record:
                self.fit_components_record[compon].append(self.signal_result_record[loc][compon].values[0])
            ################################ 快速模式, 在某一代参数全部相同, 迭代结束条件
            if speed_mode is True:
                cond = []
                for j in declist:
                    cond.append(len(np.unique(declist[j])))  
                if len(np.unique(cond)) == 1 and cond[0] == 1: 
                    #print 'iteration satisfy condition and finish', declist, 'best', parambest
                    break
            ################################进行算子操作，并不断更新pop
            for j in self.param_range:
                pop[j] = self.selection(pop[j],fitvalue)  #选择
                #print 'choose', pop[j]
                pop[j] = self.crossover(pop[j]) # 交叉
                #print 'crossover',pop[j]
                pop[j] = self.mutation(pop[j])  #变异
                #print 'mutation', pop[j]

            if elite_mode is False: continue
            ################################精英策略前的准备
            #对变异之后的pop，求解最大适应度
            for j in self.param_range:
                nextdeclist[j] = self.get_declist(pop[j], self.param_range[j]['Start'], self.param_range[j]['Stop']) 
            nextfitvalue =self.get_fitness(nextdeclist)        
            nextbestfit = max(nextfitvalue)
            ################################精英策略
            #比较深复制的个体适应度和变异之后的适应度
            pop = self.elitism(pop,popbest,nextbestfit,fitbest,nextfitvalue)



    def run(self, speed_mode=True, elite_mode=True, watch_mode=True, IS_start_date=None, IS_end_date=None, OOS_date=None):
        '''
        优化过程, 外部调用, 不使用精英策略, 优化速度会节省一倍
        '''
        self.watch_mode = watch_mode #观察模式
        ################################ 优化前表现
        if watch_mode: self.before_optimization(IS_start_date, IS_end_date, OOS_date)
        ################################ 优化中
        self.during_optimization(speed_mode=speed_mode, elite_mode=elite_mode, watch_mode=watch_mode, IS_start_date=IS_start_date, IS_end_date=IS_end_date)
        ################################ 优化后
        optimization_record = self.after_optimization(IS_start_date, IS_end_date, OOS_date)
        if watch_mode: self.plot()     # 优化过程中, 策略表现
        return optimization_record



    def plot(self):
        # if len(self.param_range) == 2:
        #     # 3D for two params
        #     x = np.linspace(-2,2)
        #     y= np.linspace(-2,2)
        #     x, y = np.meshgrid(x, y)
        #     z = y*np.sin(2*pi*x)+x*np.cos(2*pi*y)
        #     fig = plt.figure(figsize=(30,12))
        #     ax = fig.add_subplot(1, 2, 1, projection='3d')
        #     from matplotlib.colors import LightSource
        #     from matplotlib import cm
        #     ls = LightSource(270, 45)
        #     # To use a custom hillshading mode, override the built-in shading and pass
        #     # in the rgb colors of the shaded surface calculated from "shade".
        #     rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        #     ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
        #                     linewidth=0, antialiased=False, shade=False)

        #     bx = fig.add_subplot(1,2,2)
        #     t = [x for x in xrange(self.generation)]
        #     s = self.pop_best_record
        #     bx.plot(t,s)
        #     bx.set_xlabel('generation')
        #     bx.set_ylabel('optimization')
        #     plt.show()
        # else:
        # 目标函数迭代的表现
        ax = plt.figure(figsize=(16,2))
        t = [x for x in xrange(len(self.fit_best_record))]
        s = self.fit_best_record
        plt.plot(t,s)
        plt.xlabel('optimization iteration')
        plt.ylabel(self.target_func)
        plt.show()


        net_profit, trades, stability,  max_dd, win_loss_ratio, max_dd_time = [], [], [], [], [], []
        for param in self.param_best_record:
            for i in self.efficient_dict:
                if self.efficient_dict[i]['param'] == param:
                    net_profit.append(self.efficient_dict[i]['stat_df']['Net profit $'].values[0])
                    trades.append(self.efficient_dict[i]['stat_df']['# trades'].values[0])
                    stability.append(self.efficient_dict[i]['stat_df']['Stability'].values[0])
                    max_dd.append(self.efficient_dict[i]['stat_df']['Max DD $'].values[0])
                    max_dd_time.append(self.efficient_dict[i]['stat_df']['DD period'].values[0])
                    win_loss_ratio.append(self.efficient_dict[i]['stat_df']['Win/Loss Ratio'].values[0])

        ax = plt.figure(figsize=(16,2))
        t = [x for x in xrange(len(net_profit))]
        s = net_profit
        plt.plot(t,s)
        plt.xlabel('optimization iteration')
        plt.ylabel('Net profit $')
        plt.show()

        ax = plt.figure(figsize=(16,2))
        t = [x for x in xrange(len(trades))]
        s = trades
        plt.plot(t,s)
        plt.xlabel('optimization iteration')
        plt.ylabel('# trades')
        plt.show()

        ax = plt.figure(figsize=(16,2))
        t = [x for x in xrange(len(stability))]
        s = stability
        plt.plot(t,s)
        plt.xlabel('optimization iteration')
        plt.ylabel('Stability')
        plt.show()

        ax = plt.figure(figsize=(16,2))
        t = [x for x in xrange(len(max_dd))]
        s = max_dd
        plt.plot(t,s)
        plt.xlabel('optimization iteration')
        plt.ylabel('Max DD $')
        plt.show()

        ax = plt.figure(figsize=(16,2))
        t = [x for x in xrange(len(max_dd_time))]
        s = max_dd_time
        plt.plot(t,s)
        plt.xlabel('optimization iteration')
        plt.ylabel('DD period')
        plt.show()
                

        ax = plt.figure(figsize=(16,2))
        t = [x for x in xrange(len(win_loss_ratio))]
        s = win_loss_ratio
        plt.plot(t,s)
        plt.xlabel('optimization iteration')
        plt.ylabel('Win/Loss Ratio')
        plt.show()



# ---- * Walk Forward 优化 * ----
class Walk_Forward_Optimization():
    def __init__(self, signal, walk_nums=5, run_period_ratio=0.3, optimizor=None):
        '''
                                                ----------------------------------------------------
                                                |     optimization period 3    |  run period 3     |
                                                ----------------------------------------------------
                            ----------------------------------------------------
                            |     optimization period 2    |  run period 2     |
                            ----------------------------------------------------
        ----------------------------------------------------
        |     optimization period 1    |  run period 1     |
        ----------------------------------------------------
        walk_nums: 向前行走步数, 上图为3步, 建议使用5步以上
        run_period_optimization_period_ratio: 每一步的实盘样本占整个步长(优化和实盘)的比例, 默认0.3, 则optimization period: run period = 7:3
        * optimizor 优化算子的参数, 请在optimizor上设置
        * 外部调用run() 执行
        '''
        self.walk_nums = walk_nums
        self.run_period_ratio = run_period_ratio
        self.signal = signal
        if optimizor is None: raise('Should point out an optimizor')
        self.optimizor = optimizor

        # 确定总的回测时间
        data_path = signal.signal_config['Data']['Inpath']
        start = str(signal.signal_config['Macro']['Start'])
        end = str(signal.signal_config['Macro']['End'])
        maxlookback = signal.signal_config['Macro']['Maxlookback']
        data_set = signal.signal_config['data_set']
        raw_df = pd.read_csv(data_path, names=['OpenPrice', 'HighestPrice', 'LowestPrice', 'ClosePrice', 'Volume', 'Position'])
        raw_dates = raw_df.index.values
        raw_data_start = raw_dates[0]
        raw_data_end = raw_dates[-1]
        if data_set == 'is':
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            #raw_dates_parse = np.array([dt_parser.parse(i) for i in raw_dates])
            raw_dates_index = pd.to_datetime(raw_dates.astype(str))
            if start < pd.to_datetime(raw_dates[maxlookback]):
                #print >> sys.stdout, "warning! Start date besides data length! pls use -A for all sample"
                sidx = maxlookback
            else:
                sidx = np.where(raw_dates_index <= start)[0][-1]
            if end > pd.to_datetime(raw_dates[-1]):
                eidx = raw_dates.shape[0] - 1
                #raise Exception("end date besides data length!")
            else:
                eidx = np.where(raw_dates_index >= end)[0][0]

        elif data_set == 'all':
            sidx = maxlookback
            eidx = raw_dates.shape[0] - 1



        backtest_len = eidx - sidx - 2
        dates = raw_df.index.values

        # 确定每一步的回测日期区间
        self.run_period_len = (backtest_len-15)/((1/run_period_ratio-1) + walk_nums)
        self.optimization_period_len = int(self.run_period_len/run_period_ratio * (1-run_period_ratio))
        self.run_period_len = int(self.run_period_len)
        print "self.optimization_period_len:", self.optimization_period_len,  "self.run_period_len:", self.run_period_len
        self.walk_forward_point, self.walk_forward_dates = {}, {}
        for i in range(walk_nums):
            start_point = sidx
            end_point = sidx + self.optimization_period_len + self.run_period_len
            self.walk_forward_point[i] = {'IS_start': start_point, 'IS_end': sidx + self.optimization_period_len, 'OOS':end_point}
            self.walk_forward_dates[i] = {'IS_start': dates[start_point],  'IS_end': dates[sidx + self.optimization_period_len], 'OOS': dates[end_point]}
            sidx += self.run_period_len + 1
        for i in self.walk_forward_dates:
            print 'step' + str(i), self.walk_forward_dates[i]
        return 


    def run(self, watch_mode=True, elite_mode=True):
        '''
        使用指定优化算子进行walk forward backtest
        '''
        self.stat_df_record = pd.DataFrame()
        self.cpnl_record, self.net_cpnl_record, self.tranx_df_record, self.direction_record, self.dates_record = [], [], [], [], []
        self.optimization_record = {}
        time_start = pd.to_datetime(datetime.datetime.now())
        for i in range(self.walk_nums):
            print '===================================================================================='
            print '                             Walk Forwad Step ' + str(i+1) 
            print '===================================================================================='
            ####################################### 使用指定最优算法优化参数
            optimization_record = self.optimizor.run(watch_mode=watch_mode, elite_mode=elite_mode, IS_start_date=self.walk_forward_dates[i]['IS_start'], IS_end_date=self.walk_forward_dates[i]['IS_end'], OOS_date=self.walk_forward_dates[i]['OOS'])
            self.stat_df_record = self.stat_df_record.append(optimization_record['stat_df'])
            self.cpnl_record.append(optimization_record['cpnl'])
            self.net_cpnl_record.append(optimization_record['net_cpnl'])
            self.tranx_df_record.append(optimization_record['tranx_record_df'])
            self.direction_record.append(optimization_record['rolling_direction'])
            self.dates_record.append(optimization_record['rolling_dates'])
            self.optimization_record[i] = optimization_record
        # 消耗时间
        time_end = pd.to_datetime(datetime.datetime.now())
        if watch_mode is True:
            print 'Time totally consumes', str(time_end - time_start)
        self.plot()


    def plot(self):
        # 绘图
        def Rsquared(y):
            from scipy.stats import linregress
            """ Return R^2 where x and y are array-like."""
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            return r_value**2

        def stat_Ddp(cpnl):
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

        def stat_drawdown(cpnl):
            dd = cpnl - np.array([np.nanmax(cpnl[:i + 1]) for i in xrange(cpnl.shape[0])])
            return dd


        aggr_net_cpnl = self.net_cpnl_record[0][:]
        aggr_cpnl = self.cpnl_record[0][:]
        aggr_tranx_record = self.tranx_df_record[0][:]
        rolling_dates = self.dates_record[0][:]
        for i in range(1, self.walk_nums):
            tmp_net_cpnl = np.array(self.net_cpnl_record[i][self.optimization_period_len:])
            aggr_net_cpnl = np.append(aggr_net_cpnl, tmp_net_cpnl - tmp_net_cpnl[0] + aggr_net_cpnl[-1])
            tmp_cpnl = np.array(self.cpnl_record[i][self.optimization_period_len:])
            aggr_cpnl = np.append(aggr_cpnl, tmp_cpnl - tmp_cpnl[0] + aggr_cpnl[-1])
            loc = np.where(self.tranx_df_record[i]['Close time'].values>=self.walk_forward_dates[i]['IS_end'])[0][0]
            aggr_tranx_record = aggr_tranx_record.append(self.tranx_df_record[i][:loc])
            rolling_dates = np.append(rolling_dates, self.dates_record[i][self.optimization_period_len:])


        self.stat_df = pd.DataFrame(index=[''])
        self.stat_df['InstrumentId'] = self.signal.signal_config['Benchmark']
        self.stat_df['Original profit $'] =  aggr_cpnl[-1] 
        self.stat_df['Net profit $'] = aggr_net_cpnl[-1]
        self.stat_df['# trades'] = aggr_tranx_record.shape[0]
        self.stat_df['Stability'] = round(Rsquared(aggr_net_cpnl), 2)
        self.stat_df['DD period'] = stat_Ddp(aggr_net_cpnl)
        draw_down = stat_drawdown(aggr_net_cpnl)
        self.stat_df['Max DD $'] = np.max(np.abs(draw_down))
        pnl = np.diff(aggr_net_cpnl)
        self.stat_df['Ret/DD Ratio'] = round(self.stat_df['Net profit $']/self.stat_df['Max DD $'], 1)
        self.stat_df['Win/Loss Ratio'] = round(abs(np.sum(pnl[pnl>0])/np.sum(pnl[pnl<0])), 1)
        # self.stat_df['Buy profit'] = np.sum(pnl[self.rolling_direction[self.maxlookback:]==1])
        # self.stat_df['Sell profit'] = np.sum(pnl[self.rolling_direction[self.maxlookback:]==-1])
        # self.stat_df['Avg win $'] = round(np.mean(pnl[self.rolling_direction[self.maxlookback:]==1]), 1)
        # self.stat_df['Avg loss $'] = round(np.mean(pnl[self.rolling_direction[self.maxlookback:]==-1]), 1)
        # self.stat_df['Avg honding bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values), 1)
        # self.stat_df['Avg honding win bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Type'].values=='Long']), 1)
        # self.stat_df['Avg honding loss bars'] = round(np.mean(self.tranx_record_df['Bars_Since_Entry'].values[self.tranx_record_df['Type'].values=='Short']), 1)


        # WF 滑动曲线
        plt.figure(figsize=(15,12))
        rect1 = [0.05, 0.225, 0.9, 0.8]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
        rect2 = [0.05, 0.025, 0.9, 0.17]
        sub1 = plt.axes(rect1)
        sub2 = plt.axes(rect2)
        #ax = plt.add_subplot(2,1,1)
        #bx = plt.add_subplot(2,1,2)
        init_point = self.walk_forward_point[0]['IS_start']
        for i in range(self.walk_nums):
            x = np.arange(self.walk_forward_point[i]['IS_start']-init_point, self.walk_forward_point[i]['OOS']-init_point)
            #plt.plot(x, self.cpnl[i], label='Original')
            sub1.plot(x, self.net_cpnl_record[i][:len(x)], label='step' + str(i+1), linewidth=1)
            # direction line
            y = self.net_cpnl_record[i][-1] * (x-x[0])/(x[-1]-x[0])
            sub1.plot(x, y, color='r')
            x = np.arange(self.walk_forward_point[i]['IS_start']-init_point, self.walk_forward_point[i]['IS_end']-init_point)
            sub2.plot(x, i*np.ones_like(x), linewidth=10, color='r', alpha=0.5)
            x = np.arange(self.walk_forward_point[i]['IS_end']-init_point, self.walk_forward_point[i]['OOS']-init_point)
            sub2.plot(x, i*np.ones_like(x), linewidth=10, color='b', alpha=0.5)

        plt.sca(sub1)
        start = self.optimization_period_len
        for i in range(self.walk_nums):
            end = self.run_period_len + start
            p = plt.axvspan(start, end, edgecolor='red', facecolor='grey', linewidth=1.5, alpha=0.1)
            start += self.run_period_len

        plt.sca(sub1)
        plt.ylabel("Net Profit $")
        plt.legend(loc='upper left')
        plt.title(self.signal.signal_config['Name'] + ' ' + self.signal.signal_config['Benchmark'])#E + ' # ' + str(self.tranx_record_df.shape[0]) + ' trades')
        plt.sca(sub2)
        plt.ylabel("Step")


        dates_str = np.array(rolling_dates)
        if '.' in dates_str[0]:
            split_str = '.'
        elif '/' in dates_str[0]:
            split_str = '/'
        else:
            raise Exception('cannot recognise date string!')
        dates_short_str = np.array([i.split(split_str)[0] for i in dates_str])  # x轴为时间等距
        if len(np.unique(dates_short_str))<3:   # 优化x坐标轴显示
            dates_short_str = np.array([i.split(split_str)[0] + split_str + i.split(split_str)[1] for i in dates_str])             
        space = np.array([np.where(dates_short_str==i)[0][0] for i in np.unique(dates_short_str)])
        #dates_str = np.array([i[:7] for i in dates_str])  # 直接显示时间
        #space = np.int32(np.linspace(0, dates_str.shape[0] - 1, 9))
        plt.xticks(space, dates_str[space])
        plt.show()


        # WF组合收益图
        plt.figure(figsize=(16,12))
        rect1 = [0.05, 0.225, 0.9, 0.8]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
        rect2 = [0.05, 0.025, 0.9, 0.17]
        sub1 = plt.axes(rect1)
        sub2 = plt.axes(rect2)

        plt.sca(sub1)
        plt.plot(aggr_cpnl, label='aggregated pnl', linewidth=1)
        plt.plot(aggr_net_cpnl, label='aggregated net pnl', linewidth=1)
        start = self.optimization_period_len
        for i in range(self.walk_nums):
            end = self.run_period_len + start
            p = plt.axvspan(start, end, edgecolor='red', facecolor='grey', linewidth=1.5, alpha=0.1)
            start += self.run_period_len
        x = np.arange(len(aggr_net_cpnl))     
        y = aggr_net_cpnl[-1] * (x-x[0])/(x[-1]-x[0])
        plt.plot(y, color='r')

        plt.legend(loc='upper left')
        plt.ylabel("Net Profit $")
        plt.title(self.signal.signal_config['Name'] + ' ' + self.signal.signal_config['Benchmark'])#E + ' # ' + str(self.tranx_record_df.shape[0]) + ' trades')


        plt.sca(sub2)
        plt.bar(np.arange(draw_down.shape[0]), draw_down, width=0.025, edgecolor='red', color='red', alpha=0.1)
        #plt.xticks(space, np.arange(draw_down.shape[0])[space])
        plt.ylabel("Drawdown $")
        plt.grid()


        # 对比默认参数的策略表现
        for k in self.optimizor.default_param:
            setattr(self.signal, k, self.optimizor.default_param[k])
        self.signal.signal_config['Macro']['Start'] = self.walk_forward_dates[0]['IS_start']
        self.signal.signal_config['Macro']['End'] = self.walk_forward_dates[self.walk_nums-1]['OOS']
        Na = Nan(self.signal)
        Na.run(print_info=False, print_signal_info=False, plot=False, summary1=False, summary2=False, process_bar=False)
        plt.sca(sub1)
        plt.plot(Na.cpnl, color='grey', label='default param pnl')
        plt.plot(Na.net_cpnl, color='grey', label='default param net pnl')

        dates_str = np.array(rolling_dates)
        if '.' in dates_str[0]:
            split_str = '.'
        elif '/' in dates_str[0]:
            split_str = '/'
        else:
            raise Exception('cannot recognise date string!')
        dates_short_str = np.array([i.split(split_str)[0] for i in dates_str])  # x轴为时间等距
        if len(np.unique(dates_short_str))<3:   # 优化x坐标轴显示
            dates_short_str = np.array([i.split(split_str)[0] + split_str + i.split(split_str)[1] for i in dates_str])             
        space = np.array([np.where(dates_short_str==i)[0][0] for i in np.unique(dates_short_str)])
        #dates_str = np.array([i[:7] for i in dates_str])  # 直接显示时间
        #space = np.int32(np.linspace(0, dates_str.shape[0] - 1, 9))
        plt.xticks(space, dates_str[space])
        plt.show()




        print '================================ After WF =========================================='
        print 'Summary (time axis):'
        from prettytable import PrettyTable 
        x = PrettyTable(list(self.stat_df.columns.values[:5]))
        x.add_row(list(self.stat_df.iloc[0,:].values[0:5]))
        print x
        x = PrettyTable(list(self.stat_df.columns.values[5:9]))
        x.add_row(list(self.stat_df.iloc[0,:].values[5:9]))
        print x
        # x = PrettyTable(list(self.stat_df.columns.values[9:]))
        # x.add_row(list(self.stat_df.iloc[0,:].values[9:]))
        # print x
        print '================================= Before WF =========================================='
        print 'Summary (time axis):'
        from prettytable import PrettyTable 
        x = PrettyTable(list(Na.stat_df.columns.values[:5]))
        x.add_row(list(Na.stat_df.iloc[0,:].values[0:5]))
        print x
        x = PrettyTable(list(Na.stat_df.columns.values[5:9]))
        x.add_row(list(Na.stat_df.iloc[0,:].values[5:9]))
        print x
        x = PrettyTable(list(Na.stat_df.columns.values[9:13]))
        x.add_row(list(Na.stat_df.iloc[0,:].values[9:13]))
        print x
        x = PrettyTable(list(Na.stat_df.columns.values[13:16]))
        x.add_row(list(Na.stat_df.iloc[0,:].values[13:16]))
        print x
        x = PrettyTable(list(Na.stat_df.columns.values[16:]))
        x.add_row(list(Na.stat_df.iloc[0,:].values[16:]))
        print x



        #dates_str = self.rolling_dates[self.maxlookback:].astype(str)
            # if '.' in dates_str[0]:
            #     split_str = '.'
            # elif '/' in dates_str[0]:
            #     split_str = '/'
            # else:
            #     raise Exception('cannot recognise date string!')
            # dates_short_str = np.array([i.split(split_str)[0] for i in dates_str])  # x轴为时间等距
            # if len(np.unique(dates_short_str))<3:   # 优化x坐标轴显示
            #     dates_short_str = np.array([i.split(split_str)[0] + split_str + i.split(split_str)[1] for i in dates_str])             
            # space = np.array([np.where(dates_short_str==i)[0][0] for i in np.unique(dates_short_str)])
        #dates_str = np.array([i[:7] for i in dates_str])  # 直接显示时间
        #space = np.int32(np.linspace(0, dates_str.shape[0] - 1, 9))
#        plt.xticks(space, dates_str[space])

        # plt.sca(sub2)
        # plt.bar(np.arange(draw_down.shape[0]), draw_down, width=0.025, edgecolor='red', color='red', alpha=0.2)
        # plt.ylabel("Drawdown $")
        # plt.grid()
        # plt.show()
