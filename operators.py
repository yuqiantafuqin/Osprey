# coding=utf-8

# last modified: 20170504
# support 1d array and 2d arry computation
"""
==========
this is one of methods categorizing operators 
using args nums operators using
Unary Operators | Rank, Abs, Sign, Log, Pasteurize, Scale, Step
2‐operand Operators | +, ‐, *, /, ^, <, <,=, >, >,=, ==, !=, ||, &&, Min, Max,
StdDev, Ts_min, Ts_Max, Ts_Rank ,Ts_Kurtosis,
Ts_Skewness, SignedPower, Sum, Delay, Delta, Product,
Decay_linear, CountNans
3‐operand Operators | ?:, Correlation, Call_i, Ts_Moment, Decay_exp
4‐operand Operator | Tail
5‐operand Operator | Sum_i
==========
also, we can use disciplines operators belongs to, e.g. math, linear algebra, statistics etc.
==========
"""

import numpy as np
import scipy.stats as st
import talib
import time
global execution_time
execution_time = {}
print 'opeartors has been imported 20170419'

class OpException(Exception):
	def __init__( value):
		value = value

	def __str__(self):
		return repr(value)


def decorator(func):
	def wrapper(*args, **kw):
		# print 'call %s()' % func.__name__
		# try:
		# return func(*args, **kw)
		# except Exception, e:
		# raise Exception('Error operator:%s, error type:%s' % (func.__name__, e))
		t0 = time.time()
		result = func(*args, **kw)
		t1 = time.time()
		global execution_time 
		if func.func_name not in execution_time.keys():
			execution_time[func.func_name] = [t1-t0]
		else:
			execution_time[func.func_name].append(t1-t0)
		return result
	return wrapper

def exec_time():
	from prettytable import PrettyTable  
	x = PrettyTable(['Operator', 'time'])
	x.align['Operator'] = 'r'
	x.align['time'] = 'l'
	x.padding_width = 1
	global execution_time 
	for i in execution_time.keys():
		x.add_row([i, execution_time[i]])
	print x
 
	
# ====================================预处理函数====================================
@decorator
def pasteurize(x):
	"""
	set to Nan if it is INF
	"""
	res = x.copy()
	res[np.isinf(x)] = np.nan
	return res
	
@decorator
def universe_process(x, universe_t_f):
	"""
	set to Nan if the underlying instrument is not in the universe
	"""
	pass

@decorator
def indneutralize(x, y):
	"""
	num neutralization
	Neutralize alpha x against grouping specified by y. 
	e.g. indneutralize(x, industry) 
	indneutralize(x, 1) neutral against whole market ???
	"""
	pass
	

# ====================================元素级数组函数====================================

@decorator
def MAX(x, y):
	return x*(x>=y) + y*(x<y)

@decorator
def MIN(x, y):
	return x*(x<=y) + y*(x>y)

@decorator
def abs(x):
	return np.abs(x)

@decorator
def ABS(x):
	return np.abs(x)

@decorator
def sqrt(x):
	if ~np.array((x >= 0)).all():
		raise OpException('NANs exist in function sqrt()!')
	return np.sqrt(x)


@decorator
def SQRT(x):
	#print x
	# if ~np.array((x >= 0)).all():
	# 	raise OpException('NANs exist in function sqrt()!')
	return np.sqrt(x)


@decorator
def square(x):
	return np.square(x)


@decorator
def exp(x):
	return np.exp(x)


@decorator
def log(x):
	#if ~np.array((x > 0)).all():
		
			#raise OpException('NANs exist in function log()!')
	return np.log(x)


@decorator
def log10(x):
	if ~np.array((x > 0)).all():
		raise OpException('NANs exist in function log10()!')
	return np.log10(x)


@decorator
def log2(x):
	if ~np.array((x > 0)).all():
		raise OpException('NANs exist in function log2()!')
	return np.log2(x)


@decorator
def sign(x):
	return np.sign(x)


@decorator
def signedpower(x, pow):
	"""
	带符号的power
	"""
	return (np.abs(x) ** pow) * np.sign(x)

@decorator
def INTPART(x):
	# 取整
	# 用法:INTPART(A)返回沿A绝对值减小方向最接近的整数
	# 例如:INTPART(12.3)求得12,INTPART(-3.5)求得-3
	import math
	def intpart(x):
		if x<0:
			return math.ceil(x)
		else:
			return math.floor(x)
	return np.array(map(intpart, x))

# ====================================三角函数====================================


@decorator
def sin(x):
	return np.sin(x)

@decorator
def SIN(x):
	return np.sin(x)


@decorator
def cos(x):
	return np.cos(x)


@decorator
def tan(x):
	"""
	Notes:np.nan seems not to check domain of definition!
	"""
	return np.tan(x)


# ====================================反三角函数====================================


@decorator
def arcsin(x):
	if ~(np.array(np.abs(x) <= 1)).all():
		raise OpException('NANs exist in function arcsin()!')
	return np.arcsin(x)


@decorator
def arccos(x):
	if ~(np.array(np.abs(x) <= 1)).all():
		raise OpException('NANs exist in functoin arccos()!')
	return np.arccos(x)


@decorator
def arctan(x):
	return np.arctan(x)


# ====================================双曲函数====================================


@decorator
def sinh(x):
	"""
	sinh = (e^x - e^-x) / 2
	"""
	return np.sinh(x)


@decorator
def cosh(x):
	"""
	cosh = (e^x + e^-x) / 2
	"""
	return np.cosh(x)


@decorator
def tanh(x):
	"""
	tanh = (e^x - e^-x) / (e^x + e^-x)
	"""
	return np.tanh(x)


# ====================================反双曲函数====================================


@decorator
def arcsinh(x):
	"""
	domain of definition: R
	"""
	return np.arcsinh(x)


@decorator
def arccosh(x):
	"""
	domain of definition: [1, +Inf]
	"""
	if ~(np.array(x >= 1)).all():
		raise OpException('Warning: NANs exist in function arccosh()!')
	return np.arccosh(x)


@decorator
def arctanh(x):
	"""
	domain of definition: (-1, 1)
	"""
	if ~(np.array(np.abs(x) < 1)).all():
		raise OpException('Warning: NANs exist in function arctanh()!')
	return np.arctanh(x)


# ====================================数论函数====================================


@decorator
def gcd(a, b):
	try:
		import fractions
		return fractions.gcd(a, b)
	except:
		raise OpException('Error:can not import fractions module in function gcd()!')


@decorator
def lcm(a, b):
	"""
	两数乘积 = 最大公约数 * 最小公倍数
	"""
	try:
		import fractions
		return a * b / fractions.gcd(a, b)
	except:
		raise OpException('Error:can not import fractions module in function lcm()!')


@decorator
def factorial(x):
	"""
	return each element's factorial of x
	x: 1-dimension ndarray
	math.factorial() will check the input
	"""
	try:
		from math import factorial as fa
	except:
		raise OpException('can not import math.factorial() in function factorial()!')
	res = np.zeros_like(x) * np.nan
	for i in range(len(x)):
		res[i] = fa(x[i])
	return res


@decorator
def ceil(x):
	"""
	return smallest integer larger than x
	'"""
	return np.ceil(x)


@decorator
def floor(x):
	"""
	return largest integer less than x
	"""
	return np.floor(x)


'''
@decorator
def round(x):
	"""
	return nearest integer of x
	"""
	return np.round(x)
'''


@decorator
def reciprocal(x):
	"""
	Notes: do not use numpy here!
	#return np.reciprocal(x)
	testcase:
	a = [1, -1, 0, 100]
	a = np.array(a)
	np.reciprocal(a)
	document:http://docs.scipy.org/doc/numpy-1.7.0/reference/generated/numpy.reciprocal.html
	"""
	if ~x.all():
		raise OpException('INFs exist in function reciprocal()!')
	return 1.0 / x


# ====================================统计运算符====================================

@decorator
def tail(x, lower, upper, newval):
	# Set the values of x to newval if they are between lower and upper  
	cond =  (x < upper) * (x > lower) 
	return ~cond * x + newval
	
	
@decorator
def ts_sum(x, period):
	if period > 1:
		return function_wrapper("SUM", x, timeperiod=period)
	elif period == 1 or period == 0:
		return x

@decorator
def SUM(x, period):
	if type(period) is int or type(period) is float or type(period) is long:
		return ts_sum(x, period)
	elif type(period) is np.ndarray:
		res = np.zeros_like(x) * np.nan
		if np.isnan(period).all():
			return x
		loc = (np.arange(len(x))[~np.isnan(period)])[0]
		index_ = (np.arange(len(x))-period).astype('int')
		for i in xrange(loc, len(x)):
			res[i] = np.nansum(x[index_[i]:i+1]) 
		return np.array(res)
	else:
		raise Exception(e)



@decorator
def ts_product(x, period):
	"""
	time-series product over the past period days
	"""
	tmp = x ** 2
	if period > 1:
		return function_wrapper('SUM', tmp, timeperiod=period)
	elif period == 1:
		return tmp


@decorator
def ts_min(x, period):
	# 取前n天数据的最小值
	return function_wrapper("MIN", x, timeperiod=period)


@decorator
def ts_max(x, period):
	# 取前n天数据的最大值
	return function_wrapper("MAX", x, timeperiod=period)


@decorator
def ts_argmax(x, period):
	"""
	过去period天的最大值的位置，范围[1,period]
	"""
	res = function_wrapper("MAXINDEX", x, timeperiod=period) - np.arange(x.shape[0]) + period * np.ones(x.shape[0])
	res[0:period - 1] = np.nan
	return res


@decorator
def ts_argmin(x, period):
	"""
	过去period天的最小值的位置，范围[1,period]
	"""
	res = function_wrapper("MININDEX", x, timeperiod=period) - np.arange(x.shape[0]) + period * np.ones(x.shape[0])
	res[0:period - 1] = np.nan
	return res


@decorator
def ts_rank(x, period):
	"""
	time_series_rank for each stock in matrix,
	each return's matrix element is rank's order of past days
	st.mstats.rankdata算法太慢, 用np.argsort替代，需要改进
	"""
	res = np.zeros(x.shape) * np.nan
	for ix in range(0, x.shape[0] - period + 1):
		res[ix + period - 1] = (np.argsort(np.argsort(x[ix:ix + period]))[-1] + 1) * 1. / period
	return res


@decorator
def ma(x, period):
	if np.isnan(x).all():
		raise OpException('NAN returned in function median()!')
	#res = function_wrapper("MA", x, timeperiod=period)
	return talib.MA(x, period)

@decorator
def MA(x, period):
	return ma(x, period)

@decorator
def SMA(x, period, wgt=1):
	if wgt==1:
		return ma(x, period)
	return ma(x, period)

@decorator
def WMA(x, period):
	return function_wrapper("WMA", x, timeperiod=period)

@decorator
def median(x, period):
	if np.isnan(x).all():
		raise OpException('NAN returned in function median()!')
	res = [np.nan] * (period - 1)
	for i in range(len(x) - period + 1):
		res.append(np.nanmedian(x[i: i + period]))
	return np.array(res)


@decorator
def percentile(x, period, percentile):
	if np.isnan(x).all():
		raise OpException('Warning: NAN returned in function percentile()!')
	res = [np.nan] * (period - 1)
	for i in range(len(x) - period + 1):
		res.append(np.nanpercentile(x[i: i + period], percentile))
	return np.array(res)


@decorator
def skewness(x, period):
	"""
	return the rolling skewness of x
	nan_policy: decide how to handle when input contains nan.
				'omit' performs the calculations ignoring nan values.
				Default is 'propagate' that will let skewness return nan
	"""
	l = len(x)
	res = [np.nan] * (period - 1)
	for i in range(l - period + 1):
		res.append(st.skew(x[i: i + period], nan_policy='omit'))
	return np.array(res)


@decorator
def kurtosis(x, period):
	"""
	return the rolling kurtosis of x
	nan_policy: decide how to handle when input contains nan.
				'omit' performs the calculations ignoring nan values.
				Default is 'propagate' that will let kurtosis return nan
	for details:http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
	"""
	l = len(x)
	res = [np.nan] * (period - 1)
	for i in range(l - period + 1):
		res.append(st.kurtosis(x[i: i + period], nan_policy='omit'))
	return np.array(res)


@decorator
def k_moment(x, period, pow):  ##
	"""
	计算x的滚动k阶中心距
	nan_policy: decide how to handle when input contains nan.
				'omit' performs the calculations ignoring nan values.
				Default is 'propagate' that will let kurtosis return nan
	"""
	l = len(x)
	res = [np.nan] * (period - 1)
	for i in range(l - period + 1):
		res.append(st.moment(x[i: i + period], moment=pow, nan_policy='omit'))
	return np.array(res)


@decorator
def beta(high, low, period):
	return function_wrapper('BETA', high, low, timeperiod=period)


@decorator
def linearreg(data, period):
	return function_wrapper('LINEARREG', data, timeperiod=period)


@decorator
def linearreg_angle(data, period):
	return function_wrapper('LINEARREG_ANGLE', data, timeperiod=period)


@decorator
def linearreg_intercept(data, period):
	return function_wrapper('LINEARREG_INTERCEPT', data, timeperiod=period)

@decorator
def INTERCEPT(data, period):
	return function_wrapper('LINEARREG_INTERCEPT', data, timeperiod=period)

@decorator
def linearreg_slope(data, period):
	return function_wrapper('LINEARREG_SLOPE', data, timeperiod=period)

@decorator
def SLOPE(data, period):
	return linearreg_slope(data, period)



@decorator
def tsf(data, period):
	return function_wrapper('TSF', data, timeperiod=period)


@decorator
def stddev(x, period):  # 标准差
	return function_wrapper('STDDEV', x, timeperiod=period)

@decorator
def STD(x, period):  # 标准差
	return function_wrapper('STDDEV', x, timeperiod=period)

@decorator
def var(data, period):
	return function_wrapper('VAR', data, timeperiod=period)

@decorator
def AVEDEV(data, period):
	# AVEDEV—平均绝对偏差 Mean absolute deviation
	return MA(abs(data-MA(data, period)), period)

@decorator
def scale(x):  # 缩放运算
	"""
	rescaled x such that sum(abs(x)) = a (the default is a = 1)
	"""
	a = 1
	tmp = np.nansum(np.abs(x))
	return x / tmp * a


@decorator
def correlation(x, y, period):
	"""
	time-serial correlation of x and y for the past d days
	使用Talib的皮尔森相关系数算法
	"""
	return function_wrapper('CORREL', x, y, timeperiod=period)


@decorator
def covariance(x, y, period):
	"""
	time-serial covariance of x and y for the past d days
	使用皮尔森相关系数的公式算协方差公式
	"""
	corr_matrix = function_wrapper('CORREL', x, y, timeperiod=period)
	var_x = function_wrapper('VAR', x, timeperiod=period)
	var_y = function_wrapper('VAR', y, timeperiod=period)
	return corr_matrix * var_x * var_y


# ====================================decay运算符====================================


@decorator
def decay_linear(x, period):
	"""
	weighted moving average over the past d days
	with linearly decaying  weights d, d – 1, …, 1 (rescaled to sum up to 1)
	"""
	return function_wrapper("WMA", x, timeperiod=period)


@decorator
def decay_exponent(x, period):
	"""
	weighted moving average over the past d days
	with exponent decaying  weights d, d – 1, …, 1 (rescaled to sum up to 1)
	to be optimized
	"""
	return function_wrapper("EMA", x, timeperiod=period)


@decorator
def decay_fibonacci(x, period):
	"""
	rolling fibonacci-number weighteD
	with decaying weights d, d-1, …, 1 (rescaled to sum up to 1)
	return: matrix

	"""

	def fib_number(n):
		a, b = 0, 1
		for i in range(n):
			a, b = b, a + b
		return a

	fibonacci = []
	for j in range(period):
		fibonacci.append(fib_number(j + 1))
	fibonacci = np.array(fibonacci)
	weight = fibonacci / (np.nansum(np.abs(fibonacci)) + 0.0)
	res = [np.nan] * (period - 1)
	l = len(x)
	for i in range(l - period + 1):
		res.append((x[i: i + period] * weight).sum())
	return np.array(res)

# ====================================复杂表达式====================================
def sum_i(expr_str, var_str, start, stop, step):
	"""
	loop over var(from start to stop with step) and calculate expr at every iteration (presumably expr would contain var), then sum over all the values.
	e.g. sum_i(delay(c, i)*i, i, 2, 4, 1) would be equivalent to delay(c, 2) *2 + delay(c, 3)*3 +delay(close, 4) *4
	"""
	space = np.arange(start, stop, step)
	res = []
	for i in space:
		if len(res) == 0:
			res = eval(expr_str.replace(var_str, i))
		else:
			res += eval(expr_str.replace(var_str, i))
	return res
	
def call_i(expr_str, var_str, subexpr):
	"""
	call_i(x+4, x, 2+3) would be equivalent to (2+3)+4
	"""
	return eval(expr_str.replace(var_str, str(subexpr)))
	
	

# ====================================其他函数====================================


@decorator
def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


@decorator
def nike(x):
	if ~x.all():
		raise OpException('INFs exist in function nike()!')
	return x + 1.0 / x


@decorator
def roll_geometric_mean(x):
	"""
	return [sqrt(x1*x2), sqrt(x2*x3), …… , sqrt(xn-1*xn)]
	Notes:the first element is nan
	"""
	x_shift = np.zeros_like(x) * np.nan
	x_shift[: -1] = x[1:]
	x_roll = x * x_shift
	x_roll[1:] = x_roll[: -1]
	x_roll[0] = np.nan
	return np.sqrt(x_roll)


@decorator
def count_nan(x):
	"""
	return the number of NANs in x
	"""
	return np.sum(np.isnan(x))


@decorator
def count_nans(x, period):
	"""
	return the number of NANs in x during last period
	"""
	return ts_sum(1.*np.isnan(x), period)
	
@decorator
def ts_count(cond, period):
	"""
	return the number of cond=True in x during last period
	"""
	return ts_sum(1.*(cond==True), period)

@decorator
def COUNT(cond, period):
	"""
	return the number of cond=True in x during last period
	"""
	if type(period) is int or type(period) is float or type(period) is long:
		return ts_sum(1.*(cond==True), period)

	elif type(period) is np.ndarray:
		index_ = np.arange(len(cond))-np.nan_to_num(period)
		res = []
		for i in xrange(0, len(cond)):
			if i >= period[0]:
				res.append(np.nansum(cond[index_.astype('int')[i]:i+1]))
			else:
				res.append(np.nan)
		return np.array(res)
	else:
		raise Exception(e)




@decorator
def delay(x, period):
	"""
	delay() value of x d days ago
	"""
	res = np.zeros(x.shape) * np.nan
	res[period:] = x[:-period]
	return res

@decorator
def REF(x, period):
	if type(period) is int or type(period) is float or type(period) is long:
		return delay(x, period)
	elif type(period) is np.ndarray:
		index_ = np.arange(len(x))-np.nan_to_num(period)
		return x[index_.astype('int')]
	else:
		raise Exception(e)


@decorator
def delta(x, period):
	"""
	today’s value of x minus the value of x d days ago
	"""
	res = np.zeros(x.shape) * np.nan
	res[period:] = x[:-period]
	return x - res


@decorator
def delta2(x, period1, period2):
	"""
	x的二阶差分
	"""
	return delta(delta(x, period1), period2)


# ====================================Technical====================================

@decorator
def llv(data, n):
	return function_wrapper("MIN", data, timeperiod=n)

@decorator
def LLV(data, period):
	if type(period) is int or type(period) is float or type(period) is long:
		return function_wrapper("MIN", data, timeperiod=period)
	elif type(period) is np.ndarray:
		index_ = np.arange(len(data))-np.nan_to_num(period)
		res = []
		for i in xrange(0, len(data)):
			if i >= period[0]:
				res.append(np.min(data[index_.astype('int')[i]:i+1]) )
			else:
				res.append(np.nan)
		return np.array(res)
	else:
		raise Exception(e)



def LLVBARS(data, n):
	return function_wrapper("MININDEX", data, timeperiod=n)

@decorator
def hhv(data, n):
	return function_wrapper("MAX", data, timeperiod=n)

@decorator
def HHV(data, period):
	if type(period) is int or type(period) is float or type(period) is long:
		return function_wrapper("MAX", data, timeperiod=period)
	elif type(period) is np.ndarray:
		res = np.zeros_like(data) * np.nan
		if np.isnan(period).all():
			return data
		loc = (np.arange(len(data))[~np.isnan(period)])[0]
		index_ = (np.arange(len(data))-period).astype('int')
		for i in xrange(loc, len(data)):
			res[i] = np.max(data[index_[i]:i+1]) 
		return np.array(res)
	else:
		raise Exception()


def HHVBARS(data, n):
	return function_wrapper("MAXINDEX", data, timeperiod=n)

@decorator
def atr(high, low, close, period):
	return function_wrapper('ATR', high, low, close, timeperiod=period)


@decorator
def sub(high, low):
	return function_wrapper('SUB', high, low)


@decorator
def ht_trendline(data):
	return function_wrapper('HT_TRENDLINE', data)


@decorator
def kama(data, period):
	return function_wrapper('KAMA', data, timeperiod=period)


@decorator
def midpoint(data, period):
	return function_wrapper('MIDPOINT', data, timeperiod=period)


@decorator
def ema(data, period):
	#return function_wrapper('EMA', data, timeperiod=period)
	return talib.EMA(data, period) #function_wrapper("EMA", data, period)

@decorator
def EMA(data, period):
	#return function_wrapper('EMA', data, timeperiod=period)
	if period == 1:
		return data
	return function_wrapper("EMA", data, period)

@decorator
def DMA(data, arr):
	# DMA  动态移动平均
	# 求动态移动平均.
	# 用法:
	#  DMA(X,A),求X的动态移动平均.
	# 算法: 若Y=DMA(X,A)则 Y=A*X+(1-A)*Y',其中Y'表示上一周期Y值,A必须小于1.
	# 例如:DMA(CLOSE,VOL/CAPITAL)表示求以换手率作平滑因子的平均价
	return data

@decorator
def EXPMA(data, period):
	#return function_wrapper('EMA', data, timeperiod=period)
	if period == 1:
		return data
	return function_wrapper("EMA", data, period)



@decorator
def EXPMEMA(data, period):
	return ema(data, period)


@decorator
def wma(data, period):
	return function_wrapper('WMA', data, timeperiod=period)


@decorator
def tema(data, period):
	return function_wrapper('TEMA', data, timeperiod=period)


def MACD(close, short=12, long=26, mid=9):
	macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
	return macd
	
@decorator
def macd(close, short=12, long=26, mid=9):
	dif = ema(close, short) - ema(close, long)
	dea = ema(dif, mid)
	macd = (dif - dea) * 2
	return macd

@decorator
def kdj(close, high, low, n=9, m1=3, m2=3):
	rsv = (close - llv(low, n)) / (hhv(high, n) - llv(low, n)) * 100
	k = ema(rsv, m1)
	d = ema(k, m2)
	j = 3 * k - 2 * d
	return k, d, j


@decorator
def ll(low, period):
	"""
	lowest price
	"""
	return ts_min(low, period)


@decorator
def hh(high, period):
	"""
	highest price
	"""
	return ts_max(high, period)


@decorator
def mp(low, high, period):
	"""
	median price
	"""
	return (ll(low, period) + hh(high, period)) / 2


@decorator
def k(close, low, high, period):
	"""
	Stochastic %K
	"""
	return (close - ll(low, period)) / (hh(high, period) - ll(low, period))


@decorator
def d(close, low, high, period):
	"""
	%D is the moving average of %K
	"""
	return ma(k(close, low, high, period), period)


@decorator
def r(high, low, close, period):
	"""
	Larry William's %R
	"""
	return function_wrapper('WILLR', high, low, close, timeperiod=period)


@decorator
def bias(close, period):
	"""
	x-days bias
	"""
	return (close - ma(close, period)) / period


@decorator
def oscp(close, period1, period2):
	"""
	Price oscillator
	"""
	return (ma(close, period1) - ma(close, period2)) / (ma(close, period1))


@decorator
def cci(high, low, close, period):
	"""
	Commodity channel index
	"""
	return function_wrapper('CCI', high, low, close, timeperiod=period)


@decorator
def signalline(close, period1, period2):
	"""
	A signalline is also known as a trigger line
	"""
	t1 = ma(close, period1)
	t2 = ma(close, period2)
	return (t1 - t2) / (10 * t2) + t2


@decorator
def mtm(close, period):
	"""
	Momentum measures change in stock price over last x days
	"""
	return delta(close, period)


@decorator
def tsi(close, period):
	"""
	True strength index
	"""
	return 100 * (ema((ema(mtm(close, 1), period)), period)) / (ema((ema(abs(mtm(close, 1)), period)), period))


@decorator
def uo(high, low, close, period1, period2, period3):
	"""
	Ultimate oscillator
	"""
	return function_wrapper('ULTOSC', high, low, close, timeperiod1=period1,
							timeperiod2=period2, timeperiod3=period3)


@decorator
def yj(high, low, close, period=26):
	# yj strategy
	atr = function_wrapper('ATR', high, low, close, timeperiod=period)
	middleLine = ma(close, period)
	atrUp = middleLine + atr
	atrDown = middleLine - atr
	res = []
	MarketPosition = 0
	for di in range(0, close.shape[0]):
		if MarketPosition == 0 and close[di] > atrUp[di]:
			MarketPosition = 1
		elif MarketPosition == 0 and close[di] < atrDown[di]:
			MarketPosition = -1
		elif MarketPosition == 1 and close[di] < middleLine[di]:
			MarketPosition = 0
		elif MarketPosition == -1 and close[di] > middleLine[di]:
			MarketPosition = 0
		res.append(MarketPosition)
	return np.array(res)


# ====================================逻辑运算符====================================
@decorator
def logical(cond, arr1, arr2):
	if type(arr1) is int or type(arr1) is float:
		arr1 = np.ones_like(arr2) * arr1
	elif type(arr2) is int or type(arr2) is float:
		arr2 = np.ones_like(arr1) * arr2
	return (cond==1)*arr1 + (cond==0)*arr2

@decorator
def IFF(cond, arr1, arr2):
	#if type(arr1) is int or type(arr1) is float:
	#	arr1 = np.ones_like(cond) * arr1
	#elif type(arr2) is int or type(arr2) is float:
	#	arr2 = np.ones_like(cond) * arr2
	return (cond==1)*arr1 + (cond==0)*arr2

@decorator
def EXIST(cond, period):
	# EXIST  存在
	# 是否存在.
	# 用法:
	#  EXIST(CLOSE>OPEN,10) 
	#  表示前10日内存在着阳线	
	if type(period) is int or type(period) is float or type(period) is long:
		return ts_sum(1.*(cond==True), period) > 0

	elif type(period) is np.ndarray:
		index_ = np.arange(len(cond))-np.nan_to_num(period)
		res = []
		for i in xrange(0, len(cond)):
			if i >= period[0]:
				res.append(np.nansum(cond[index_.astype('int')[i]:i+1])>0)
			else:
				res.append(np.nan)
		return np.array(res)
	else:
		# print type(period)
		raise Exception(e)

@decorator
def NOT(cond):
	return EXIST == 0


@decorator
def BETWEEN(cond, val1, val2):
	return (cond>=val1) * (cond<=val2)

# ====================================动作运算符====================================
def CROSS(arr1, arr2):
	# up cross
	if type(arr1) is int or type(arr1) is float:
		arr1 = np.ones_like(arr2) * arr1
		res = np.zeros_like(arr2).astype('bool')
	elif type(arr2) is int or type(arr2) is float:
		arr2 = np.ones_like(arr1) * arr2
		res = np.zeros_like(arr1).astype('bool')
	else:
		res = np.zeros_like(arr1).astype('bool')
	cross_c = np.sign(arr1 - arr2)
	res[2:] = np.logical_and(cross_c[2:] == 1, cross_c[:-2] == -1)
	return res


def BARSLAST(arr):
	# find the last True location since now 
	#arr == 1
	if arr.dtype != 'bool':
		raise Exception('arr should be bool dtype')
	res = np.zeros_like(arr)*np.nan
	if len(np.where(arr)[0]) ==0:
		return res
	loc = np.where(arr)[0][0]
	for i in xrange(loc,len(arr)):
		if arr[i] == 1:
			res[i] = 0
		else:
			#print i, arr[:i]#, (np.arange(i)[arr[:i]])
			res[i] = i - (np.arange(i)[arr[:i]])[-1]
	return res


# def BARSLAST(arr):
# 	# find the last True location since now 
# 	res = []
# 	true_loc = np.where(arr)[0]
# 	for i in xrange(0,len(arr)):
# 		if arr[i] == 1:
# 			res.append(0)
# 		else:
# 			res.append(i-np.arange(i+1)[arr[:i+1]][-1])
# 	return np.array(res)


def TFILTER(arr1, arr2, N):
	# TFILTER  交易信号过滤
	# 过滤连续出现的交易信号.
	# 用法:FILTERX(开仓,平仓,N):N=1,表示仅对开仓信号过滤;
	# N=2,表示仅对平仓信号过滤;
	# N=0,表示对开仓信号和平仓信号都过滤;
	if N==1:
		res = arr1*(arr2 == 0)
		return res
	if N==2:
		res = arr2*(arr1 == 0)
		return res 
	if N==0:
		return (arr1==1) * (arr2==1)

def FILTER(arr, period):
	# 	FILTER  过滤
	# 过滤连续出现的信号.
	# 用法:FILTER(X,N):X满足条件后，删除其后N周期内的数据置为0.
	#  例如：FILTER(CLOSE>OPEN,5)查找阳线，5天内再次出现的阳线不被记录在内
	res = np.zeros_like(arr) * np.nan
	for i in xrange(period, len(arr)):
		res[i] = arr[i-period:i]>0




# ==================================== 形态 ====================================
def ZIG(arr, N):
	# 之字转向.
	# ZIG(K,N),当价格变化量超过N%时转向,K表示  0:开盘价,1:最高价,2:最低价,3:收盘价,其余:数组信息
	pass


def TROUGHBARS(arr, val, val2):
	# TROUGHBARS  波谷位置
	# 前M个ZIG转向波谷到当前距离.
	# 用法:
	# TROUGHBARS(K,N,M)表示之字转向ZIG(K,N)的前M个波谷到当前的周期数,M必须大于等于1
	pass

# ====================================以下请勿轻易改动====================================
def function_wrapper(func, *args, **kwargs):
	s = tuple()
	for a in args:
		if isinstance(a, np.ndarray):
			s = a.shape

	res = np.zeros(s, ).astype(np.float)
	func = func.upper()

	if func in talib.func.__TA_FUNCTION_NAMES__:
		f = getattr(talib, func)
	else:
		raise Exception("%s is not a valid talib function" % func)

	loop_num = s[-1] if len(s) <> 1 else 1
	# print str(loop_num) + '*'
	if loop_num <> 1:
		for i in xrange(s[-1]):
			one_dim_args = []
			for a in args:
				if isinstance(a, np.ndarray) and a.shape == s:
					# a = np.nan_to_num(a)
					al = a[:, i]
					al = np.nan_to_num(al)

					# Talib functions can not handle input with all element is np.nan
					# So give a 0
					if np.all(np.isnan(al)):
						al[0] = 0
					one_dim_args.append(al)
				else:
					one_dim_args.append(a)

			ri = f(*one_dim_args, **kwargs)

			if isinstance(ri, tuple):
				if i == 0:
					sl = list(s)
					sl.insert(0, len(ri))
					res = np.zeros(tuple(sl), dtype=float)
				for q in xrange(len(ri)):
					res[q, :, i] = ri[q]
			else:
				res[:, i] = ri
			return res

	else:
		one_dim_args = []
		for a in args:
			if isinstance(a, np.ndarray) and a.shape == s:
				# a = np.nan_to_num(a)
				al = a[:]
				al = np.nan_to_num(al)

				# Talib functions can not handle input with all element is np.nan
				# So give a 0
				if np.all(np.isnan(al)):
					al[0] = 0
				one_dim_args.append(al)
			else:
				one_dim_args.append(a)

		ri = f(*one_dim_args, **kwargs)
		if isinstance(ri, tuple):
			if i == 0:
				sl = list(s)
				sl.insert(0, len(ri))
				res = np.zeros(tuple(sl), dtype=float)
			for q in xrange(len(ri)):
				res[q, :, i] = ri[q]
		else:
			res[:] = ri
		return res


	# 01--------------------------------------------------------------
	# DMA
	# 平行线差指标
	def trend_DMA( close_price, n1=10, n2=50, m=10):

		dif = ma(close_price, n1) - ma(close_price, n2)
		ama = ma(dif, m)
		x = dif - ama
		return x>0

	# 02--------------------------------------------------------------
	# DMI
	# 动向指标或趋向指标，由PDI、MDI、ADX、ADXR四条线组成，
	# N[2,80] m[1,80],m=6
	def trend_DMI( close, high, low, n=14, m=6):
		pre_close = np.vstack((np.nan*np.ones(close.shape[1]), close[:-1,:])) #前一天收盘价，保持矩阵size不变
		pre_high = np.vstack((np.nan*np.ones(high.shape[1]), high[:-1,:]))    #前一天最高价
		pre_low = np.vstack((np.nan*np.ones(low.shape[1]), low[:-1,:]))    #前一天最高价

		ITR = expmema(max_arr(max_arr(high-low, abs(high-pre_close)), abs(pre_close-low)), n)
		HD = high - pre_high
		LD = pre_low - low
		DMP_hd = np.zeros(HD.shape)
		DMM_ld = np.zeros(LD.shape)
		for j in range(0, HD.shape[0]):
			for i in range(0, HD.shape[1]):
				if HD[j][i] > 0 and HD[j][i] > LD[j][i]:
					DMP_hd[j][i] = HD[j][i]
				else:
					DMP_hd[j][i] = 0
				if LD[j][i] > 0 and LD[j][i] > HD[j][i]:
					DMM_ld[j][i] = LD[j][i]
				else:
					DMM_ld[j][i] = 0

		DMP = expmema(DMP_hd, n)
		DMM = expmema(DMM_ld, n)
		PDI = DMP*100/ITR
		MDI = DMM*100/ITR
		# ADX = fm.expmema(abs(MDI-PDI)/(MDI+PDI)*100, m)
		# ADXR = fm.expmema(ADX, m)
		x = PDI -MDI         # 判断条件是PDI[-1] > MDI[-1]
		return x>0
	#
	# # 03--------------------------------------------------------------
	# # DPO
	# # 区间震荡线，短期的波动和超买超卖水平,n[2,90] m[2,60]
	def trend_DPO( close_price, n=20, m=6):
		temp = ma(close_price, n/2+1)  #可以改进，使用20ma做close
		dpo = close_price - temp
		madpo = ma(dpo, m)
		x = dpo - madpo
		return x>0

	# 04--------------------------------------------------------------
	# EMV指标
	# 简易波动指标，n[2,90] m[2,60]
	def trend_EMV( volume, high, low, pre_high, pre_low, n=14, m =9):
		vol = ma(volume, n)/volume
		mid = 100*(high + low - (pre_high + pre_low))/(high + low)
		emv = ma(mid*vol*(high-low)/ma(high - low, n), n)
		maemv = ma(emv, m)
		x = emv > maemv
		return x>0
	#
	# 05--------------------------------------------------------------
	# MACD
	# short[2,60] long[60,90] mid[2,60]
	def trend_MACD( close, s=12, l = 26, m=9):
		dif = ema(close, s) - ema(close, l)
		dea = ema(dif, m)
		macd = (dif-dea)*2          # 柱子长度
		return macd>0

	# 06--------------------------------------------------------------
	# TRIX指标
	# n=12 m=9
	def trend_TRIX( close,pre_close, n=12, m=9):
		tr = ema(ema(ema(close, n),n),n)
		pre_tr = ema(ema(ema(pre_close, n),n),n)
		trix = (tr-pre_tr)/(pre_tr)*100
		matrix = ma(trix, m)
		x = trix > matrix
		return x>0

	# 07--------------------------------------------------------------
	# WVAD 威廉变异离散量 n[2,120] m[2,60]
	def trend_WVAD( close_price, open_price, high_price, low_price,volume, n=24, m=6):
		wvad = ma((close_price-open_price)/(high_price-low_price)*volume,n)*n/10000
		mawvad = ma(wvad,m)
		x = wvad - mawvad
		return x>0

	# 08--------------------------------------------------------------
	# JS 加速线
	def trend_JS( close_price, m=5):
		pre_five_close_price = pre_n_day(close_price, m) #前5日收盘价
		js = 100*(close_price-pre_five_close_price)/(m*pre_five_close_price)
		majs = ma(js, m)
		x= js - majs
		return x>0
	# # 09--------------------------------------------------------------
	# # CYE 趋势线
	# def trend_CYE(close_price, n =5, m=20):
	#     mal = ma(close_price,n)
	#     mas = ma(ma(close_price,m),n)
	#     l = (mal - pre_n_day(mal,1))/pre_n_day(mal,1)*100
	#     s = (mas - pre_n_day(mas,1))/pre_n_day(mas,1)*100
	#     x = l - s
	#     return x>0
	#
	# # 10--------------------------------------------------------------
	# # JLHB 绝对路径
	# def trend_JLHB( close, low, high, n=60, m=80):
	#     var = (close-llv(low, 60))/(hhv(high, 60)-llv(low, 60))*80    # llv hhv: n天最低最高值
	#     b = sma(var, 7, 1)        #sma(VAR1,7,1)
	#     var2 = sma(b, 5, 1)        #sma(B,5,1)
	#     x = b - var2
	#     return x>0

	# 11--------------------------------------------------------------
	# def trend_CYC( )
	#     CYCJJJ = if(DYNAINFO(8)>0.01,0.01*DYNAINFO(10)/DYNAINFO(8),DYNAINFO(3));
	# CYCDDD:=(DYNAINFO(5)<0.01 || DYNAINFO(6)<0.01);
	# CYCJJJT:=IFF(CYCDDD,1,(CYCJJJ<(DYNAINFO(5)+0.01) && CYCJJJ>(DYNAINFO(6)-0.01)));
	# CYCCYC1:=IFF(CYCJJJT,0.01*EXPMA(AMOUNT,5)/EXPMA(VOL,5),EMA((HIGH+LOW+CLOSE)/3,5));
	# CYCCYC2:=IFF(CYCJJJT,0.01*EXPMA(AMOUNT,13)/EXPMA(VOL,13),EMA((HIGH+LOW+CLOSE)/3,13));
	# CYCX:=CYCCYC1>CYCCYC2;
	# P12:=CYCX;

	# 12--------------------------------------------------------------
	# BBI
	# def trend_BBI( close):
	#     bbi = (ma(close,3)+ma(close,6)+ma(close,12)+ma(close,24))/4
	#     x = close - bbi
	#     return x>0
	#
	# # 13--------------------------------------------------------------
	# def trend_DDI( high, low):
	#     tr = max_arr(abs(high - pre_n_day(high,1)),abs(low - pre_n_day(low,1)))
	#     a = high + low
	#     b = pre_n_day(high,1)+pre_n_day(low,1)
	#     dmz = np.zeros(high.shape)
	#     dmf = np.zeros(high.shape)
	#     for i in range(0,high.shape[1]):
	#         for j in range(0,high.shape[0]):
	#             if a[j,i] <= b[j,i]:
	#                 dmz[j,i] = 0
	#             else:
	#                 dmz[j,i] = tr[j,i]
	#             if a[j,i] >=b [j,i]:
	#                 dmf[j,i] = 0
	#             else:
	#                 dmf[j,i] = tr[j,i]
	#     diz = sum_n_day(dmz,13)/(sum_n_day(dmz,13)+sum_n_day(dmf,13))
	#     dif = sum_n_day(dmf,13)/(sum_n_day(dmf,13)+sum_n_day(dmz,13))
	#     ddi = diz - dif
	#     addi = sma(ddi,30,10)
	#     ad = ma(addi,5)
	#     x = addi - ad
	#     return x>0

	# 14--------------------------------------------------------------
	# def trend_FSL( close, volume, capital):
	#     swl = (expmema(close, 5)*7 + expmema(close, 10)*3)/10
	#     sws = np.zeros(swl.shape)
	#     for j in range(0, swl.shape[1]):
	#         for i in range(0, swl.shape[0]):
	#             m = max_arr(np.ones((swl.shape)), 100*(sum_n_day(volume, 5)/(3*capital)))
	#             data = ema(close, 12)
	#             dif = ma(data, 10) - ma(data, 50)
	#             ama = ma(dif, m[i,j])
	#             temp = dif - ama
	#             swl[i,j] = temp[i,j]
	#     x = swl - sws
	#     return x>0

	# 15--------------------------------------------------------------
	# ma day5/day10
	# def trend_MA( close):
	#     ma1 = ma(close, 5)
	#     ma2 = ma(close, 10)
	#     x = ma1 - ma2
	#     return x>0

	# 16--------------------------------------------------------------
	def trend_EXPMA( close):
		ema1 = expma(close, 12)
		ema2 = expma(close, 17)
		x = ema1 > ema2 and ema1 > pre_n_day(ema1, 1) and ema2 > pre_n_day(ema2, 1)
		return x

	# 17--------------------------------------------------------------
	def trend_PBX(close):
		pbx1 = (expma(close,4) + ma(close,4*2) + ma(close, 4*4))/3
		pbx2 = (expma(close,6) + ma(close,6*2) + ma(close, 6*4))/3
		x = pbx1>pbx2 and pbx1 > pre_n_day(pbx1, 1) and pbx2 > pre_n_day(pbx2, 1)
		return x

	# 18--------------------------------------------------------------
	def trend_SAR(close, high, low):
		sar = function_wrapper("SAR", high, low, acceleration=0.02, maximum=0.2)
		x = close > sar # sar(4,2,20)
		return x

	# 19--------------------------------------------------------------
	def trend_QLL(high,close,low,volume):
		if high> pre_n_day(close,1):
			tb = high - pre_n_day(close,1) + close - low
		else:
			tb = close - low
		if pre_n_day(close,1)>low:
			ts = pre_n_day(close,1)-low+high-close
		else:
			ts = high -close
		vol_ = (tb - ts)*volume/(tb+ts)/100
		vol_0 = dma(vol_,0.1)
		vol_1 = dma(vol_,0.05)
		res1 = vol_0 - vol_1
		lon = sum_n_day(res1,0)
		long = lon
		ma1 = ma(lon, 10)
		x = long > pre_n_day(long, 1) and ma1 > pre_n_day(ma1, 1) and long > ma1
		return x

	# 20--------------------------------------------------------------
	def trend_QLS(high,close,low,volume):
		if high> pre_n_day(close,1):
			tb = high - pre_n_day(close,1) + close - low
		else:
			tb = close - low
		if pre_n_day(close,1)>low:
			ts = pre_n_day(close,1)-low + high-close
		else:
			ts = high -close
		vol_ = (tb - ts)*volume/(tb+ts)/100
		vol_0 = dma(vol_,0.1)
		vol_1 = dma(vol_,0.05)
		short = vol_0 - vol_1
		ma1 = ma(short,10)
		x = short> ma1
		return x

	# 21--------------------------------------------------------------
	def trend_AMV(volume,open,close):
		amv0 = volume*(open + close)/2
		amv1 = sum_n_day(amv0, 5)/sum_n_day(volume, 5)
		amv2 = sum_n_day(amv0,13)/sum_n_day(volume, 13)
		x = amv1>amv2
		return x

	# 22--------------------------------------------------------------
	def trend_MTM(close):
		mtm = close - pre_n_day(close, 12)
		ma_mtm = ma(mtm, 6)
		x = mtm - ma_mtm
		return x

	# 23--------------------------------------------------------------
	def trend_ZJTJ(close):
		var1 = ema(ema(close,9), 9)
		kong_pan = (var1 - pre_n_day(var1, 1))/pre_n_day(var1, 1)*1000
		x = kong_pan > pre_n_day(kong_pan, 1) and kong_pan > 0
		return x


	# 01--------------------------------------------------------------
	def volume_LHLN( close, open_price, high, low, volume):
		# print 'using function volume_LHLN'
		cond = close >= open_price
		dir_vol = np.where(cond, volume, -volume)
		mid = (3*close + open_price + high + low)/6
		short = wma(mid, 5)
		long = wma(mid, 10)
		lhln_buy = short > long
		return dir_vol, lhln_buy
