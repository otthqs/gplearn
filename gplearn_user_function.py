def _minp(d):
    if not isinstance(d, int):
        d = int(d)
    if d <= 10:
        return d - 1
    else:
        return d * 2 // 3



def _rank(x_input, mask):

    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input

    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)
    res = x.rank(axis=1).sub(0.5).div(x.count(axis=1), axis=0).values.reshape(sp)
    return res[np.where(mask['fmt'])]
rank = gplearn2.functions.make_function(function = _rank, name = "rank", arity = 1, wrap = True)


def _delay(x_input, d, mask):

    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input

    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = x.shift(d).values.reshape(sp)

    return res[np.where(mask['fmt'])]

delay = gplearn2.functions.make_function(function = _delay, name = "delay",arity = 2, wrap = True)


def _correlation(x_input, y_input, d, mask):

    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    y = np.array([np.nan] * len(mask['fmt']))
    y[np.where(mask['fmt'])] = y_input

    x = x.reshape(mask['rows'],-1)
    y = y.reshape(mask['rows'],-1)

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    res = (x.rolling(window=int(d), min_periods=_minp(d)).corr(y)).values.reshape(sp)
    return res[np.where(mask['fmt'])]

correlation = gplearn2.functions.make_function(function = _correlation, name = "correlation", arity = 3, wrap = True)


def _covariance(x_input, y_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    y = np.array([np.nan] * len(mask['fmt']))
    y[np.where(mask['fmt'])] = y_input

    x = x.reshape(mask['rows'],-1)
    y = y.reshape(mask['rows'],-1)

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    res = (x.rolling(window=int(d), min_periods=_minp(d)).cov(y)).values.reshape(sp)
    return res[np.where(mask['fmt'])]

covariance = gplearn2.functions.make_function(function = _covariance, name = "covariance",arity = 3, wrap = True)



def _scale(x_input, a, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = x.mul(a).div(x.abs().sum(axis = 1), axis = 0).values.reshape(sp)
    return  res[np.where(mask['fmt'])]

scale = gplearn2.functions.make_function(function= _scale, name = "scale", arity = 2, wrap = True)



def _delta(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = x.diff(int(d)).values.reshape(sp)
    return res[np.where(mask['fmt'])]

delta = gplearn2.functions.make_function(function = _delta, name = "delta", arity = 2, wrap = True)


def _signedpower(x_input, a, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = (np.sign(x) * x.abs().pow(a)).values.reshape(sp)
    return res[np.where(mask['fmt'])]

signedpower = gplearn2.functions.make_function(function= _signedpower, name = "signedpower", arity=2, wrap=True)



def _decay_linear(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    weight = np.arange(0, int(d)) + 1
    res = (x.rolling(window = int(d), min_periods = _minp(d)).apply(lambda z: np.nansum(z * weight[-len(z):]) / weight[-len(z):][~np.isnan(z)].sum(),raw = True)).values.reshape(sp)
    return res[np.where(mask['fmt'])]

decay_linear = gplearn2.functions.make_function(function= _decay_linear, name = "decay_linear", arity = 2, wrap = True)




def _ts_min(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).min()).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_min = gplearn2.functions.make_function(function = _ts_min, name = "ts_min", arity = 2, wrap = True)




def _ts_max(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).max()).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_max = gplearn2.functions.make_function(function = _ts_max, name = "ts_max", arity = 2, wrap = True)




def _ts_argmin(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res =  (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).apply(lambda z: np.nan if np.all(np.isnan(z)) else np.nanargmin(z), raw = True) + 1).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_argmin = gplearn2.functions.make_function(function = _ts_argmin, name = "ts_argmin", arity = 2, wrap = True)



def _ts_argmax(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).apply(lambda z: np.nan if np.all(np.isnan(z)) else np.nanargmax(z), raw = True) + 1).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_argmax = gplearn2.functions.make_function(function = _ts_argmax, name = "ts_argmax", arity = 2, wrap = True)




def _ts_rank(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = (x.rolling(int(d), min_periods = _minp(d)).apply(
    lambda z: np.nan if np.all(np.isnan(z)) else ((rankdata(z[~np.isnan(z)])[-1] - 1) * (len(z) - 1) / (len(z[~np.isnan(z)]) - 1) + 1), raw = True)).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_rank = gplearn2.functions.make_function(function = _ts_rank, name = "ts_rank", arity = 2, wrap = True)



def _ts_sum(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res = (x.rolling(window = int(d), min_periods = _minp(d)).sum()).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_sum = gplearn2.functions.make_function(function = _ts_sum, name = "ts_sum", arity = 2, wrap = True)




def _ts_product(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res =  (np.log(np.exp(x).rolling(window = int(d), min_periods = _minp(d), axis = 0).mean() * int(d))).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_product = gplearn2.functions.make_function(function = _ts_product, name = "ts_product", arity = 2, wrap = True)




def _ts_stddev(x_input, d, mask):
    sp = mask['fmt'].shape
    x = np.array([np.nan] * len(mask['fmt']))
    x[np.where(mask['fmt'])] = x_input
    x = x.reshape(mask['rows'],-1)
    x = pd.DataFrame(x)

    res =  (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).std()).values.reshape(sp)
    return res[np.where(mask['fmt'])]

ts_stddev = gplearn2.functions.make_function(function = _ts_stddev, name = "ts_stddev", arity = 2, wrap = True)





function_set_test = ["add", "sub", "mul", "div","abs", "sqrt","log", "inv", rank, delay, correlation, covariance, scale, delta, signedpower,
                     decay_linear, ts_min, ts_max, ts_argmin, ts_argmax, ts_rank, ts_sum, ts_product, ts_stddev]
reshape_function_set_test = [rank, delay, correlation, covariance, scale, delta, signedpower, decay_linear, ts_min, ts_max, ts_argmin,
                            ts_argmax, ts_rank, ts_sum, ts_product, ts_stddev]
feature_function_set_test = [delay, correlation, covariance, scale, delta, signedpower, decay_linear, ts_min, ts_max, ts_argmin,
                            ts_argmax, ts_rank, ts_sum, ts_product, ts_stddev]
