import pprint, traceback, logging, functools, threading, os
from datetime import datetime, timedelta, time
import time as ptime

import numpy as np
import pandas as pd

from math import copysign
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from algotrading.refdata.MarketData import retrieve_current_price
import algotrading.tda.TDAStreamingPrices as TDAStreamingPrices

from algotrading.ContextFields import *
from algotrading.Common import import_module_attr, get_datetime
import algotrading.refdata.MarketData as MarketData
from algotrading.CommonUtils import max_time, min_time


_last_critical_reported = None
_last_warn_reported = None
_pricing_server_active = True
_current_price_data = {}
_last_price_update_dt = None
_last_reported_update_error_dt = None
_streaming_prices_interface = None
_store_todays_prices = False
_price_change_list = deque(maxlen=100000)
_is_time_machine = False

_price_history = {}
_price_history_init_lock = threading.Lock()

# changed if needed for backward compatability with the old TDA streaming interface
__last_vol_field_label = 'last_traded_volume'
__last_traded_field_label = 'last_traded_price'


def get_price_data( context, symbol ):
    global _current_price_data
    
    if PARAM_TIME_MACHINE in context:
        cur_history = get_todays_price_history(symbol)
        if len(cur_history) > 0:
            return cur_history.iloc[-1]
    elif symbol in _current_price_data:
        return _current_price_data[symbol]
        
    return None


def get_symbol_mark( context, symbol ):
    global _current_price_data, _pricing_server_active

    if PARAM_TIME_MACHINE in context:
        cur_history = get_todays_price_history(symbol)
        if len(cur_history) > 0:
            return round((cur_history.iloc[-1].bid + cur_history.iloc[-1].ask) / 2,3)
        else:
            sym_prices_df = MarketData.prices_sql(context,symbol,min_time(datetime.now()-timedelta(days=7)),max_time(datetime.now()),'1min')
            last_rec = sym_prices_df[sym_prices_df.index <= datetime.now()].iloc[-1]
            return round((last_rec['open']+last_rec['close'])/2,2)
            
    if _pricing_server_active and symbol in _current_price_data:
        ret_price = (_current_price_data[symbol]['bid']+_current_price_data[symbol]['ask'])/2
        if ret_price != 0:
            return ret_price
    
    return retrieve_current_price( context, symbol, bar = False )
    
    
def get_bidask( context, symbol ):
    global _current_price_data, _pricing_server_active
    
    if _pricing_server_active and symbol in _current_price_data:
        return pd.DataFrame( {'bid' : _current_price_data[symbol]['bid'],
                              'ask' : _current_price_data[symbol]['ask']},
                              index = [symbol])
    elif PARAM_TIME_MACHINE in context:
        cur_history = get_todays_price_history(symbol)
        if len(cur_history) > 0:
            return pd.DataFrame( {'bid' : cur_history.iloc[-1].bid,
                                  'ask' : cur_history.iloc[-1].ask },
                                  index = [symbol])        
    elif PARAM_STREAMING_PRICES in context:
        return context[PARAM_STREAMING_PRICES].get_bid_ask([symbol])

    price = get_symbol_mark(context,symbol)
    return pd.DataFrame( {'bid' : price,
                          'ask' : price },
                          index = [symbol])
    

def get_price_latency_status():
    global _last_critical_reported, _last_warn_reported
    
    if _last_critical_reported is not None:
        return "outage"
    elif _last_warn_reported is not None:
        return "delayed"
    
    return "ok"


def get_todays_price_history(symbol):
    global _price_history, _is_time_machine
    
    if symbol in _price_history:
        if _is_time_machine:
            return _price_history[symbol].set_index('timestamp').loc[:datetime.now()].reset_index()
        return _price_history[symbol]
    
    return pd.DataFrame()
    
    
def get_sanitized_price_history(symbol, last_idx = None, fill_latest_entries = False):
    
    raw_history_df = get_todays_price_history(symbol)
    if len(raw_history_df) == 0:
        return pd.DataFrame()
    
    raw_history_df = raw_history_df.set_index('timestamp')
    raw_history_df.sort_index(inplace=True)     # small possibility an update can be out of order time-wise which causes .loc to fail
    dt_now = datetime.now()
    
    # get the history since the last update at time index last_idx (which may be before last_idx)
    if last_idx is not None:
        if raw_history_df.index[0] <= last_idx:
            last_update_idx = raw_history_df.loc[:last_idx].index[-1]
            raw_history_df = raw_history_df.loc[last_update_idx:]
            if len(raw_history_df) == 0:
                return pd.DataFrame()
    
    raw_history_df.index = raw_history_df.index.astype('datetime64[ns]')
    raw_history_df.sort_index(inplace=True)
    raw_history_df = raw_history_df.resample('1s').ffill()
    raw_history_df.reset_index(inplace=True)
    raw_history_df['timestamp'] = raw_history_df['timestamp'].dt.round(freq='s')
    raw_history_df.set_index('timestamp',inplace=True)
        
    if last_idx is not None:
        raw_history_df = raw_history_df.loc[last_idx+timedelta(seconds=1):dt_now]
        if len(raw_history_df) == 0:
            return pd.DataFrame()

    if fill_latest_entries and \
        raw_history_df.index[-1] < dt_now.replace(microsecond=0)-timedelta(seconds=2):
        
        cur_idx = raw_history_df.index[-1] + timedelta(seconds=1)
        end_idx = dt_now.replace(microsecond=0) - timedelta(seconds=2)
        
        add_rows = []
        while cur_idx < end_idx:
            add_rows.append(raw_history_df.iloc[-1].rename(cur_idx))
            #raw_history_df = raw_history_df.append(raw_history_df.iloc[-1].rename(cur_idx))
            cur_idx = cur_idx + timedelta(seconds=1)
        
        raw_history_df = pd.concat([raw_history_df,pd.DataFrame(add_rows)]) 
    
    return raw_history_df.reset_index().rename(columns={'index':'timestamp'})
    

def bulk_price_update_func( context, updates ):
    for update in updates:
        price_update_func( context, update['symbol'], update )
    

def price_update_func( context, symbol, data_dict ):
    global _current_price_data, _last_price_update_dt, _last_reported_update_error_dt, \
            _store_todays_prices, _price_history, _price_history_init_lock, _price_change_list, \
            __last_vol_field_label, __last_traded_field_label
    
    dt_now = datetime.now()
    
    try:
        if symbol not in _current_price_data:
            _current_price_data[symbol] = { __last_traded_field_label : 0, __last_vol_field_label : 0, 'bid' : np.nan, 'ask' : np.nan }

        # in the unlikely case we get 2 updates in the same second, establish order through a microsecond addition
        # (TDA timestamps have no ms component)
        if type(data_dict['timestamp']) is int or \
            type(data_dict['timestamp']) is float:
            data_dict['timestamp'] = datetime.fromtimestamp(data_dict['timestamp'] // 1000)
            
        if symbol in _price_history and \
            len(_price_history[symbol]) > 0 and \
            data_dict['timestamp'] == _price_history[symbol]['timestamp'].iloc[-1].replace(microsecond=0):
            data_dict['timestamp'] = _price_history[symbol]['timestamp'].iloc[-1] + timedelta(microseconds=200)
            
        orig_price_data = _current_price_data[symbol].copy()
        
        for k,v in data_dict.items():
            if v is None or \
                type(v) is str and v == 'None':
                _current_price_data[symbol][k] = np.nan
            else:
                _current_price_data[symbol][k] = v
            
        _price_change_list.append((symbol,orig_price_data,_current_price_data[symbol].copy()))
            
        _last_price_update_dt = dt_now
        
        if _store_todays_prices and \
            ('bid' in data_dict or 'ask' in data_dict):
            
            if _price_history_init_lock is not None:
                if not _price_history_init_lock.acquire(timeout=3):
                    context[PARAM_LOGGER].error("Failed to get price history init lock. Disabling.")
                    _price_history_init_lock = None
                
            try:
                if symbol not in _price_history:
                    _price_history[symbol] = pd.DataFrame()
                
                if 'bid' in _current_price_data[symbol] and \
                    ~np.isnan(_current_price_data[symbol]['bid']) and \
                    'ask' in _current_price_data[symbol] and \
                    ~np.isnan(_current_price_data[symbol]['ask']):
                    if len(_price_history[symbol]) == 0 or \
                        _current_price_data[symbol]['timestamp'] not in _price_history[symbol]['timestamp']:
                        _price_history[symbol] = pd.concat([_price_history[symbol],
                                                                 pd.DataFrame([{"timestamp" : _current_price_data[symbol]['timestamp'],
                                                                      "bid" : float(_current_price_data[symbol]['bid']),
                                                                      "ask" : float(_current_price_data[symbol]['ask']),
                                                                      "volume" : float(_current_price_data[symbol][__last_vol_field_label]),
                                                                      "last" : float(_current_price_data[symbol][__last_traded_field_label])}])])
            finally:
                if _price_history_init_lock is not None and \
                    _price_history_init_lock.locked():
                    _price_history_init_lock.release()
                    
    except:
        if _last_reported_update_error_dt is None or \
            dt_now > (_last_reported_update_error_dt+timedelta(minutes=context.get("trading.pricing.error_reporting_freq",5))):
            context[PARAM_LOGGER].error(traceback.format_exc())
            context[PARAM_LOGGER].error("Data received in callback: {}".format(pprint.pformat(data_dict)))
            _last_reported_update_error_dt = dt_now


_price_thresholds_df = pd.DataFrame()
_all_price_thresholds = np.array([])
_price_thresholds_callbacks = {}
_price_threshold_lock = threading.Lock()


def register_price_threshold_callback( context, label, price, handler ):
    global _price_thresholds_callbacks, _price_thresholds_df, _all_price_thresholds, _price_threshold_lock
    
    with _price_threshold_lock:
        _price_thresholds_df.at[label,'price'] = price
        _price_thresholds_callbacks[label] = handler
        _all_price_thresholds = np.array(_price_thresholds_df['price'].unique())
            
    
def unregister_price_threshold_callback( context, label ):
    global _price_thresholds_callbacks, _price_thresholds_df, _all_price_thresholds, _price_threshold_lock
    
    if label in _price_thresholds_df.index:
        with _price_threshold_lock:
            _price_thresholds_df.drop(index=[label],inplace=True)
            del _price_thresholds_callbacks[label]
            _all_price_thresholds = np.array(_price_thresholds_df['price'].unique())

def get_price_threshold_callback_labels( context ):            
    global _price_thresholds_callbacks

    return list(_price_thresholds_df.index)


_label_blackouts = pd.DataFrame()

def check_for_price_events( context, executor ):
    global _price_change_list, _price_thresholds, _price_thresholds, _label_blackouts
    
    trigger_blackout_time = context.get('trading.pricing.trigger_blackout_time',10)
    
    if len(_label_blackouts) > 0:
        earliest_blackout = datetime.now()-timedelta(seconds=trigger_blackout_time)
        expire_blackouts =_label_blackouts[_label_blackouts['start_dt'] < earliest_blackout]
        _label_blackouts.drop(index=expire_blackouts.index,inplace=True)
                                
    while len(_price_change_list) > 0:
        (symbol,original_prices,updated_prices) = _price_change_list.popleft()
        if len(original_prices) > 0 and \
            (original_prices['bid'] != updated_prices['bid'] or \
             original_prices['ask'] != updated_prices['ask']) and \
             ~np.isnan(original_prices['bid']) and \
             ~np.isnan(original_prices['ask']):
            
            old_mark = (original_prices['bid'] + original_prices['ask'])/2
            new_mark = (updated_prices['bid'] + updated_prices['ask'])/2

            with _price_threshold_lock:
                active_thresholds = set(_all_price_thresholds)
                
                if len(_label_blackouts) > 0:
                    active_thresholds = active_thresholds - set(_label_blackouts['price'].unique())
                
                active_thresholds = np.sort(np.array(list(active_thresholds)))
                
                if len(active_thresholds) > 0:
                    thresholds_hit = active_thresholds[((active_thresholds > old_mark) & (active_thresholds < new_mark)) | \
                                              ((active_thresholds < old_mark) & (active_thresholds > new_mark))]
                    for threshold_price in thresholds_hit:
                        labels = _price_thresholds_df[_price_thresholds_df.price == threshold_price].index
                        for label in labels:
                            if label not in _label_blackouts.index:
                                executor.submit(_price_thresholds_callbacks[label],context,label,threshold_price,old_mark,new_mark)
                                _label_blackouts = _label_blackouts.append( pd.DataFrame( {'start_dt' : datetime.now(),
                                                                                           'price' : threshold_price}, index=[label] ) )

    

_price_event_executor = None
_pricing_exception_reported = False

def pricing_maintenance( context ):
    global _price_event_executor, _pricing_exception_reported, _last_warn_reported, _last_critical_reported, _last_price_update_dt
    
    logger = context[PARAM_LOGGER]
    exception_reported = False

    warn_threshold = context.get("pricing.stale_pricing_warning_threshold",5)
    critical_threshold = context.get("pricing.stale_pricing_critical_threshold",15)

    if _price_event_executor is None:
        _price_event_executor = ThreadPoolExecutor( context.get("trading.pricing.thread_workers",3) )   

    try:
        dt_now = datetime.now()
        
        latency = (dt_now - _last_price_update_dt).seconds
                 
        if latency > warn_threshold and \
            _last_warn_reported is None and \
             MarketData.is_trading_hours(context):
            logger.warning("Last price update received: {}".format(_last_price_update_dt))
            _last_warn_reported = dt_now
        elif latency > critical_threshold and \
            _last_critical_reported is None and \
            MarketData.is_trading_hours(context):
            logger.critical("Last price update received: {}".format(_last_price_update_dt))
            _last_critical_reported = dt_now
        elif latency < warn_threshold and latency < critical_threshold and \
                (_last_warn_reported is not None or _last_critical_reported is not None):
            _last_warn_reported = None
            _last_critical_reported = None
        
        check_for_price_events(context, _price_event_executor)
    except:
        if not _pricing_exception_reported:
            context[PARAM_LOGGER].error("Maintenance thread exception")
            context[PARAM_LOGGER].error(traceback.format_exc())
            _pricing_exception_reported = True
                        
                
    
def start_pricing_server( context ):
    global _pricing_server_active, _store_todays_prices, _streaming_prices_interface, \
            _price_history, _last_price_update_dt, _is_time_machine, __last_vol_field_label, __last_traded_field_label
    
    _store_todays_prices = context.get("trading.pricing.store_todays_prices", False)
    _last_price_update_dt = datetime.now()
    _pricing_server_active = True
    
    if PARAM_TIME_MACHINE in context:
        _is_time_machine = True
        
        dt_str = datetime.now().strftime("%Y-%m-%d")
        symbol_filter = context.get('trading.pricing.historical_symbol_filter',None)
        historical_log_paths = context['trading.pricing.historical_log_paths']
        for historical_log_path in historical_log_paths:
            fname = os.path.join( historical_log_path, f"price_log_{dt_str}.csv" )
            if os.path.exists(fname):
                price_df = pd.read_csv(fname)
                context[PARAM_LOGGER].info(f"Loaded prices from {fname}")
                if len(price_df.columns) == 5:
                    price_df.columns = ['timestamp','symbol','bid','ask','last']
                else:
                    price_df.columns = ['timestamp','symbol','bid','ask','last','volume']
        
                for col in (set(price_df.columns) - set({'timestamp','symbol'})):
                    price_df[col] = price_df[col].replace('None',np.nan).replace(0,np.nan).astype('float64')

                price_df = price_df.dropna(subset=['bid','ask'])
        
                price_df['timestamp'] = [datetime.fromisoformat(dt) for dt in price_df['timestamp']] 
                
                for symbol in price_df.symbol.unique():
                    if symbol_filter is None or \
                        symbol in symbol_filter:
                        symbol_prices_df = price_df[price_df.symbol == symbol].set_index('timestamp').sort_index().drop(columns=['symbol']).loc[dt_str+" 00:00:00":dt_str+" 12:59:59"]
                        symbol_prices_df = symbol_prices_df[~symbol_prices_df.index.duplicated()].reset_index()
                        
                        if symbol in _price_history:
                            _price_history[symbol] = pd.concat([_price_history[symbol],symbol_prices_df]).set_index('timestamp').sort_index()
                            _price_history[symbol] = _price_history[symbol][~_price_history[symbol].index.duplicated()].reset_index().copy()
                        else:
                            _price_history[symbol] = symbol_prices_df.sort_values('timestamp').copy()
            
            # if we dont have a log file for a symbol, we use minute granularity from the database
            # use with care due to many ffills -- usually needs decimation when processing downstream
            symbols_from_db = context.get('trading.pricing.time_machine.symbols_from_db',None)
            if symbols_from_db is not None:
                for symbol in symbols_from_db:
                    prices_df = MarketData.prices_sql(context,symbol, min_time(datetime.now()),max_time(datetime.now()),'1min',tm_filter_override=True)
                    if len(prices_df > 0):
                        prices_df = prices_df[(prices_df.index.time >= MarketData.std_mkt_open)&(prices_df.index.time < MarketData.std_mkt_close)]
                        prices_df['bid'] = prices_df['open']
                        prices_df['ask'] = prices_df['open']
                        prices_df['last'] = prices_df['open']
                        prices_df['volume'] = prices_df['volume'].cumsum().shift(1)
                        prices_df.drop(columns=['open','close','low','high'],inplace=True)
                        
                        prices_df = prices_df.resample('1s').ffill()
                        prices_df.index.name ='timestamp'
                        if len(prices_df) > 0:
                            _price_history[symbol] = prices_df.reset_index()
                    else:
                        context[PARAM_LOGGER].warning(f"Unable to retrieval symbol {symbol} from database")
            
        if len(_price_history) == 0:
            raise ValueError("No historical prices loaded. Check filters or log path.")      
                        
    else:
        _streaming_prices_interface = import_module_attr(context["trading.pricing.local_streaming_prices_class"])(context, context['active_stocks'],enable_streaming=True)
        
        if 'TDA' in context["trading.pricing.local_streaming_prices_class"]:
            _streaming_prices_interface.register_streaming_updates_func( \
                                    functools.partial(price_update_func,context) )
        else:
            _streaming_prices_interface.register_streaming_updates_func( \
                                    functools.partial(bulk_price_update_func,context) )
        
        # for backward compatibility
        if type(_streaming_prices_interface) is TDAStreamingPrices:
            __last_vol_field_label = 'volume'
            __last_traded_field_label = 'close'
        
        price_history_init(context)
    
    
    
def price_history_init( context ):
    global _pricing_server_active, _last_price_update_dt, _streaming_prices_interface, \
            _last_critical_reported, _last_warn_reported, _store_todays_prices, _price_history_init_lock, \
            _price_history
    
    logger = context[PARAM_LOGGER]
    ptime.sleep(3)  # to give a chance for price updates to come in
    
    if _store_todays_prices:
        prices_df = context[PARAM_STREAMING_PRICES].get_realtime_price_history()
        if prices_df is not None:
            starting_price_history = {}
            for symbol in prices_df.symbol.unique():
                starting_price_history[symbol] = prices_df[prices_df.symbol == symbol].drop(columns=['symbol']).reset_index()
            try:
                if _price_history_init_lock is None or \
                    not _price_history_init_lock.acquire(timeout=3):
                    logger.error("Failed to get price history lock. Updating anyway")
                    
                for symbol in starting_price_history.keys():
                    if symbol in _price_history:
                        all_history_df = pd.concat([starting_price_history[symbol],_price_history[symbol]])
                        _price_history[symbol] = all_history_df.set_index('timestamp').sort_index()
                        _price_history[symbol] = _price_history[symbol][~_price_history[symbol].index.duplicated()]
                        _price_history[symbol] = _price_history[symbol].reset_index()
                    else:
                        _price_history[symbol] = starting_price_history[symbol]
            finally:
                if _price_history_init_lock is not None and \
                     _price_history_init_lock.locked():
                    _price_history_init_lock.release()
                    _price_history_init_lock = None
        else:
            logger.critical("Unable to retrieve price history")
                
                
                
                


