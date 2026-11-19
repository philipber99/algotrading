import pprint, traceback
from datetime import datetime, timedelta, time
import time as ptime

import numpy as np
import pandas as pd

from algotrading.refdata.MarketData import get_trading_calendar

from algotrading.refdata import ServerStreamingPrices
from .Objects import Order, Position

from algotrading.ContextFields import *
from algotrading.Common import get_sql_connection
import algotrading.refdata.MarketData as MarketData


class ValidationError(Exception):
    pass

    
def get_current_is_opening_order_flag( context, order, positions = None, adj_positions = False ):
    if positions is None or PARAM_TIME_MACHINE in context:
        positions = context[PARAM_BROKER_INTERFACE].get_open_positions()
    
    if order.fields['symbol'] in positions:
        side_map = { "B" : int(positions[order.fields['symbol']].fields['quantity'] >= 0), 
                     "S" : int(positions[order.fields['symbol']].fields['quantity'] <= 0) }
        
        is_opening_order = side_map[order.fields['side']]
    else:
        is_opening_order = 1

    if adj_positions and order.fields['order_type'] == Order.TYPE_MARKET:
        adj_qty = (order.fields['quantity'] - order.fields['filled_qty'])
        adj_qty = adj_qty if order.fields['side'] == 'B' else -1*adj_qty
        if order.fields['symbol'] not in positions:
            new_pos = Position(context)
            new_pos.fields["symbol"] = order.fields['symbol']
            new_pos.fields["quantity"] = adj_qty
            positions[order.fields['symbol']] = new_pos
        else:
            positions[order.fields['symbol']].fields['quantity'] = positions[order.fields['symbol']].fields['quantity'] + adj_qty
        
    return is_opening_order


def check_for_market_events( context, symbol, start_dt, end_dt ):
    
    try:
        blackout_days = context.get("trading.market_events.blackout_days",2)
        blackout_start_dt = MarketData.get_relative_trading_date(context,start_dt,-1*blackout_days)
        blackout_end_dt = MarketData.get_relative_trading_date(context,end_dt,blackout_days)
       
        return len(MarketData.get_earnings_calendar(context, symbol, blackout_start_dt, blackout_end_dt)) > 0
    except:
        context[PARAM_LOGGER].error( "Unable to check for market events on symbol: {0}".format(symbol) )
        context[PARAM_LOGGER].error( traceback.format_exc() )
        
    return False

def check_for_dividend( context, symbol, start_dt, end_dt ):
    global dividend_cache_df
    
    if start_dt.date() == end_dt.date():
        return False

    if start_dt.time() != time.min:
        start_dt = datetime.combine((start_dt + timedelta(days=1)).date(),time.min)
        
    if dividend_cache_df is not None:
        div_df = dividend_cache_df[(dividend_cache_df.symbol == symbol) & \
                                (dividend_cache_df.ex_dividend_date >= start_dt.date()) &
                                (dividend_cache_df.ex_dividend_date <= end_dt.date())]
    else:
        div_df =  MarketData.get_dividend_dates( context, symbol, start_dt, end_dt, search_criteria = 'ex_dividend_date' )
    
    return len(div_df) > 0


def check_bid_ask_spread( context, symbol ):
    global bid_ask_df
    
    max_spread = context.get('trading.max_bidask_spread',None)
    
    if max_spread is not None:
        if bid_ask_df is not None and \
            symbol in bid_ask_df.index:
            bid = bid_ask_df.loc[symbol].bid
            ask = bid_ask_df.loc[symbol].ask
            
            spread = (ask - bid) / (ask + bid)
            
            return spread < max_spread
        
        if PARAM_TIME_MACHINE not in context:
            context[PARAM_LOGGER].warning( "Unable to find bid/ask spread from streaming prices" )
        
    return True
    

def get_execution_datetime( context, creation_date = None, field = PARAM_TARGET_OPEN_POSITION_TIME):
    
    if creation_date is None:
        creation_date = datetime.now()
    
    if context.get('trading.continuous.enable',False):
        if field == PARAM_TARGET_OPEN_POSITION_TIME:
            if PARAM_TARGET_OPEN_POSITION_TIME in context:
                return datetime.combine(datetime.now(),
                                          datetime.strptime(context[PARAM_TARGET_OPEN_POSITION_TIME],"%H:%M").time())
            
            return datetime.now()
        else:
            if PARAM_TARGET_CLOSE_POSITION_TIME in context:
                return datetime.combine(datetime.now(),
                                          datetime.strptime(context[PARAM_TARGET_CLOSE_POSITION_TIME],"%H:%M").time())
            
            if context.get('trading.continuous.holding_period',None) is not None:
                ret_dt = datetime.now() + timedelta(seconds=context['trading.continuous.holding_period'])
            else:
                raise ValueError("No target close time or holding period configured. Unable to determine close position time.")
                
            close_dt = context[PARAM_BROKER_INTERFACE].get_todays_close()
            return (close_dt-timedelta(seconds=context['trading.pre_close_trade_cutoff'])) if ret_dt > close_dt else ret_dt
        
    if field == PARAM_TARGET_OPEN_POSITION_TIME:
        hold_time = 0
    else:
        hold_time = context[PARAM_ALGO_TRADE_HOLDING_PERIOD]
        
    start_dt = datetime.combine(creation_date.date(),time.min)
    end_dt = start_dt+timedelta(days=2*hold_time+4)
    trade_cal_df = get_trading_calendar(context, start_dt, end_dt )        
    close_date = list(trade_cal_df['date'])[hold_time]

    return MarketData.get_market_time(context, context[field], close_date)




