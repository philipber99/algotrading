import pprint, traceback, copy,  logging, signal, threading
from datetime import datetime, timedelta, time
import time as ptime
from time import time_ns

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from algotrading.trading.Objects import TradeInstruction, Order
from algotrading.trading.TradeManagement import create_trade_instructions_from_results, \
                                                execute_trade_instructions, close_trade, init_static_trade_quantities, \
                                                check_trade_instructions
from algotrading.trading.OrderManagement import process_order_notification, expire_orders, execute_orders, check_orders_is_opening_order_flag, \
                                                create_closing_leg, create_closing_oco
from algotrading.trading.Validations import check_circuit_breakers
from algotrading.refdata import ServerStreamingPrices
from algotrading.trading.Pricing import start_pricing_server, pricing_maintenance, get_bidask
from algotrading.analysis.lib.Execution import perform_analysis
from algotrading.trading.TestBrokerInterface import TestBrokerInterface
from algotrading.trading.Pricing import get_symbol_mark

from algotrading.ContextFields import *
from algotrading.Common import apply_configuration_overrides, analysis_load_data, get_executor_context
import algotrading.refdata.MarketData as MarketData

PARAM_POLL_ORDERS_INTERVAL = "trading.order_polling_time"

dividend_cache_df = None

sigint = threading.Event()
sigint.clear()

def signal_handler(sig, frame):
    global sigint
    sigint.set()

signal.signal(signal.SIGINT, signal_handler)
   
   
def check_results( context ):
    
    results = context[PARAM_ALGO_RESULTS]

    if len(results) > 0:    
        if context.get('trading.continuous.enable'):
            remove_idxs = []
            
            results = results.set_index(['Date',PARAM_ANALYSIS_STOCK_TICKERS])
            cutoff_dt = datetime.now() - timedelta(seconds=context.get('trading.continuous.prediction_cutoff',120))
            for idx in results.index:
                res_date,symbol = idx
                if res_date.date() < cutoff_dt.date():
                    context[PARAM_LOGGER].warning("Old prediction encountered: {}".format(results.loc[res_date]))
                    remove_idxs.append(idx)
                    
            results = results.drop(remove_idxs)
            context[PARAM_ALGO_RESULTS] = results.reset_index()
                   
        else:
            results = results.set_index('Date')
            bad_result_dates = results.loc[results.index != pd.Timestamp(datetime.now().date())]
            
            if MarketData.is_trading_day_today(context):
                if len(results) == 0:
                    context[PARAM_LOGGER].warn("Algo results has 0 entries")
                
                if len(bad_result_dates) > 0:
                    context[PARAM_LOGGER].warn("Results with wrong prediction date")
                    context[PARAM_LOGGER].warn(bad_result_dates.to_string())
            
                return len(bad_result_dates) == 0
        
    return True


test_trade_submitted = False
def analyze_and_create_trades( context, suppress_trades = False ):
    global test_trade_submitted
    
    if not suppress_trades:
        stop_time = context.get("trading.algo_stop_time",None)
        if stop_time is not None and \
            datetime.now() > datetime.combine(datetime.now().date(),datetime.strptime(stop_time,"%H:%M:%S").time()):
            return  
        
        start_time = context.get("trading.algo_start_time",None)
        if start_time is not None and \
            datetime.now() < datetime.combine(datetime.now().date(),datetime.strptime(start_time,"%H:%M:%S").time()):
            return  

    try:     
        logger = context[PARAM_LOGGER]
        context[PARAM_LOGGER] = context['algo_logger']
    
        remove_params = [PARAM_BROKER_INTERFACE]
        orig_params = {}
    
        if context.get('trading.test.live_test',False):
            if not test_trade_submitted:
                current_price = get_symbol_mark( context, context['trading.test.test_symbol'] )
                stop_limit_bandwidth = context.get('trading.test.stop_limit_bandwidth',0.001)
                test_rec = { 
                             "Date" : datetime.now(),
                             "StockTickers" : context['trading.test.test_symbol'],
                             "side" : 1,
                             "TradeInstruction.close_out_date" : datetime.now()+timedelta(seconds=600),
                             "TradeInstruction.expire_at" : datetime.now()+timedelta(seconds=120),
                             "TradeInstruction.notes" : "test trade",
                             "TradeInstruction.stop_gain" : round(current_price + current_price*stop_limit_bandwidth,2),
                             "TradeInstruction.stop_loss" : round(current_price - current_price*stop_limit_bandwidth,2)
                             } 
                context[PARAM_ALGO_RESULTS] = pd.DataFrame([test_rec])
                
                test_trade_submitted = True
            else:
                context[PARAM_ALGO_RESULTS] = pd.DataFrame()
        else:
            if context.get(PARAM_ALGO_EXECUTION_TREE,None) is not None:
                if PARAM_STREAMING_PRICES in context.keys() and \
                    not isinstance(context[PARAM_STREAMING_PRICES], ServerStreamingPrices.ServerStreamingPrices):
                        remove_params = remove_params + [PARAM_STREAMING_PRICES]
        
                for param in remove_params:
                    if param in context.keys():
                        orig_params[param] = context[param]
                        del context[param]
                
                calc_context = get_executor_context(context,set_orig=True)
                
                try:
                    if context.get(PARAM_LOAD_DATA,None) is not None:
                        new_context = analysis_load_data( calc_context[PARAM_LOAD_DATA], context )
                        context[PARAM_ALGO_RESULTS] = new_context[PARAM_ALGO_RESULTS]
                    else:
                        logger.debug("Performing algo analysis")
                        context['trading.algo_test_run'] = suppress_trades
                        context = perform_analysis( calc_context, persist_data = context[PARAM_PERSIST_RESULTS_DATA], force_rerun = True )
                        if PARAM_ANALYSIS_STOCK_TICKERS in context[PARAM_ALGO_RESULTS].columns:
                            tickers = list(context[PARAM_ALGO_RESULTS][PARAM_ANALYSIS_STOCK_TICKERS].unique())
                                            
                        secondary_algo_specs = context.get("trading.secondary_algo_specs",[])
                        for i in range(len(secondary_algo_specs)):
                            logger.info("Performing secondary algo analysis for {}".format(context["secondary_algos"][i]))
                            secondary_context = copy.deepcopy(context)
                            secondary_algo_specs[i](secondary_context)
                            secondary_context = apply_configuration_overrides(secondary_context)
                            secondary_context = perform_analysis( secondary_context, persist_data = secondary_context[PARAM_PERSIST_RESULTS_DATA] )
                            
                            if PARAM_ANALYSIS_STOCK_TICKERS in secondary_context[PARAM_ALGO_RESULTS].columns:
                                tickers = tickers + list(secondary_context[PARAM_ALGO_RESULTS][PARAM_ANALYSIS_STOCK_TICKERS].unique())
                            context[PARAM_ALGO_RESULTS] = pd.concat( [context[PARAM_ALGO_RESULTS],secondary_context[PARAM_ALGO_RESULTS]] )
                            context['active_stocks'] = tickers
                finally:
                    if context.get(PARAM_ALGO_EXECUTION_TREE,None) is not None:
                        for param_name,value in orig_params.items():
                            context[param_name] = value
            elif PARAM_POLL_ALGO_INSTRUCTIONS in context:
                context[PARAM_POLL_ALGO_INSTRUCTIONS](context)
            else:
                context[PARAM_ALGO_ITERATION_FUNC](context)
    finally:
        context[PARAM_LOGGER] = logger
        
    if len(context[PARAM_ALGO_RESULTS]) > 0:
        logger.debug(context[PARAM_ALGO_RESULTS])
    #logger.debug("Algo analysis complete")

    if not check_results( context ):
        raise Exception( "Results contain dates that are not for today" )
    elif not suppress_trades and \
        context[PARAM_BROKER_INTERFACE].is_market_open():
        create_trade_instructions_from_results( context )
        

def check_all_trades_completed( context ):
    
    # call this to ensure that any retry trades have the opportunity to execute
    if not context.get("trading.only_process_notifications",False):
        execute_trade_instructions( context )

    instructions = TradeInstruction.get(context, state=TradeInstruction.non_terminal_states)
    orders = Order.get(context,state=Order.non_terminal_states)
    
    hung_instructions = []
    
    instructions_complete = True
    for instruction in instructions:
        if instruction.fields["state"] not in [TradeInstruction.S_HOLDING, TradeInstruction.S_PARTIAL_HOLDING]:
            if instruction.fields['state'] != TradeInstruction.S_CLOSE_PENDING or \
                 instruction.fields['close_out_date'] is not None and \
                 instruction.fields['close_out_date'].date() == datetime.now().date():
                instructions_complete = False
                hung_instructions.append(instruction)
        elif context.get("trading.test.live_test",False) and \
                instruction.fields["symbol"] == context.get("trading.test.test_symbol","") and \
                instruction.fields["state"] == TradeInstruction.S_HOLDING:
            instructions_complete = False
        elif instruction.fields['state'] == TradeInstruction.S_HOLDING and \
            instruction.fields['close_out_date'] is not None and \
            instruction.fields['close_out_date'].date() == datetime.now().date():
            instructions_complete = False

    for order in copy.copy(orders):
        if order.fields["time_in_force"] == Order.TIF_GTC and order.fields["state"] == Order.S_WORKING and \
            (order.fields["expire_at"] is None or order.fields["expire_at"].date() != datetime.now().date()):
            orders.remove(order)        # order should not be used in determining all orders are complete
    
    todays_close = context[PARAM_BROKER_INTERFACE].get_todays_close()
    after_close = datetime.now() > context[PARAM_BROKER_INTERFACE].get_todays_close()
    continuous_trading = context.get('trading.continuous.enable',False)
    
    if datetime.now() > todays_close:
        open_tis = TradeInstruction.get(context, state=TradeInstruction.S_OPEN)
        for ti in open_tis:
            ti.fields['state'] = TradeInstruction.S_CANCELLED
            ti.update()
                                    
    if not continuous_trading:
        if todays_close != None and datetime.now() >= (context[PARAM_BROKER_INTERFACE].get_todays_close()+timedelta(minutes=1)):
            context[PARAM_LOGGER].critical( "Algo still running after market has closed." )
            instructions_complete = True
        elif todays_close == None:
            context[PARAM_LOGGER].critical( "Today's close is None" )
            
        if context.get("trading.max_run_time",None) is not None:
            process_uptime = datetime.now() - context['trading.process_start_time']
            if process_uptime > timedelta(minutes=context["trading.max_run_time"]):
                context[PARAM_LOGGER].critical("Max run time exceeded. Forcing exit.")
                return True
    elif datetime.now() > (todays_close + \
                             timedelta(seconds=context.get('trading.continuous.max_run_time_post_close',120))):
        context[PARAM_LOGGER].critical("Max run time exceeded. Forcing exit.")
        return True
    
    if "trading.shutdown_time" in context:
        shutdown_dt = datetime.combine(datetime.now().date(),datetime.strptime(context["trading.shutdown_time"], "%H:%M").time())
        if datetime.now() > shutdown_dt:
            if len(orders) > 0:
                order_ids = [order.fields['internal_order_id'] for order in orders]
                context[PARAM_LOGGER].warning("{} orders still outstanding: {}".format(len(orders), order_ids))
            elif not instructions_complete:
                context[PARAM_LOGGER].warning("Incomplete trade instructions still running")
            return True
    
    after_algo_stop = "trading.algo_stop_time" not in context or \
                        datetime.now() > datetime.combine(datetime.now().date(),datetime.strptime(context["trading.algo_stop_time"],"%H:%M:%S").time())
    
    notify_mode_only = context.get("trading.only_process_notifications",False)
    
    return (notify_mode_only or (after_close or not continuous_trading or continuous_trading and after_algo_stop) and \
            instructions_complete) and len(orders) == 0
    

def order_notification_loop( context ):

    broker = context[PARAM_BROKER_INTERFACE]
    order_notification_timeout = 0 if PARAM_TIME_MACHINE in context else context[PARAM_ORDER_NOTIFICATION_TIMEOUT]
    
    order = broker.retrieve_order_update(block=True,timeout=order_notification_timeout)
    while ( order is not None ):
        process_order_notification( context, order )
        order = broker.retrieve_order_update(block=True,timeout=order_notification_timeout)    

def poll_for_order_updates( context ):
    
    logger = context[PARAM_LOGGER]
    broker = context[PARAM_BROKER_INTERFACE]

    # get broker orders
    now_dt = datetime.now()
    lookback = context[PARAM_DATABASE_HISTORY_LOOKBACK]    
    broker_orders = broker.get_orders( status=None, from_dt = datetime.combine(now_dt, time(0,0))-timedelta(days=lookback),
                                                           to_dt = now_dt )

    # find each broker order record in the database
    for broker_order in broker_orders:
        try:
            if broker_order.fields["symbol"] in context.get("trading.symbol_filter",[]):
                logger.info("Symbol {} filtered. Skipping polled order update.".format(broker_order.fields["symbol"]))
                continue
            
            # check that there is a single db record
            db_orders = Order.get(context, broker_order_id=broker_order.fields["broker_order_id"])
            
            order_notify = False
            if len(db_orders) == 0:
                order_notify = True
            elif len(db_orders) == 1:
                field_faults = db_orders[0].compare(broker_order,broker_fields=True)
                if len(field_faults) > 0:
                    order_notify = True

            if order_notify:
                process_order_notification( context, broker_order )    
            
        except Exception as e:
            logger.error("Unable to reconcile broker order: {0}".format(broker_order.fields["broker_order_id"]) )
            logger.error(traceback.format_exc())



def check_order_leg_integrity(context,last_validate_check):
    
    logger = context[PARAM_LOGGER]
    broker = context[PARAM_BROKER_INTERFACE]
    
    # check for orphaned held orders 
    held_orders = Order.get(context,state=Order.S_HELD)
    for held_order in held_orders:
        if held_order.fields['trade_instruction_id'] is not None:
            ti_orders = Order.get(context,trade_instruction_id=held_order.fields['trade_instruction_id'])
            num_working = [ti_order for ti_order in ti_orders if ti_order.fields['state'] == Order.S_WORKING]
            if len(num_working) == 0:
                context[PARAM_BROKER_INTERFACE].cancel_order( held_order.fields["broker_order_id"] )
    
    # check TI legs are there
    skip_tis = []   # tis that have been processed
    for ti in TradeInstruction.get(context,state=TradeInstruction.non_terminal_states):          
        if ti.fields['trade_instruction_id'] not in skip_tis:
            if ti.fields['state'] in [TradeInstruction.S_CLOSE_PENDING,TradeInstruction.S_PARTIAL_CLOSED,TradeInstruction.S_HOLDING]:      
                # if there's no position and there was/is a closing order, then ensure all orders are terminal
                ti_pos = ti.get_open_position_quantity()
                if ti_pos == 0:
                    running_orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'],
                                               state=list(set(Order.non_terminal_states) - set([Order.S_HELD])),
                                               side="B" if ti.fields['side'] == "S" else "S")
                    
                    if len(running_orders) > 0:
                        logger.info(f"Trade Instruction {ti.fields['trade_instruction_id']} position is 0 with a close order attached, but there are still running orders: {[running_order.fields['internal_order_id'] for running_order in running_orders]}")
                        logger.info(f"Closing all orders for TI {ti.fields['trade_instruction_id']}")
                        close_trade(context,ti)
                elif abs(ti_pos) < ti.fields['quantity']:
                    # we have partially executed. close out the trade as the remaining quantity may not be filled
                    running_orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'],
                                               state=list(set(Order.non_terminal_states) - set([Order.S_HELD])),
                                               side="B" if ti.fields['side'] == "S" else "S")
                    
                    not_market_count = 0
                    for running_order in running_orders:
                        if running_order.fields['order_type'] != Order.TYPE_MARKET:
                            not_market_count += 1
                    
                    # any non market orders means this trade is not being actively closed, so close it
                    if not_market_count > 0:
                        logger.info(f"TI {ti.fields['trade_instruction_id']}, ti pos: {ti_pos}, ti quantity: {ti.fields['trade_instruction_id']}. Closing out trade as TI is partially closed.")
                        logger.debug(pprint.pformat(ti.fields))
                        close_trade(context,ti)
                     
                elif abs(ti_pos) == ti.fields['quantity']:
                    try:
                        # do not process any TIs that have expired (these are in process of being cleaned up elsewhere)
                        if ti.fields['close_out_date'] <= datetime.now():
                            continue
                        
                        # check that held position is the right side
                        if ti.fields['side'] == "B" and ti_pos > 0 or \
                            ti.fields['side'] == "S" and ti_pos < 0:
                            
                            symbol = ti.fields['symbol']
                            bidask = get_bidask(context,symbol)
                            
                            # check if trade should be closed
                            open_order = Order.get(context,internal_order_id=ti.fields['open_order_id'])[0]
                            gain_met = open_order.fields['bkt_stop_gain'] is not None and \
                                        (open_order.fields['side'] == "B" and \
                                         bidask.loc[symbol]['bid'] > open_order.fields['bkt_stop_gain'] or
                                         open_order.fields['side'] == "S" and \
                                         bidask.loc[symbol]['ask'] < open_order.fields['bkt_stop_gain'])
                                
                            loss_met = open_order.fields['bkt_stop_loss'] is not None and \
                                        (open_order.fields['side'] == "B" and \
                                         bidask.loc[symbol]['bid'] < open_order.fields['bkt_stop_loss'] or
                                         open_order.fields['side'] == "S" and \
                                         bidask.loc[symbol]['ask'] > open_order.fields['bkt_stop_loss'])
                            
                            if gain_met or loss_met:
                                logger.info(f"Trade Instruction {ti.fields['trade_instruction_id']} has a non-zero position but gain or loss has been met. Closing trade.")
                                close_trade(context,ti)
                            else:
                                # check if we use OTO or separate stop/limit
                                if context.get('trading.bracket_recreation_order_type','oco') == 'oco' and \
                                    ti.fields['stop_loss'] is not None and \
                                    ti.fields['stop_gain'] is not None:
                                    # check that we have 1 stop and 1 limit
                                    stop_orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'],
                                                               state=Order.non_terminal_states,
                                                               side="B" if ti.fields['side'] == "S" else "S",
                                                               order_type=Order.TYPE_STOP)
                                    limit_orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'],
                                                               state=Order.non_terminal_states,
                                                               side="B" if ti.fields['side'] == "S" else "S",
                                                               order_type=Order.TYPE_LIMIT)
                                    
                                    if len(stop_orders) == 0 or \
                                        len(limit_orders) == 0:
                                        # cancel any orders outstanding
                                        if len(stop_orders) > 0:
                                            for stop_order in stop_orders:
                                                broker.cancel_order(stop_order.fields['broker_order_id'])
                                        if len(limit_orders) > 0:
                                            for limit_order in limit_orders:
                                                broker.cancel_order(limit_order.fields['broker_order_id'])
                                        
                                        # create OTO
                                        create_closing_oco( context, ti )
                                else:
                                    # check that stop/limits are in place
                                    if ti.fields['stop_loss'] is not None:
                                        stop_orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'],
                                                                   state=Order.non_terminal_states,
                                                                   side="B" if ti.fields['side'] == "S" else "S",
                                                                   order_type=Order.TYPE_STOP)
                                        if len(stop_orders) == 0:
                                            logger.info(f"Creating missing stop order for ti {ti.fields['trade_instruction_id']}")
                                            create_closing_leg( context, ti, Order.TYPE_STOP )
                                       
                                    if ti.fields['stop_gain'] is not None: 
                                        limit_orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'],
                                                                   state=Order.non_terminal_states,
                                                                   side="B" if ti.fields['side'] == "S" else "S",
                                                                   order_type=Order.TYPE_LIMIT)
                                        if len(limit_orders) == 0:
                                            logger.info(f"Creating missing limit order for ti {ti.fields['trade_instruction_id']}")
                                            create_closing_leg( context, ti, Order.TYPE_LIMIT )
                        else:
                            logger.warning(f"Trade Instruction {ti.fields['trade_instruction_id']} has position {ti_pos} which is the incorrect side")
                            logger.warning(f"Closing trade {ti.fields['trade_instruction_id']}")
                            close_trade(context,ti)
                    except:
                        logger.error(f"Exception on trade instruction {ti.fields['trade_instruction_id']}. Closing trade.")
                        logger.error(traceback.format_exc())
                        close_trade(context,ti)
            
                 
        skip_tis.append(ti.fields['trade_instruction_id'])
    
    
def start_processing( context, algo_func = None, execution_mode_only = False ):
    
    global dividend_cache_df, bid_ask_df, sigint
    
    broker = context[PARAM_BROKER_INTERFACE]

    continuous_enable = context.get("trading.continuous.enable",False)
    continuous_loop_delay = context.get("trading.continuous.loop_delay",0.01)
    continuous_tm_shift = context.get("trading.continuous.tm_shift",30)
    continuous_tm_shift_snap = context.get("trading.continuous.tm_shift_snap",True)
    
    t_executor = None
    if continuous_enable:
        t_executor = ThreadPoolExecutor(1)
    
    micro_tm_shift_window = context.get("trading.continuous.micro_tm_shift_window",None)
    if micro_tm_shift_window is not None:
        for i in range(len(micro_tm_shift_window)):
            micro_tm_shift_window[i] = datetime.combine(datetime.now(),datetime.strptime(micro_tm_shift_window[i],"%H:%M").time())
    micro_tm_shift = context.get("trading.continuous.micro_tm_shift",5)

    def validate_state( last_validate_check ):
        
        # hack because Alpaca does not return "canceled" on bracket leg orders, so we need to manually pull them up
        if context.get(PARAM_POLL_ORDERS_INTERVAL,-1) != -1 and \
            ((datetime.now() - last_validate_check) > timedelta(seconds=context[PARAM_POLL_ORDERS_INTERVAL]) or \
             broker.has_state_changed()):
            #context[PARAM_LOGGER].debug("Performing order state validation check")
            broker.reset_state_changed_flag()
            poll_for_order_updates(context)
            
            # check open order flags if needed
            if context.get('trading.check_open_order_flag',False):
                check_orders_is_opening_order_flag(context)
                
            # validate order leg integrity of LT_STOP and LT_BRACKET orders
            check_order_leg_integrity(context,last_validate_check)
            
            return datetime.now()
        
        return last_validate_check

    def open_objects_for_processing( context ):
        open_orders = Order.get(context,state=Order.S_OPEN)
        for order in open_orders:
            if order.fields['execute_after'] is None or \
                order.fields['execute_after'] <= datetime.now():
                return True
        
        open_tis = TradeInstruction.get(context,state=TradeInstruction.S_OPEN)
        num_open_tis = len(open_tis)
        
        for ti in open_tis:
            if ti.fields['activation_date'] is not None and \
                ti.fields['activation_date'] > datetime.now():
                num_open_tis = num_open_tis - 1

        return num_open_tis > 0

    def processing_actions( last_validate_check ):
        
        order_notification_loop( context )
        
        market_close_dt = context[PARAM_BROKER_INTERFACE].get_todays_close()
        while True:
            if not context.get("trading.only_process_notifications",False):
                execute_trade_instructions( context )
                expire_orders( context )
                execute_orders( context )
                
            order_notification_loop( context )

            if context.get(PARAM_POLL_ORDERS_INTERVAL,-1) != -1: 
                last_validate_check = validate_state(last_validate_check)            
            
            if not context.get("trading.only_process_notifications",False):
                check_trade_instructions( context )
                execute_trade_instructions( context ) # required otherwise retry trades may be missed as they will be in the CANCELLED state

            if not open_objects_for_processing(context) or datetime.now() > market_close_dt:
                break
                
        
        """
        # using OTO for this now
        # if stop loss only, we create orders manually. otherwise we already have bracket orders created
        if PARAM_GTC_STOP_LOSS_THRESHOLD in context.keys() and PARAM_GTC_STOP_GAIN_THRESHOLD not in context.keys():
            create_stop_loss_orders( context )
        """
        
        # used to speed up regression            
        if PARAM_TIME_MACHINE in context and not continuous_enable:
            context[PARAM_TIME_MACHINE].shift(context.get("time_machine.shift",300))
            
        return last_validate_check

    pricing_enabled = context.get("trading.pricing.enable",False)
    
    if pricing_enabled:
        start_pricing_server(context)

    if context.get('trading.enforce_static_trade_quantity',False):
        init_static_trade_quantities( context )
    
    mkt_open_dt, mkt_close_dt = context[PARAM_BROKER_INTERFACE].get_market_hours(datetime.now())

    last_cb_check = None
    dividend_cache_size = context.get('trading.dividends.cache_size',30)                
    dividend_cache_df = MarketData.get_dividend_dates(context, context['active_stocks'],
                                                   datetime.now() - timedelta(days=1),
                                                   datetime.now() + timedelta(days=dividend_cache_size))
            
    cpu_saver = context.get('trading.continuous.cpu_saver',False)
    cpu_saver_adj = context.get('trading.continuous.cpu_saver_adj_ms',50)/1000
    last_loop_start_time_ns = time_ns()
    try:
        last_validate_check = datetime.now()
        
        if not execution_mode_only and not continuous_enable:
            algo_func(context)
            algo_func = None
            
        last_validate_check = processing_actions(last_validate_check)
        
        while not check_all_trades_completed(context) and not sigint.is_set():
            if pricing_enabled and \
                PARAM_TIME_MACHINE not in context:
                pricing_maintenance(context)
            
            if not execution_mode_only and continuous_enable:
                if algo_func is not None:
                    algo_future = t_executor.submit(algo_func,context)
                    algo_future.result(context.get("trading.continuous.result_timeout",60))     # required in case the algo hangs (eg. dask issue)
                
            if type(context[PARAM_BROKER_INTERFACE]) is TestBrokerInterface:
                context[PARAM_BROKER_INTERFACE].update_open_orders(dispatch=True)

            last_validate_check = processing_actions(last_validate_check)
            
            # circuit breaker checks are expensive so we need to space them out
            if last_cb_check is None or \
                last_cb_check + timedelta(seconds=context.get('trading.circuit_breakers_check_interval',5)) < datetime.now():
                check_circuit_breakers(context)
                last_cb_check = datetime.now()
    
            if PARAM_TIME_MACHINE in context and continuous_tm_shift > 0:
                if datetime.now() < mkt_open_dt:
                    context[PARAM_TIME_MACHINE].shift((mkt_open_dt - datetime.now()).seconds)
                else:
                    if micro_tm_shift_window is not None and \
                        datetime.now() >= micro_tm_shift_window[0] and \
                        datetime.now() <= micro_tm_shift_window[1]:
                        context[PARAM_TIME_MACHINE].shift(micro_tm_shift)
                    else:
                        dt_now = datetime.now()
                        
                        target_dt = dt_now.replace(microsecond=0) + timedelta(seconds=continuous_tm_shift)
                        
                        if continuous_tm_shift_snap:
                            target_dt = dt_now.replace(microsecond=0) + timedelta(seconds=continuous_tm_shift-(target_dt.second%continuous_tm_shift))
                        else:
                            target_dt = dt_now.replace(microsecond=0) + timedelta(seconds=continuous_tm_shift)
                        
                        if target_dt <= dt_now:
                            target_dt = dt_now + timedelta(seconds=continuous_tm_shift)

                        pending_orders = Order.get(context,
                                                   creation_date=(dt_now.replace(hour=0,second=0,microsecond=0),
                                                                  dt_now), 
                                                   state=Order.non_terminal_states,
                                                   expire_at=(dt_now,target_dt-timedelta(seconds=1)))
                        
                        for order in pending_orders:
                            if order.fields['expire_at'] < target_dt:
                                target_dt = order.fields['expire_at'] 

                        pending_tis = TradeInstruction.get(context,
                                                   creation_date=(dt_now.replace(hour=0,second=0,microsecond=0),
                                                                  dt_now), 
                                                   state=TradeInstruction.non_terminal_states,
                                                   close_out_date=(dt_now,target_dt-timedelta(seconds=1)))
                        
                        for ti in pending_tis:
                            if ti.fields['close_out_date'] < target_dt and \
                                target_dt > dt_now:
                                target_dt = ti.fields['close_out_date'] 
                        
                        if continuous_tm_shift_snap:
                            if dt_now < target_dt.replace(second=0,microsecond=0):
                                target_dt = target_dt.replace(second=0,microsecond=0)
                        
                        if target_dt <= dt_now:
                            target_dt = dt_now + timedelta(seconds=continuous_tm_shift)
                        
                        context[PARAM_TIME_MACHINE].shift((target_dt - dt_now).seconds)
                        
            elif continuous_loop_delay > 0:
                ptime.sleep(continuous_loop_delay)
                
            if type(context[PARAM_BROKER_INTERFACE]) is TestBrokerInterface:
                context[PARAM_BROKER_INTERFACE].update_open_orders(dispatch=True)
                order_notification_loop( context )
            
            if cpu_saver:
                loop_time = (time_ns() - last_loop_start_time_ns) / 1000000000
                
                if loop_time < (1-cpu_saver_adj):       # ensure we tick at least every minute
                    ptime.sleep(cpu_saver_adj)
                    
            last_loop_start_time_ns = time_ns()
                        
    except:
        context[PARAM_LOGGER].critical(traceback.format_exc())
        context[PARAM_LOGGER].critical("Exception in main processing loop. Closing orders/trades and exiting.")
        execution_mode_only = True
        cancel_all_orders( context )
        close_all_positions( context )
    
    if sigint.is_set():
        context[PARAM_LOGGER].warning("SIGINT received. Process exiting gracefully.")
    
    return True



def cancel_all_orders( context ):
    broker = context[PARAM_BROKER_INTERFACE]
    logger = context[PARAM_LOGGER]

    logger.info("cancel_all_orders: cancelling all orders for the past 5 days")
    orders = broker.get_orders( from_dt=datetime.combine(datetime.now().date(),time.min)-timedelta(days=5),
                                to_dt=datetime.combine(datetime.now().date(),time.max) )    

    for order in orders:
        try:
            if order.fields["state"] in Order.non_terminal_states and \
                order.fields['state'] != Order.S_PENDING_CANCEL:
                broker.cancel_order( order.fields["broker_order_id"] )
                
                broker_orders = Order.get(context,broker_order_id=order.fields['broker_order_id'])
                if len(broker_orders) > 0:
                    for broker_order in broker_orders:
                        broker_order.fields['state'] = broker_order.S_PENDING_CANCEL
                        broker_order.update()
        except:
            logger.warning("Error cancelling broker order {}".format(order.fields["broker_order_id"]))
            logger.warning(traceback.format_exc())
    
    execute_orders(context)
    
    logger.info("cancel_all_orders: complete")


def close_all_positions( context ):
    broker = context[PARAM_BROKER_INTERFACE]
    logger = context[PARAM_LOGGER]
    
    logger.info("close_all_positions: retrieving positions and cancelling")    
    
    open_positions = broker.get_open_positions()
    for symbol,position in open_positions.items():
        if symbol in context.get("trading.symbol_filter",[]):
            logger.info("Symbol {} filtered. Position will not be closed.".format(symbol))
            continue
        
        order = Order.new( context, 0 )
        order.fields["side"] = "B" if position.fields["quantity"] < 0 else "S"
        order.fields["order_type"] = Order.TYPE_MARKET
        order.fields["quantity"] = abs(position.fields["quantity"])
        order.fields["is_opening_order"] = 0
        order.fields["symbol"] = position.fields["symbol"]
        order.fields["state"] = Order.S_OPEN
        
        try:
            broker.place_order( order )
        except:
            logger.warning("unable to place closing order for symbol: {}".format(order.fields["symbol"]))
            logger.warning(traceback.format_exc())
    
    execute_orders(context)
    
    # clean up any created but unexecuted orders
    created_orders = Order.get(context,state=Order.S_CREATED)
    for order in created_orders:
        order.fields['state'] = Order.S_CANCELLED
        order.update()

   
def close_all_trades( context ):
    logger = context[PARAM_LOGGER]

    logger.info("close_all_trades: closing all open trades")
    
    active_trades = TradeInstruction.get(context, state=TradeInstruction.non_terminal_states)
    
    for trade in active_trades:
        close_trade(context,trade)
        
    context["trading.immediate_trade_execution"] = True
    
    logger.info("close_all_trades complete")


def create_basic_order(context,ti_id,is_open):
    instruction = TradeInstruction.get(context,trade_instruction_id=ti_id)[0]
    
    order = Order.new( context, trade_instruction_id=ti_id)
    order.fields["state"] = Order.S_OPEN
    order.fields["symbol"] = instruction.fields["symbol"]
    order.fields["algo_id"] = instruction.fields["algo_id"]
    order.fields['side'] = "B" if is_open and instruction.fields['side'] == "B" or \
                                  not is_open and instruction.fields['side'] == "S" else "S"
    order.fields['order_type'] = Order.TYPE_MARKET
    order.fields['expire_at'] = instruction.fields['close_out_date']
    order.fields['is_opening_order'] = is_open
    if order.fields['expire_at'].date() > datetime.now().date():
        order.fields['time_in_force'] = Order.TIF_GTC
    else:
        order.fields['time_in_force'] = Order.TIF_DAY

    return order


def add_bracket_to_trade(context,ti_id,low_exit,high_exit):
    
    instruction = TradeInstruction.get(context,trade_instruction_id=ti_id)[0]
    orders = Order.get(context,trade_instruction_id=ti_id,state=Order.non_terminal_states)
    
    if len(orders) > 0:
        context[PARAM_LOGGER].warning(f"Unable to add bracket as there are active orders on TI {ti_id}")
        return False
    
    pos_qty = instruction.get_open_position_quantity()
    
    stop_order = create_basic_order(context,ti_id,False)
    stop_order.fields['order_type'] = Order.TYPE_STOP
    stop_order.fields['stop_price'] = low_exit if stop_order.fields['side'] == "S" else high_exit
    stop_order.fields['quantity'] = abs(pos_qty)
    
    limit_order = create_basic_order(context,ti_id,False)
    limit_order.fields['order_type'] = Order.TYPE_LIMIT
    limit_order.fields['limit_price'] = high_exit if limit_order.fields['side'] == "S" else low_exit
    limit_order.fields['quantity'] = abs(pos_qty)

    instruction.fields['close_order_id'] = stop_order.fields['internal_order_id']
    
    stop_order.update()
    limit_order.update()
    instruction.update()
    
    return True
    
    
     
    
    
    
    

    
    
    