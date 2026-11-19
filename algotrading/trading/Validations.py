import traceback, sys
from datetime import datetime, timedelta, date, time

import pandas as pd

from algotrading.ContextFields import *
from algotrading.refdata import MarketData
from .Objects import Order, TradeInstruction, PnlEffect
from algotrading.trading.OrderManagement import process_order_notification
from algotrading.trading.Common import ValidationError


startup_time = datetime.now()
warn_once = False

def check_circuit_breakers(context):
    global startup_time, warn_once
    
    dt_now = datetime.now()
    todays_orders = Order.get(context,creation_date=(datetime.combine(dt_now.date(),time.min),
                                           datetime.combine(dt_now.date(),time.max)))

    recent_orders = Order.get(context,creation_date=(dt_now-timedelta(seconds=context.get('trading.circuit_breaker.recent_order_window',600)),
                                                        datetime.combine(dt_now.date(),time.max)))

    todays_tis = TradeInstruction.get(context,creation_date=(datetime.combine(dt_now.date(),time.min),
                                           datetime.combine(dt_now.date(),time.max)))

    todays_pnls = PnlEffect.get(context,creation_date=(datetime.combine(dt_now.date(),time.min),
                                           datetime.combine(dt_now.date(),time.max)))
    
    # Check #1: Ensure that orders on a trade instruction id are not infinitely looping
    order_fails = [order for order in recent_orders if order.fields['state'] in [Order.S_CANCELLED,Order.S_REJECTED]]
        
    fails_by_ti = {}          
    for order in order_fails:
        if order.fields['creation_date'] > startup_time:
            if order.fields['trade_instruction_id'] in fails_by_ti:
                fails_by_ti[order.fields['trade_instruction_id']].append(order)
            else:
                fails_by_ti[order.fields['trade_instruction_id']] = [order]
    
    for ti_id,order_list in fails_by_ti.items():
        if len(order_list) > context.get("trading.circuit_breaker.max_cancel_rejects_per_ti",10):
            raise ValidationError(f"Circuit breaker: too many cancel/rejects on trade_instuction_id {ti_id}")
    
    # Check #2: No hanging CREATED trade instructions or orders
    for ti in todays_tis:
        if ti.fields['state'] == TradeInstruction.S_CREATED:
            raise ValidationError("Circuit breaker: Trade Instruction stuck in CREATED state: {}".format(ti.fields['trade_instruction_id']))
    
    for order in todays_orders:
        if order.fields['state'] == Order.S_CREATED:
            raise ValidationError("Circuit breaker: Order stuck in CREATED state: {}".format(order.fields['internal_order_id']))

    # Check #3: No position exceeds the maximum position exposure limit
    pos_exp_tolerance = context.get("trading.circuit_breaker.position_exposure_tolerance",0.25)
    
    total_exposure = { "long" : 0, "short" : 0 }
    positions = context[PARAM_BROKER_INTERFACE].get_open_positions()
    for symbol, position in positions.items():
        if position.fields['market_value'] > 0:
            total_exposure['long'] = total_exposure['long'] + position.fields['market_value']
        else:
            total_exposure['short'] = total_exposure['short'] + position.fields['market_value']
    
        if abs(position.fields['market_value']) > context[PARAM_MAX_POSITION_EXPOSURE]*(1+pos_exp_tolerance):
            raise ValidationError("Circuit breaker: Position exposure for {} exceeds exposure limits".format(position.fields['market_value']))
    
    # Check #4: Overall exposure does not exceed the exposure limit
    exp_tolerance = context.get("trading.circuit_breaker.overall_exposure_tolerance",0.25)
    if (total_exposure['long'] + total_exposure['short']) > context[PARAM_MAX_NET_EXPOSURE]*(1+exp_tolerance):
        raise ValidationError("Circuit breaker: Net exposure (long: {}, short: {}, net: {}) exceeds exposure limits"\
                              .format(total_exposure['long'],
                                      total_exposure['short'],
                                      total_exposure['long']+total_exposure['short'] ))

    if total_exposure['long'] > context[PARAM_MAX_NET_EXPOSURE]*(1+exp_tolerance) or \
        abs(total_exposure['short']) > context[PARAM_MAX_NET_EXPOSURE]*(1+exp_tolerance):
        raise ValidationError("Circuit breaker: Max total exposure (long: {}, short: {}) exceeds exposure limits"\
                                    .format(total_exposure['long'],total_exposure['short']))
    
    # Check #5: Is there a looping call into the broker API
    broker_loop_threshold = context.get('trading.circuit_breaker.broker_loop_threshold',10)
    broker_loop_lookback = context.get('trading.circuit_breaker.broker_loop_lookback',20)
    loop_check_result = context[PARAM_BROKER_INTERFACE].check_for_api_loop(datetime.now() - timedelta(seconds=broker_loop_lookback),
                                                                           broker_loop_threshold, req_id = True)
    if loop_check_result is not None:
        raise ValidationError(f"Circuit breaker: Broker API calls exceeds {broker_loop_threshold} for lookback period {broker_loop_lookback}. Offending id: {loop_check_result}")
        
    # Check #6: Prevent conditions where we keep trading at a loss
    if 'trading.symbol_blacklist' not in context:
        context['trading.symbol_blacklist'] = []
        
    if len(todays_tis) > 0 and \
        not context.get('trading.circuit_breaker.disable_winrate',False):
        ti_df = pd.DataFrame([ti.fields for ti in todays_tis])
        today_pnls_df = pd.DataFrame([pe.fields for pe in todays_pnls])
        
        if len(today_pnls_df) > 0:
            ti_pnls_df = ti_df.set_index('trade_instruction_id').join(today_pnls_df.set_index('trade_instruction_id')[['pnl']])
            ti_pnls_df = ti_pnls_df[~ti_pnls_df['pnl'].isna()]
            
            min_winrate = context.get('trading.circuit_breaker.minimum_winrate',0.50)
            sample_count = context.get('trading.circuit_breaker.minimum_winrate_sample_count',4)
            for symbol in context[PARAM_ANALYSIS_STOCK_TICKERS]:
                if symbol not in context['trading.symbol_blacklist']:
                    symbol_pnls = ti_pnls_df[(ti_pnls_df['symbol'] == symbol)]
                    if len(symbol_pnls) >= sample_count:
                        pos_pnls = symbol_pnls[symbol_pnls['pnl'] > 0]
                        neg_pnls = symbol_pnls[symbol_pnls['pnl'] < 0]
                    
                        current_winrate = (pos_pnls.shape[0])/(pos_pnls.shape[0]+neg_pnls.shape[0])
                        if current_winrate < min_winrate:            
                            context['trading.symbol_blacklist'].append(symbol)
                            context[PARAM_LOGGER].warning(f"Circuit breaker: Symbol {symbol} added to trading.symbol_blacklist. Current winrate {current_winrate:.3f} is less than minimum winrate {min_winrate}")
    
    
def check_valid_algo_runtime(context):
    broker = context[PARAM_BROKER_INTERFACE]
    logger = context[PARAM_LOGGER]
    
    if context.get("trading.continuous.enable",False):
        return True
    
    if context[PARAM_TARGET_OPEN_POSITION_TIME] == 'close':
        open_pos_time = broker.get_todays_close()
    else:
        open_pos_time = datetime.combine(datetime.now().date(),
                                         datetime.strptime(context[PARAM_TARGET_OPEN_POSITION_TIME],"%H:%M").time())
        
    if context[PARAM_TARGET_CLOSE_POSITION_TIME] == 'close':
        close_pos_time = broker.get_todays_close()
    else:
        close_pos_time = datetime.combine(datetime.now().date(),
                                          datetime.strptime(context[PARAM_TARGET_CLOSE_POSITION_TIME],"%H:%M").time())
    
    runtime_window = context.get("trading.runtime_window",5)
    
    logger.debug("Current time: {}, Open Pos Time: {}, Close Pos Time: {}, Runtime Window: {}"\
                 .format(datetime.now(),open_pos_time,close_pos_time,runtime_window))

    return datetime.now() > (open_pos_time - timedelta(minutes=runtime_window)) and \
            datetime.now() < (open_pos_time + timedelta(minutes=runtime_window))



def repair_order_state( context, db_order, broker_order, field_faults ):
   
    logger = context[PARAM_LOGGER]

    fields_repaired = True
     
    for field in field_faults:
        if field in ['state','execution_date','filled_qty','avg_fill_price']:
            logger.info("Updating DB order {0} from {1} to {2}".format(field,db_order.fields[field],broker_order.fields[field]))
            db_order.fields[field] = broker_order.fields[field]
        else:
            logger.error("Field {0} must be repaired manually".format(field))
            fields_repaired = False

    db_order.update()
   
    return fields_repaired




def check_order_db_integrity( context, skip_missing_order_legs = False ):
    
    logger = context[PARAM_LOGGER]
    
    return_result = True
    broker = context[PARAM_BROKER_INTERFACE]
    
    lookback = context[PARAM_DATABASE_HISTORY_LOOKBACK]
    
    # get broker orders
    now_dt = datetime.now()
    broker_orders = broker.get_orders( from_dt = datetime.combine(now_dt, time(0,0))-timedelta(days=lookback),
                                                           to_dt = now_dt )

    # used in finding stale orders
    second_last_trading_dt = MarketData.get_relative_trading_date( context, datetime.now(), -2 )

    # find each broker order record in the database
    broker_order_ids = []
    for broker_order in broker_orders:
        if broker_order.fields['symbol'] in context.get("trading.symbol_filter",[]):
            logger.info("Symbol {} filtered. Skipping for DB integrity checks.".format(broker_order.fields['symbol']))
            continue
        
        broker_order_ids.append(broker_order.fields["broker_order_id"])
        try:
            # check that there is a single db record
            db_orders = Order.get(context, broker_order_id=broker_order.fields["broker_order_id"])
            
            order_notify = False
            if len(db_orders) == 0:
                order_notify = True
            elif len(db_orders) == 1:
                field_faults = db_orders[0].compare(broker_order)
                if len(field_faults) > 0:
                    order_notify = True
                        
                # check for stale GTC orders
                if broker_order.fields["state"] in Order.non_terminal_states and \
                    datetime.combine(broker_order.fields["creation_date"].date(),time.min) <= second_last_trading_dt:
                    
                    if db_orders[0].fields["trade_instruction_id"] is not None:
                        ti = TradeInstruction.get(context,trade_instruction_id=db_orders[0].fields["trade_instruction_id"])[0]
                        if datetime.combine(ti.fields["creation_date"].date(),time.min) <= second_last_trading_dt:
                            logger.warn("Possible stale order: {0}. Transaction instruction may be stale: {1}".format(broker_order.fields["broker_order_id"],
                                                                                                                      db_orders[0].fields["trade_instruction_id"]))
                    else:
                        logger.warn("Stale order with no trade instruction id: {0}".format(broker_order.fields["broker_order_id"]))
            else:
                raise LookupError("Found {0} orders in the database for broker order id: {1}".format(len(db_orders),
                                         broker_order.fields["broker_order_id"]))

            if order_notify:
                process_order_notification( context, broker_order )    
            
        except Exception as e:
            if type(e) is LookupError:
                logger.error(str(e))
            else:
                logger.error("Unable to reconcile broker order: {0}".format(broker_order.fields["broker_order_id"]) )
                logger.error(traceback.format_exc())
            return_result = False
        
    # look at db orders and examine any orders not previously handled by the broker order check
    db_orders = Order.get(context, creation_date=(datetime.combine(now_dt.date(), time(0,0)),now_dt))
    for db_order in db_orders:
        try:
            # only reconcile orders that are no a non-internal state (ie. broker should have a copy)
            if db_order.fields["state"] not in Order.internal_only_states and \
                db_order.fields["state"] not in [Order.S_REJECTED] and \
                db_order.fields["quantity"] > 0:
                # check if order is missing on broker side
                if db_order.fields["broker_order_id"] is not None and db_order.fields["broker_order_id"] not in broker_order_ids:
                    logger.warn("db order {0} missing on broker (broker id {1})".format(db_order.fields["internal_order_id"],db_order.fields["broker_order_id"]))
                    # as long as positions reconcile, we'll not worry about the broker not showing an order that we know to have executed
                    #return_result = False
        except:
            logger.error("unable to reconcile db order: {0}".format(db_order.fields["internal_order_id"]) )
            logger.error(traceback.format_exc())
            return_result = False

    # ensure netted order state is up to date (can get out of sync if child order is updated but the update failed to ripple up)
    net_pending_db_orders = Order.get(context, state=Order.S_NETTED_PENDING, 
                                      creation_date=(datetime.combine(now_dt.date(), time(0,0)),now_dt))
    for order in net_pending_db_orders:
        try:
            netted_order = Order.get(context, internal_order_id=order.fields["netted_order_id"])[0]

            if netted_order.fields['state'] in Order.terminal_states:
                logger.info("Fixing NETTED_PENDING order {} when netted_order {} is in terminal state {}"\
                            .format(order.fields["internal_order_id"],netted_order.fields["internal_order_id"],
                                        netted_order.fields["state"]))
                
                order.apply_netted_order_fill(netted_order,0)
                order.update()
        except:
            logger.error("Unable to fix netted order {}".format(order.fields["internal_order_id"]))
            logger.error(traceback.format_exc())
            return_result = False
    
    # check for open orders from previous to today -- these should be cancelled
    prev_day_open_orders = Order.get(context, state=Order.S_OPEN, 
                                     creation_date=(None,datetime.combine(now_dt.date(), time(0,0))))
    for order in prev_day_open_orders:
        logger.warn("OPEN order {0} is from previous date {1}. Cancelling.".format(order.fields["internal_order_id"],order.fields["creation_date"]))
        order.fields["state"] = Order.S_CANCELLED
        order.update()

    return return_result


def check_trade_instruction_integrity( context ):

    logger = context[PARAM_LOGGER]
    
    return_result = True
    
    # figure out range to verify
    days_to_verify = context[PARAM_DATABASE_HISTORY_LOOKBACK]
    end_verify_dt = datetime.now()
    start_verify_dt = end_verify_dt - timedelta(days=days_to_verify,hours=end_verify_dt.hour,minutes=end_verify_dt.minute)
    
    # check state of all trade instructions not in a terminal state
    instructions = TradeInstruction.get(context,creation_date=(start_verify_dt,end_verify_dt))

    db_orders = Order.get(context, creation_date=(start_verify_dt,end_verify_dt))
    db_order_ids = [db_order.fields['internal_order_id'] for db_order in db_orders] 
    
    for inst_rec in instructions:
        # check that the subordinate order states are correctly reflected in the trade instruction's state
        
        try:
            for order_field in ['open_order_id','close_order_id']:
                if inst_rec.fields[order_field] is not None and \
                    inst_rec.fields[order_field] not in db_order_ids:
                    logger.warning("Order id {} missing from db for trade instruction {}".format(inst_rec.fields[order_field],
                                                                                                       inst_rec.fields['trade_instruction_id']))
                    inst_rec.fields[order_field] = None
                    inst_rec.update
            
            if inst_rec.fields["state"] == TradeInstruction.S_CREATED:
                logger.warning("Trade instruction record stuck in CREATED state. Id is: {0}".format(inst_rec.fields["trade_instruction_id"]))
            else:
                # check that the state is correct
                correct_state = inst_rec._get_correct_state()
                if correct_state != inst_rec.fields["state"]:
                    logger.warning("State mismatch for trade instruction {0}".format(inst_rec.fields["trade_instruction_id"]))
                    logger.info("Updating state from current state {0} to correct state {1}.".format(inst_rec.fields["state"],correct_state))
                    inst_rec.update_state()
                    
            if inst_rec.fields["state"] in ["OPEN","CREATED"] and \
                inst_rec.fields["creation_date"].date() < datetime.now().date():
                logger.warning("Previous day trade instruction {} with state {}. Cancelling."\
                                        .format(inst_rec.fields["trade_instruction_id"],inst_rec.fields["state"]))
                inst_rec.fields["state"] = TradeInstruction.S_CANCELLED
                inst_rec.update()
                
        except:
            logger.error("Unable to check state for trade instruction id: {0}".format(inst_rec.fields["trade_instruction_id"]))
            logger.error(traceback.format_exc())
            return_result = False
        
        try:
            net_filled_qty = inst_rec.get_open_position_quantity()
    
            if inst_rec.fields["state"] in ["CLOSED","EXECUTED","CANCELLED","CREATED","OPEN"] and \
                net_filled_qty != 0:
                logger.warning("Open position found on terminal state Trade Instruction. ID: {0}, Net Quantity: {1}." \
                                .format(inst_rec.fields["trade_instruction_id"],net_filled_qty))
                inst_rec.fields["state"] = TradeInstruction.S_HOLDING
                inst_rec.update()
                #return_result = False
        except:
            logger.error("Unable to check net position for trade instruction id: {0}".format(inst_rec.fields["trade_instruction_id"]))
            logger.error(traceback.format_exc())
            return_result = False
    
    
    return return_result


def check_broker_positions( context ):
    
    logger = context[PARAM_LOGGER]

    return_result = True
    
    broker = context[PARAM_BROKER_INTERFACE]
    pos_dict = broker.get_open_positions()
    
    lookback = context[PARAM_DATABASE_HISTORY_LOOKBACK]
    
    for (symbol,pos) in pos_dict.items():
        try:
            if symbol in context.get("trading.symbol_filter",[]):
                logger.info("Symbol {} filtered. Skipping for validations.".format(symbol))
                continue
            
            # query all non-terminal trades for this symbol
            instructions = TradeInstruction.get( context,
                                                 creation_date = (datetime.combine(datetime.now().date(),time.min) - timedelta(days=lookback), \
                                                                    datetime.now()),
                                                 symbol=symbol,
                                                 state=(TradeInstruction.non_terminal_states))
            
            total_open = 0
            for instruction in instructions:
                # get all executed orders and total the amount
                total_open = total_open + instruction.get_open_position_quantity()
            
            # if its not equal, indicate an error
            if pos.fields["quantity"] != total_open:
                logger.error("Unexpected position for {0}. Broker position: {1}. Expected position: {2}.".format(
                                            symbol, pos.fields["quantity"],total_open))
                return_result = False
            
        except:
            logger.error("Unable to reconcile positions for symbol {0}".format(symbol))
            logger.error(traceback.format_exc())
            return_result = False
    
    instructions = TradeInstruction.get( context,
                                                 creation_date = (datetime.combine(datetime.now().date(),time.min) - timedelta(days=lookback), \
                                                                    datetime.now()),
                                                 state=(TradeInstruction.non_terminal_states))
    
    skip_symbols = []
    for instruction in instructions:
        if instruction.fields["symbol"] not in pos_dict and \
            instruction.fields["symbol"] not in skip_symbols:
            open_pos_qty = instruction.get_open_position_quantity()
            if abs(open_pos_qty) > 0:
                symbol_instructions = TradeInstruction.get( context, symbol=instruction.fields["symbol"],
                                                                    state=(TradeInstruction.non_terminal_states) )
                for symbol_instruction in symbol_instructions:
                    if symbol_instruction.fields["trade_instruction_id"] != instruction.fields["trade_instruction_id"]:
                        open_pos_qty = open_pos_qty + symbol_instruction.get_open_position_quantity()
                
                if open_pos_qty == 0:
                    logger.error("broker position flat with offsetting active trade instructions")
                    logger.info("expiring all active orders on these trade instructions")
                    first_instruction = None
                    for symbol_instruction in symbol_instructions:
                        orders = Order.get(context,trade_instruction_id=symbol_instruction.fields["trade_instruction_id"],
                                                    state=Order.non_terminal_states)
                        for order in orders:
                            order.fields['expire_at'] = datetime.now()
                            order.update()
                            
                        if first_instruction is None or \
                            symbol_instruction.fields['creation_date'] < first_instruction.fields['creation_date']:
                            first_instruction = symbol_instruction
                
                    logger.info("Moving all orders to trade instruction {}".format(first_instruction.fields['trade_instruction_id']))
                    for symbol_instruction in symbol_instructions:
                        if symbol_instruction.fields['trade_instruction_id'] != first_instruction.fields['trade_instruction_id']:
                            symbol_instruction.fields['open_order_id'] = None
                            symbol_instruction.fields['close_order_id'] = None
                            symbol_instruction.fields['quantity'] = 0
                            symbol_instruction.fields['state'] = TradeInstruction.S_CANCELLED
                            symbol_instruction.update()

                            orders = Order.get(context,trade_instruction_id=symbol_instruction.fields["trade_instruction_id"])
                            
                            for order in orders:
                                order.fields['trade_instruction_id'] = first_instruction.fields['trade_instruction_id']
                                order.update()
                                
                    if first_instruction is not None:
                        first_instruction.update()

                    skip_symbols.append(instruction.fields["symbol"])
     
                if abs(open_pos_qty) > 0:
                    unattached_pos = 0
                    unattached_orders = Order.get(context,symbol=instruction.fields["symbol"],trade_instruction_id=None)

                    for u_order in unattached_orders:
                        if u_order.fields['filled_qty'] is not None and \
                            u_order.fields['filled_qty'] != 0:
                            unattached_pos = unattached_pos + (u_order.fields['filled_qty'] \
                                                               if u_order.fields['side'] == "B" else \
                                                               -u_order.fields['filled_qty'])
                    if unattached_pos == -open_pos_qty:
                        logger.error("Missing TI position found in unattached orders")
                        for symbol_instruction in symbol_instructions:
                            if symbol_instruction.get_open_position_quantity() != 0:
                                attach_all_orders = symbol_instruction.get_open_position_quantity() == -unattached_pos
                                
                                rev_ti_side = "B" if symbol_instruction.fields['side'] == "S" else "S"
                                open_order_qty = symbol_instruction.fields['quantity']
                                for u_order in unattached_orders:
                                    if attach_all_orders or \
                                       u_order.fields['filled_qty'] > 0 and \
                                       u_order.fields['state'] == Order.S_FILLED and \
                                       u_order.fields['quantity'] == abs(open_order_qty) and \
                                       u_order.fields['side'] == rev_ti_side:
                                        u_order.fields['trade_instruction_id'] = symbol_instruction.fields['trade_instruction_id']
                                        u_order.update()
                                        unattached_orders.remove(u_order)
                    else:
                        logger.error("Position missing on broker for symbol {}, qty {}"\
                                     .format(instruction.fields["symbol"],
                                        open_pos_qty) )
                        return_result = False
    
    return return_result


def prices_up_to_date( context ):
    
    logger = context[PARAM_LOGGER]
    
    return_result = True
    
    tickers_for_analysis = context['active_stocks']
    
    # get trading dates
    end_dt = datetime.combine( datetime.now().date(), time.max )
    start_dt = datetime.combine( datetime.now() - timedelta(days=context[PARAM_PRICE_RECON_LOOKBACK]),
                                 time.min) 
    
    trade_cal_df = MarketData.get_trading_calendar(context, start_dt, end_dt)

    trading_dt_set = set(trade_cal_df['date'])

    # check that the prices the trading dates listed above
    for ticker in tickers_for_analysis:
        try:
            stock_prices = MarketData.prices_sql( context, ticker, start_dt, end_dt )[["close"]]
            sp_set = set([ts.to_pydatetime().date() for ts in stock_prices.index])
        
            sp_extra = sp_set - trading_dt_set
            sp_missing = trading_dt_set - sp_set
            
            if len(sp_extra) > 0 or len(sp_missing) > 0:
                logger.error("Price date reconciliation failed for {0}. Extra entries: {1}. Missing Entries {2}."\
                                        .format(ticker, sp_extra, sp_missing))
                return_result = False
        except:
            logger.error("Error reconciling prices for {0}".ticker)
            logger.error(traceback.format_exc())
            return_result = False
    
    return return_result


def perform_startup_validations( context, validation_mode = False ):
    """
    Checks to ensure data consistency and conditions are suitable for processing
    """
    
    logger = context[PARAM_LOGGER]
    
    return_result = True

    logger.info("Performing startup checks...")
    broker = context[PARAM_BROKER_INTERFACE]
    
    if broker.is_market_open():
        logger.info("Market is open")
    elif not validation_mode: 
        test_df = MarketData.get_trading_calendar(context,datetime.now()-timedelta(days=10),datetime.now())
        if len(test_df) > 0 and datetime.now().date() not in list(test_df['date']):
            logger.info("Today is a weekend or holiday. Exiting...")
            sys.exit(0)
        elif len(test_df) == 0:
            raise ValidationError("No trading days returned by calendar")
        
        if datetime.now() > test_df.set_index('date').loc[datetime.now().date()].close:
            msg = "Market is closed"
            return_result = False
            logger.fatal(msg)
        else:
            logger.info("Running pre-open")
    
    # - check that account has sufficient funds for start-up
    account_equity = broker.get_account_balance()
    if account_equity > context[PARAM_MIN_EQUITY_REQUIREMENT]:
        logger.info("Account equity is sufficient: {0}".format(account_equity))
    else:
        msg = "Account equity is insufficient at {0}".format(account_equity)
        logger.warning(msg)
    
    # - check that we have sufficient time before market closet (configured)
    trading_minutes_req = context[PARAM_MIN_TRADING_DAY_MINUTES]
    (start,end) = broker.get_market_hours(datetime.now())
    
    if not validation_mode:
        if end is not None:
            time_to_close = (end - datetime.now())
            if time_to_close > timedelta(minutes=trading_minutes_req):
                logger.info("Sufficient time until market close: {0} minutes".format(time_to_close.seconds // 60))
            else:
                msg = "Insufficient time until market close: {0} minutes".format(time_to_close.seconds // 60)
                logger.fatal(msg)
                return_result = False
        else:
            raise ValidationError("No close time found for market today")
    
    # - compare state for all orders today
    if not check_order_db_integrity( context ):
        return_result = False
        
    # - check trade instruction integrity
    if not check_trade_instruction_integrity( context ):
        return_result = False
    
    # check that net positions on the broker match what we are expecting
    if not check_broker_positions( context ):
        return_result = False
        
    if context.get('trading.circuit_breaker.disable_winrate',False):
        logger.warning("Winrate circuit breaker has been disabled through config option trading.circuit_breaker.disable_winrate")
    
    return return_result
    

def perform_shutdown_validations( context ):

    logger = context[PARAM_LOGGER]

    return_result = True

    logger.info("Performing shutdown checks...")

    # - compare state for all orders today
    if not check_order_db_integrity( context ):
        return_result = False
        
    # - check trade instruction integrity
    if not check_trade_instruction_integrity( context ):
        return_result = False
    
    # check that net positions on the broker match what we are expecting
    if not check_broker_positions( context ):
        return_result = False

    return return_result 
    
    
    
    
    
    
    
    
    