import pprint, traceback, logging
from datetime import datetime, timedelta, time
import time as ptime

import numpy as np
import pandas as pd

from algotrading.trading.Objects import TradeInstruction, Order
from algotrading.trading.Common import check_for_market_events, check_for_dividend, check_bid_ask_spread, \
                                        get_execution_datetime
from algotrading.trading.OrderManagement import create_order, check_for_sufficient_exposure
from algotrading.trading.Pricing import get_symbol_mark

from algotrading.ContextFields import *


def init_static_trade_quantities( context ):
    static_trade_quantities = {}
    
    for symbol in context[PARAM_ANALYSIS_STOCK_TICKERS]: 
        mark = get_symbol_mark(context,symbol)
        static_trade_quantities[symbol] = int(context[PARAM_SINGLE_TRADE_EXPOSURE]/mark)
        
        if context.get('trading.even_lots_only',False):
            static_trade_quantities[symbol] = int(np.round(static_trade_quantities[symbol] / 100) * 100)
            if static_trade_quantities[symbol] == 0:
                static_trade_quantities[symbol] = 100
                
            if static_trade_quantities[symbol] * mark > context[PARAM_MAX_POSITION_EXPOSURE]:
                static_trade_quantities[symbol] -= 100
                
            if static_trade_quantities[symbol] * mark > context[PARAM_MAX_POSITION_EXPOSURE]:
                context[PARAM_LOGGER].warning(f"Symbol {symbol}'s nearest even lot of {static_trade_quantities[symbol]} has "\
                                              f"exposure of {np.round(static_trade_quantities[symbol] * mark,2)} which exceeds"\
                                              f"max exposure of {context[PARAM_MAX_POSITION_EXPOSURE]}. Trades on {symbol} will be rejected.")
                static_trade_quantities[symbol] = 0
                
    context['trading.static_trade_quantities'] = static_trade_quantities
    
    return context
    

def optimize_trade_instruction_allocations( context, instructions, target_portfolio_value, max_pos_exposure ):

    quantities = []
    calc_df = pd.DataFrame(columns=["qty","pos_error","marginal_value"])
    current_portfolio_value = 0
    
    per_instruction_target_value = target_portfolio_value / len(instructions)
    if per_instruction_target_value >= max_pos_exposure:
        per_instruction_target_value = max_pos_exposure
        target_portfolio_value = max_pos_exposure * len(instructions)
        initial_allocations_ok = True
    else:
        initial_allocations_ok = False
        
    for instruction in instructions:
        # get qty as even lot size that is closest to the target average position value
        price = get_symbol_mark( context, instruction.fields["symbol"] )
        actual_qty = int(per_instruction_target_value / price)
        if actual_qty % 2 == 1:
            adjusted_qty = int(actual_qty) + 1
            if (adjusted_qty*price) > max_pos_exposure:
                adjusted_qty = int(actual_qty) - 1
        else:
            adjusted_qty = int(actual_qty)
        
        # compute marginal value of +/- 2 shares and current error
        marginal_value = 2 * price
        pos_error = (adjusted_qty - actual_qty) * price
        
        # store in a DF
        calc_df = calc_df.append({"qty" : adjusted_qty, "price" : price, "pos_error" : pos_error, "marginal_value" : marginal_value}, ignore_index=True)
        current_portfolio_value = current_portfolio_value + adjusted_qty * price
     
    # get working DF
    ref_calc_df = calc_df
    if current_portfolio_value > target_portfolio_value:
        calc_df = calc_df[calc_df["pos_error"] >= 0].copy()
        calc_df["marginal_value"] = -1 * calc_df["marginal_value"]      # we are selling down
    else:
        calc_df = calc_df[calc_df["pos_error"] <= 0].copy()
        
    if len(calc_df) == 0 or initial_allocations_ok:
        return np.array(ref_calc_df["qty"].astype("int64"))

    # iterate and keep adjusted the quantity of the stock with the smallest single increment error     
    starting_portfolio_value = current_portfolio_value
    while np.sign(starting_portfolio_value-target_portfolio_value) == np.sign(current_portfolio_value-target_portfolio_value):
        calc_df["single_inc_error"] = calc_df["pos_error"] + calc_df["marginal_value"]
        calc_df = calc_df.sort_values(["single_inc_error"], ascending=target_portfolio_value>current_portfolio_value)

        new_qty = calc_df.iloc[0]["qty"] + np.copysign(2, calc_df.iloc[0]["marginal_value"])
        if abs(new_qty*calc_df.iloc[0]['price']) > max_pos_exposure:
            break
        
        calc_df.loc[calc_df.index[0],"qty"] = new_qty
        calc_df.loc[calc_df.index[0],"pos_error"] = calc_df.iloc[0]["pos_error"] + calc_df.iloc[0]["marginal_value"]
        
        current_portfolio_value = current_portfolio_value + calc_df.iloc[0]["marginal_value"]

    calc_df = calc_df.sort_index()
    ref_calc_df.update(calc_df)

    return np.array(ref_calc_df["qty"].astype("int64"))


def cancel_trade_instruction( context, ti_id ):
    
    broker = context[PARAM_BROKER_INTERFACE]
    
    def cancel_all_orders( ti ):
        orders = Order.get(context,trade_instruction_id=ti)
        for order in orders:
            if order.fields['state'] != Order.S_PENDING_CANCEL and \
                order.fields['state'] in Order.non_terminal_states:
                broker.cancel_order(order.fields['broker_order_id'])
                order.fields['state'] = Order.S_PENDING_CANCEL
                order.update()
                
    tis = TradeInstruction.get(context, trade_instruction_id=ti_id)
    if len(tis) > 0:
        for ti in tis:
            close_trade(context,ti)
            """
            if ti.fields['state'] in [TradeInstruction.S_CREATED,TradeInstruction.S_OPEN]:
                ti.fields['state'] = TradeInstruction.S_CANCELLED
                ti.update()
            elif ti.fields['state'] == TradeInstruction.S_PENDING_EXEC:
                cancel_all_orders(ti.fields['trade_instruction_id'])
            elif ti.fields['state'] in TradeInstruction.non_terminal_states:
                orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'])
                if len(orders) == 1 and orders[0].fields['order_type'] != Order.TYPE_MARKET or \
                    len(orders) > 1:
                    cancel_all_orders(ti.fields['trade_instruction_id'])
                    
            ti.fields['close_out_date'] = datetime.now()
            ti.update()
            """
    
    
def create_trade_instructions_from_results( context ):
    
    logger = context[PARAM_LOGGER]
    
    static_quantities_only = context.get('trading.enforce_static_trade_quantity',False)
    
    results = context[PARAM_ALGO_RESULTS]
    
    if len(results) == 0:
        return True
    
    def calculate_exposure( side, max_gross_exposure = context[PARAM_MAX_GROSS_EXPOSURE] ):
        cur_dt = datetime.now()
        instructions = TradeInstruction.get(context,
                                         creation_date=(datetime.combine(cur_dt, time.min),cur_dt),
                                         side=["B","S"] if side is None else side )
        
        agg_symbol_instructions = context.get('trading.agg_symbol_instructions',False)
        
        # compute today's active exposure
        todays_ti_exposure = 0
        symbol_exposure = {}
        adjustable_instructions = []
        for instruction in instructions:
            if side is None or instruction.fields['side'] == side:
                if instruction.fields["state"] == TradeInstruction.S_OPEN:
                    adjustable_instructions.append(instruction)
                elif instruction.fields['state'] in TradeInstruction.non_terminal_states:
                    if (instruction.fields["close_out_date"]-timedelta(minutes=context.get(PARAM_PRE_CLOSE_INSTRUCTION_CLOSEOUT_TIMING,3)) > datetime.now()) or \
                            context.get('trading.continuous.enable',True):
                        if agg_symbol_instructions: # only count the largest size trade in trade exposure
                            if instruction.fields['symbol'] in symbol_exposure and \
                                symbol_exposure[instruction.fields['symbol']] < instruction.fields['expected_notional']:
                                todays_ti_exposure = todays_ti_exposure - symbol_exposure[instruction.fields['symbol']]
                                del symbol_exposure[instruction.fields['symbol']]
                                
                        if not agg_symbol_instructions or instruction.fields['symbol'] not in symbol_exposure:
                            todays_ti_exposure = todays_ti_exposure + instruction.fields["expected_notional"]
                            symbol_exposure[instruction.fields['symbol']] = instruction.fields["expected_notional"]
    
        # how much exposure is remaining?
        remaining_exposure = max_gross_exposure - todays_ti_exposure
        
        return (remaining_exposure,todays_ti_exposure,instructions,adjustable_instructions)
        
    if context[PARAM_ALGO_TRADE_SIDE] in ["longshort","neutral"]:
        (remaining_exposureS,todays_ti_exposureS,instructionsS,adjustable_instructionsS) = calculate_exposure("S")
        (remaining_exposureB,todays_ti_exposureB,instructionsB,adjustable_instructionsB) = calculate_exposure("B")
        remaining_exposure = context[PARAM_MAX_GROSS_EXPOSURE] - todays_ti_exposureS - todays_ti_exposureB
        adjustable_instructions = adjustable_instructionsB + adjustable_instructionsS
        instructions = instructionsB + instructionsS
        todays_ti_exposure = todays_ti_exposureB - todays_ti_exposureS
    elif context[PARAM_ALGO_TRADE_SIDE] == "long":
        (remaining_exposure,todays_ti_exposure,instructions,adjustable_instructions) = calculate_exposure("B")
    elif context[PARAM_ALGO_TRADE_SIDE] == "short":
        (remaining_exposure,todays_ti_exposure,instructions,adjustable_instructions) = calculate_exposure("S")
    else:
        logger.error("Unknown trade side {} specified".format(context[PARAM_ALGO_TRADE_SIDE]))
    
    min_free_exposure = context.get("trading.min_free_exposure_for_ti_creation",0.05)
    
    exposure_ratio = remaining_exposure / context[PARAM_MAX_GROSS_EXPOSURE]
    if context[PARAM_ALGO_TRADE_SIDE] in ['long','short'] and \
        abs(exposure_ratio) < min_free_exposure:
        logger.debug(f"Insufficient remaining daily exposure: {exposure_ratio}. New instructions will not be created.")
        return False

    elif context[PARAM_ALGO_TRADE_SIDE] in ['neutral','longshort']:
        if abs(exposure_ratio) < min_free_exposure:
            logger.debug(f"Insufficient remaining gross exposure: {exposure_ratio}. New instructions will not be created.")
            return False
        
        if abs(todays_ti_exposure) > context[PARAM_MAX_NET_EXPOSURE]:
            logger.debug(f"Insufficient remaining net exposure: {todays_ti_exposure}. New instructions will not be created.")
            return False
            
    results = results.set_index('Date')
    
    logger.info("Algo results are: {0}".format(results))
    
    instruct_col = context[PARAM_ALGO_TRADE_INSTRUCTION_COLUMN]
    trade_sides = context[PARAM_ALGO_TRADE_SIDE]
    invert_trades = context.get(PARAM_ALGO_INVERT_INSTRUCTIONS,False)
    holding_period = context[PARAM_ALGO_TRADE_HOLDING_PERIOD]
    default_trade_value = context.get(PARAM_SINGLE_TRADE_EXPOSURE,context[PARAM_MAX_POSITION_EXPOSURE])
    
    # calculate future closeout date if needed
    closeout_date = None
    if PARAM_ALGO_TRADE_HOLDING_PERIOD in context.keys():
        target_close_dt = get_execution_datetime( context, field=PARAM_TARGET_CLOSE_POSITION_TIME )
        closeout_date = target_close_dt - timedelta(seconds=context.get(PARAM_PRE_CLOSE_CLOSE_ORDER_TIMING,30))

    # check that we have predictions for everything
    results_tickers = set(results[PARAM_ANALYSIS_STOCK_TICKERS])
    expected_tickers = set(context['active_stocks'])
    missing = list(expected_tickers - results_tickers)
    
    # can skip the check if we loaded data from disk)
    if len(missing) > 0 and not context.get('trading.continuous.enable',False) and \
                (PARAM_LOAD_DATA not in context.keys() or \
                   context[PARAM_LOAD_DATA] is None):
        logger.error("Analysis results missing the following tickers: {0}".format(missing))
        return True
    
    if datetime.now() > \
            (context[PARAM_BROKER_INTERFACE].get_todays_close()-timedelta(seconds=context.get("trading.pre_close_trade_cutoff",0))):
        logger.info("Last trade cutoff has passed. No new trade instructions will be created.")
        return True
    
    if "trading.test.max_instructions" in context:
        max_instructions = context["trading.test.max_instructions"]
    else:
        max_instructions = len(results)
        
    buy_created_instructions = 0
    sell_created_instructions = 0

    results = results.sort_values(PARAM_ANALYSIS_STOCK_TICKERS)

    working_trades = TradeInstruction.get(context,
                                            creation_date=(datetime.combine(datetime.now().date(),time.min),
                                                           datetime.now()),
                                            algo_id=context[PARAM_APP_NAME])
    
    all_trades_by_symbol = {}
    working_trades_by_symbol = {}
    for trade in working_trades:
        if trade.fields['symbol'] not in all_trades_by_symbol:
            all_trades_by_symbol[trade.fields['symbol']] = []
            working_trades_by_symbol[trade.fields['symbol']] = []
        
        if trade.fields['state'] in TradeInstruction.non_terminal_states:    
            working_trades_by_symbol[trade.fields['symbol']].append(trade)
        all_trades_by_symbol[trade.fields['symbol']].append(trade)

    for i in range(len(results)):
            
        row = results.iloc[i]
        
        algo_trade_flag = row[instruct_col]
        ticker = row[PARAM_ANALYSIS_STOCK_TICKERS]
        
        if ticker in context.get("trading.symbol_blacklist",[]):
            logger.info("Symbol {} in trading.symbol_blacklist. Trade instruction not generated.".format(ticker))
            max_instructions = max_instructions - 1
            continue

        if context[PARAM_ALGO_TRADE_SIDE] == "neutral" and \
            buy_created_instructions == max_instructions and \
            sell_created_instructions == max_instructions or \
            context[PARAM_ALGO_TRADE_SIDE] == "long" and \
            buy_created_instructions == max_instructions or \
            context[PARAM_ALGO_TRADE_SIDE] == "short" and \
            sell_created_instructions == max_instructions:
            break

        if row.name.date() != datetime.now().date():
            logger.error("Prediction on {0} for trade date {1} instead of {2}.".format(ticker,row.name,datetime.now().date()))
            continue

        try:
            start_dt = datetime.now()
            if closeout_date is not None:
                end_dt = closeout_date
            else:
                end_dt = start_dt
                
            if context.get("trading.filter_market_events",False) and \
                check_for_market_events( context, ticker, start_dt, end_dt ):
                logger.info( "Market event for ticker {0} near {1}. Skipping trade instruction creation." \
                                    .format(ticker,closeout_date.strftime("%Y-%m-%d")))
                continue
        
            tradeable = False
            if not np.isnan(algo_trade_flag):
                if trade_sides.upper() == 'LONG':
                    tradeable = bool(algo_trade_flag)
                    side = "B"
                elif trade_sides.upper() == 'SHORT':
                    tradeable = not bool(algo_trade_flag)
                    side = "S"
                elif trade_sides.upper() in ['NEUTRAL','LONGSHORT']:
                    tradeable = True
                    side = 'B' if bool(algo_trade_flag) == True else 'S'
                    
                if invert_trades:
                    side = 'S' if side == 'B' else 'B'
        
            if context.get('trading.delta_trade_mode',False) and \
                ticker in working_trades_by_symbol and \
                len(working_trades_by_symbol[ticker]) > 0:
                
                if len(working_trades_by_symbol[ticker]) > 1:
                    logger.info("More than 1 active trade instruction on symbol {}".format(ticker))
                    continue
                
                working_trade = working_trades_by_symbol[ticker][0]
                
                if side == working_trade.fields['side']:
                    continue
                
                if np.isnan(algo_trade_flag) or side != working_trade.fields['side']:
                    for trade in working_trades_by_symbol[ticker]:
                        trade.fields['close_out_date'] = datetime.now()
                        trade.update()
        
            if np.isnan(algo_trade_flag):
                continue 
        
            if not check_bid_ask_spread(context, ticker):
                logger.info("Spreads on symbol {} are too wide. Trade instruction will not be created.")
                continue
            
            if context.get('trading.continuous.enable',False) and \
                ticker in all_trades_by_symbol:
                
                window_len = context.get('trading.continuous.min_time_between_trades',120)
                window_start_dt = datetime.now() - timedelta(seconds=window_len)
                
                num_trades = 0
                for trade in all_trades_by_symbol[ticker]:
                    if trade.fields['creation_date'] is not None and \
                         trade.fields['creation_date'] > window_start_dt:
                        num_trades = num_trades + 1

                if num_trades > 2:                        
                    logger.info("More than 2 trades created in the last {} seconds. Skipping new trade instruction".format(window_len))
                    continue
            
            if tradeable and side == 'S' and \
                context.get("trading.filter_short_dividends",True) and \
                check_for_dividend(context, ticker, start_dt, end_dt):
                logger.info("Dividend ex-date found for {} over trade lifetime. Skipping trade instruction creation." \
                                    .format(ticker))
                continue
            
            if tradeable or context.get("trading.test.live_test",False):
                current_price = get_symbol_mark( context, ticker )
                if current_price > 0:
                    end_dt = datetime.now()
                    start_dt = datetime.combine(datetime.now(), time.min)

                    if context.get('trading.continuous.enable',False):
                        dup_instructions = []
                    elif context.get('trading.allow_repeat_symbol_instructions',False):
                        dup_instructions = working_trades_by_symbol.get(ticker,[])
                    else:
                        dup_instructions = all_trades_by_symbol.get(ticker,[])
                    
                    min_invest = 2*current_price if context.get('trading.force_even_quantities',False) else current_price
                    if min_invest < context.get(PARAM_SINGLE_TRADE_EXPOSURE,context[PARAM_MAX_POSITION_EXPOSURE]):
                        if side == 'S':
                            if sell_created_instructions == max_instructions:
                                continue
                            else:
                                sell_created_instructions = sell_created_instructions + 1
                        else:
                            if buy_created_instructions == max_instructions:
                                continue
                            else:
                                buy_created_instructions = buy_created_instructions + 1

                        if len(dup_instructions) == 0:
                            # check that if we are using static quantities that it's non-zero
                            if static_quantities_only and \
                                not context.get("trading.test.live_test",False) and \
                                context['trading.static_trade_quantities'][row[PARAM_ANALYSIS_STOCK_TICKERS]] == 0:
                                continue
                            
                            # do net exposure check
                            if static_quantities_only:
                                qty = context['trading.static_trade_quantities'][row[PARAM_ANALYSIS_STOCK_TICKERS]]
                            else:
                                qty = default_trade_value // float(current_price) 

                            # check if max loss is violated by current trade instruction parameters
                            max_loss = context.get('trading.pnl.max_loss',None)
                            if max_loss is not None and \
                                'TradeInstruction.stop_loss' in row:
                                cur_stop_loss = row['TradeInstruction.stop_loss']
                                loss_pct = abs((current_price - cur_stop_loss)/current_price) if cur_stop_loss > 1 else cur_stop_loss
                                if loss_pct > max_loss:
                                    qty *= max_loss / loss_pct
                                    qty = int(qty)
                                    if context.get('trading.even_lots_only',False):
                                        qty = np.floor(qty/100) * 100
                                        
                                    if qty == 0:
                                        logger.info(f"Unable to create trade instruction:\n{row}\nQuantity rounds down to 0 when applying trading.pnl.max_loss")
                                        continue

                            notional = round(qty * float(current_price),2)
                            
                            if not check_for_sufficient_exposure( context, notional if side == 'B' else -notional ):
                                logger.info(f"Insufficient net exposure for notional {notional} to create trade instruction.")
                                continue
                                    
                            # create trade instructions
                            instruction = TradeInstruction.new( context )
                            instruction.fields['state'] = TradeInstruction.S_OPEN
                            instruction.fields['side'] = side
                            instruction.fields['symbol'] = row[PARAM_ANALYSIS_STOCK_TICKERS]
                            
                            instruction.fields['quantity'] = int(qty)
                            instruction.fields['expected_notional'] = notional
                            
                            if PARAM_TARGET_CLOSE_POSITION_TIME in row.index:
                                closeout_date = datetime.combine( closeout_date.date() if closeout_date is not None else datetime.now().date(),
                                                                    datetime.strptime(row[PARAM_TARGET_CLOSE_POSITION_TIME],"%H:%S").time() )

                            if closeout_date is not None:
                                instruction.fields['close_out_date'] = closeout_date
                                
                            for data_field in row.index:
                                if data_field.startswith("TradeInstruction."):
                                    ti_field = data_field.replace("TradeInstruction.","")
                                    
                                    if type(row[data_field]) is not float or \
                                        not np.isnan(row[data_field]):
                                        instruction.fields[ti_field] = row[data_field]

                            close_dt = get_execution_datetime(context,field=PARAM_TARGET_CLOSE_POSITION_TIME)
                            if instruction.fields['close_out_date'] is not None and \
                                instruction.fields['close_out_date'] > close_dt:
                                instruction.fields['close_out_date'] = close_dt
                    
                            if context.get("trading.test.live_test",False):
                                instruction.fields['side'] = context.get("trading.test.live_test_side","B")
                                instruction.fields['symbol'] = context["trading.test.test_symbol"]
                                
                                instruction.fields['quantity'] = 1
                                instruction.fields['expected_notional'] = round(instruction.fields['quantity'] * float(current_price),2)
                                
                                instruction.fields['close_out_date'] = datetime.now() + timedelta(minutes=2)
                                
                                instruction.update()
                                return True
                                
                            if instruction.fields['quantity'] == 0:
                                instruction.fields['state'] = TradeInstruction.S_CANCELLED
                            # update in the database
                            instruction.update()
                            
                            logger.info("Created trade instruction: {0}".format(instruction.pprint()))
                            
                            if instruction.fields['quantity'] == 0:
                                logger.warn( "Trade instruction with zero quantity: {0}. Price: {1}. Desired Trade Value: {2}"\
                                             .format(instruction.fields["trade_instruction_id"], current_price, default_trade_value))
                        else:
                            logger.warn( "Trade instruction already exists today for symbol {0}.".format(ticker) )
                    else:
                        logger.warn( "Unable to create trade instruction for symbol {0} as qty would be 0.".format(ticker) )                        
                else:
                    logger.error("Trade instruction not generated for ticker {0}. Price is {0}".format(ticker,current_price))
        except:
            logger.error("Unable to create trade instruction for ticker {0}.".format(ticker)) 
            logger.error(traceback.format_exc())            

    # resize losses if configured. supersedes static quantity configuration option.
    if context.get('trading.pnl.standardize_pnl',False):
        if context['trading.pnl.max_loss'] < 0.0001 or \
            context['trading.pnl.max_loss'] > 0.05:
            raise ValueError(f"Invalid value for {context['trading.pnl.max_loss']}")
        open_tis = TradeInstruction.get(context,state=TradeInstruction.S_OPEN)
        for open_ti in open_tis:
            symbol_price = get_symbol_mark(context,open_ti.fields['symbol'])
            
            if open_ti.fields['stop_loss'] is not None and \
                open_ti.fields['stop_gain'] is not None:
                    
                if open_ti.fields['stop_loss'] > 1:     #its price
                    adj_stop_loss = abs(open_ti.fields['stop_loss'] - symbol_price) / symbol_price 
                else:
                    adj_stop_loss = open_ti.fields['stop_loss']
                
                if context.get('trading.pnl.standardize_pnl',False):
                    adj_ratio = context['trading.pnl.max_loss'] / adj_stop_loss
                    
                    max_adj_ratio = context.get('trading.max_trade_loss_adj_ratio',10)
                    if adj_ratio > max_adj_ratio and not context.get('trading.pnl.standardize_pnl',False):
                        open_ti.fields['state'] = TradeInstruction.S_ABORTED
                        logger.warning(f"Trade loss adjustment ratio {adj_ratio} is greater than {max_adj_ratio} for ti {open_ti.fields['trade_instruction_id']}. TI aborted.")
                    else:
                        new_qty = int(open_ti.fields['quantity'] * adj_ratio)
                        
                        if new_qty*symbol_price > context[PARAM_MAX_POSITION_EXPOSURE]:
                            orig_new_qty = new_qty
                            new_qty = int(context[PARAM_MAX_POSITION_EXPOSURE] / symbol_price)
                            logger.info(f"Required quantity for max_loss {orig_new_qty} exceeds maximum exposure threshold of {context[PARAM_MAX_POSITION_EXPOSURE]}. Quantity adjusted to {new_qty}.")
                        
                        if context.get('trading.even_lots_only',False):
                            new_qty = int(100*np.floor(new_qty//100))  # keep quantity even
                            
                        if new_qty == 0:
                            open_ti.fields['state'] = TradeInstruction.S_ABORTED
                            logger.info(f"Adjusted trade instruction {open_ti.fields['trade_instruction_id']} had quantity reduced to 0 for max trade loss {context['trading.pnl.max_loss']}. TI aborted.")
                        else:
                            logger.info(f"Adjusted quantity for trade instruction {open_ti.fields['trade_instruction_id']} to ensure max loss is {context['trading.pnl.max_loss']*context[PARAM_SINGLE_TRADE_EXPOSURE]}")
                            open_ti.fields['quantity'] = new_qty
                            open_ti.fields['expected_notional'] = round(new_qty*symbol_price,2) 
                    
                open_ti.update()

    # now adjust allocations per trade instruction so that they fit our daily exposure requirements
    def enforce_target_exposure_restrictions( side, target_daily_exposure = context[PARAM_MAX_GROSS_EXPOSURE],
                                                    max_pos_exposure = context.get(PARAM_SINGLE_TRADE_EXPOSURE,context[PARAM_MAX_POSITION_EXPOSURE]) ):
        
        (remaining_exposure,
         todays_ti_exposure,
         instructions,
         adjustable_instructions) = calculate_exposure( side, target_daily_exposure )

        agg_symbol_instructions = context.get('trading.agg_symbol_instructions',False)
    
        # do we have instructions left to adjust
        new_exposure = 0
        if len(adjustable_instructions) > 0:
            if remaining_exposure > 0 and \
                not static_quantities_only:
                allocs = optimize_trade_instruction_allocations( context, adjustable_instructions, remaining_exposure, max_pos_exposure )

                if agg_symbol_instructions:
                    adj_instruct_by_symbol = {}
                    for instruction in adjustable_instructions:
                        if instruction.fields['symbol'] not in adj_instruct_by_symbol:
                            adj_instruct_by_symbol[instruction.fields['symbol']] = [instruction]
                        else:
                            adj_instruct_by_symbol[instruction.fields['symbol']].append(instruction)
                            
                    for symbol, instructions in adj_instruct_by_symbol.items():
                        if len(instructions) > 1:
                            total_allocs = 0
                            for instruction in instructions:
                                total_allocs = total_allocs + allocs[adjustable_instructions.index(instruction)]
                            
                            for instruction in instructions:
                                allocs[adjustable_instructions.index(instruction)] = total_allocs
                            
                symbol_exposure = {}
                max_trade_exposure = context.get(PARAM_SINGLE_TRADE_EXPOSURE,context[PARAM_MAX_POSITION_EXPOSURE])
                
                for i in range(len(adjustable_instructions)):
                    instruction = adjustable_instructions[i]
                    price = get_symbol_mark( context, instruction.fields["symbol"] )
                    instruction.fields['quantity'] = allocs[i]
                    instruction.fields['expected_notional'] = round(instruction.fields['quantity']  * price,2)
                    if instruction.fields['expected_notional'] > max_trade_exposure:
                        instruction.fields['expected_notional'] = max_trade_exposure      # price may have moved since allocations were calculated
                    
                    instruction.update()
                    
                    if not agg_symbol_instructions or instruction.fields["symbol"] not in symbol_exposure:
                        new_exposure = new_exposure + instruction.fields['expected_notional']
                        symbol_exposure[instruction.fields["symbol"]] = instruction.fields['expected_notional']
            elif remaining_exposure < 0:
                logger.error("Remaining exposure {0} is negative with instructions just created. All new instructions will be cancelled.".format(remaining_exposure))
                for instruction in adjustable_instructions:
                    instruction.fields['quantity'] = 0
                    instruction.fields['expected_notional'] = 0
                    instruction.fields['state'] = TradeInstruction.S_CANCELLED
                    instruction.update()
            elif static_quantities_only:
                for ti in adjustable_instructions:
                    new_exposure = new_exposure + ti.fields['expected_notional']

        # return the expected total exposure after trading                    
        return todays_ti_exposure + new_exposure

    # make sure long and shorts are roughly in balance
    if context[PARAM_ALGO_TRADE_SIDE] == "longshort":
        enforce_target_exposure_restrictions(None)
    elif context[PARAM_ALGO_TRADE_SIDE] == "neutral":
        if context["trading.neutral_long_short_ratio"] is not None:
            buy_exposure = float(context["trading.neutral_long_short_ratio"]) * context[PARAM_MAX_GROSS_EXPOSURE]
            sell_exposure = (1-float(context["trading.neutral_long_short_ratio"])) * context[PARAM_MAX_GROSS_EXPOSURE]
            
            enforce_target_exposure_restrictions("B", target_daily_exposure=buy_exposure)
            enforce_target_exposure_restrictions("S", target_daily_exposure=sell_exposure)
            
        else:
            total_sell_exposure = enforce_target_exposure_restrictions("S")
            total_buy_exposure = enforce_target_exposure_restrictions("B")

            tolerance = context["trading.exposure_tolerance"]

            if total_sell_exposure < (total_buy_exposure*(1-tolerance)):
                enforce_target_exposure_restrictions("B",total_sell_exposure)
            elif total_sell_exposure > (total_buy_exposure*(1+tolerance)):
                enforce_target_exposure_restrictions("S",total_buy_exposure)
                
        if total_sell_exposure == 0 or total_buy_exposure == 0:
            logger.warning("Total sell or buy exposure is 0 (one-sided trades today). This will result in no trades being placed.")
    elif context[PARAM_ALGO_TRADE_SIDE] == "long":
        enforce_target_exposure_restrictions("B")
    elif context[PARAM_ALGO_TRADE_SIDE] == "short":
        enforce_target_exposure_restrictions("S")
    else:
        logger.error("Unknown trade side {} specified".format(context[PARAM_ALGO_TRADE_SIDE]))
        
    return True
    
    
def close_trade( context, trade ):
    logger = context[PARAM_LOGGER]
    broker = context[PARAM_BROKER_INTERFACE]
    
    if trade.fields['state'] in TradeInstruction.terminal_states:
        return
    
    if trade.fields["state"] in [TradeInstruction.S_CREATED,TradeInstruction.S_OPEN]:
        logger.info("Trade Instruction {} in CREATED/OPEN, cancelling.".format(trade.fields["trade_instruction_id"]))
        trade.fields["state"] = TradeInstruction.S_CANCELLED
    else:
        logger.info("Trade Instruction {}, cancelling active orders.".format(trade.fields["trade_instruction_id"]))
        orders = Order.get(context, trade_instruction_id=trade.fields["trade_instruction_id"])
        
        for order in orders:
            try:
                if order.fields["state"] in Order.non_terminal_states and \
                    order.fields['state'] != Order.S_PENDING_CANCEL and \
                    order.fields["broker_order_id"] is not None:
                    broker.cancel_order( order.fields["broker_order_id"] )
                    order.fields['state'] = Order.S_PENDING_CANCEL
                    order.update()
            except:
                logger.warning("Unable to cancel broker order {}".format(order.fields["broker_order_id"]) )
                logger.warning(traceback.format_exc())

        if trade.get_open_position_quantity() == 0:
            if trade.fields['open_order_id'] is not None and \
                trade.fields['close_order_id'] is not None:
                trade.fields["state"] = TradeInstruction.S_CLOSED
                logger.info("Trade Instruction {}, state set to closed.".format(trade.fields["trade_instruction_id"]))
            else:
                trade.fields["state"] = TradeInstruction.S_CANCELLED
                logger.info("Trade Instruction {}, state set to cancelled.".format(trade.fields["trade_instruction_id"]))
            
        else:
            logger.info("Trade Instruction {}, setting close out date to now.".format(trade.fields["trade_instruction_id"]))
            trade.fields["close_out_date"] = datetime.now()
            trade.fields["close_order_id"] = None
    
    trade.update_state()
    trade.update()


def check_trade_instructions( context ):
    
    # ensure that any TIs with a flat position in CLOSE_PENDING are moved to CLOSED
    # this can happen if a linked OCO trade does not auto-cancel when the other side executes 
    active_tis = TradeInstruction.get(context,state=TradeInstruction.S_CLOSE_PENDING)
    for active_ti in active_tis:
        if active_ti.get_open_position_quantity() == 0:
            context[PARAM_LOGGER].info(f"Trade Instruction {active_ti.fields['trade_instruction_id']} in CLOSE_PENDING but has a flat position. Closing trade.")
            close_trade( context, active_ti )

    

ignored_tis_for_retry_msg = []
    
def execute_trade_instructions( context ):
    global ignored_tis_for_retry_msg
    
    logger = context[PARAM_LOGGER]
    
    # resubmits trades that were created today that didnt fully execute
    cur_dt = datetime.now()
    retry_trades = []
    target_open_pos_dt = get_execution_datetime(context, field=PARAM_TARGET_OPEN_POSITION_TIME) + \
                        timedelta(seconds=context.get('trading.open_position_time_window',45))

    execute_after = get_execution_datetime(context) - \
                                            timedelta(seconds=context.get(PARAM_PRE_OPEN_OPEN_ORDER_TIMING,0))

    if datetime.now() > target_open_pos_dt:
        if "trading.ti_cancel_is_final" not in context or \
            not context["trading.ti_cancel_is_final"]:
            candidate_trades = TradeInstruction.get(context, 
                                                 state=[TradeInstruction.S_CANCELLED,TradeInstruction.S_PARTIAL_HOLDING],
                                                 creation_date=(datetime.combine(cur_dt, time.min),cur_dt))

            for trade in candidate_trades:
                if trade.fields['state'] == TradeInstruction.S_PARTIAL_HOLDING or \
                    trade.fields['state'] == TradeInstruction.S_CANCELLED and \
                        (trade.fields['close_out_date'] is None or trade.fields['close_out_date'] > datetime.now()) and \
                         (trade.fields['expire_at'] is None or trade.fields['expire_at'] > datetime.now()):
                    retry_trades.append(trade)
        
        def retry_trade(trade):
            logger.info("Retrying {0} trade instruction: {1}".format(trade.fields["state"],trade.fields["trade_instruction_id"]))
            trade.fields["state"] = TradeInstruction.S_OPEN
            trade.fields["open_order_id"] = None
            trade.update()
            
        for trade in retry_trades:
            if trade.fields["quantity"] > 0:
                open_pos = trade.get_open_position_quantity()
                if open_pos > 0:
                    if trade.fields["close_out_date"] >= datetime.now():
                        side = "S" if open_pos > 0 else "B"
                    else:
                        target_pos = trade.fields["quantity"] if trade.fields['side'] == "B" else -trade.fields["quantity"]
                        trade_gap = target_pos - open_pos
                        side = "B" if trade_gap > 0 else "S" 
                    
                    max_retry_orders = context[PARAM_TRADE_CANCEL_MAX_RETRY_ORDERS]
                    
                    retry_orders = len(Order.get(context,trade_instruction_id=trade.fields["trade_instruction_id"],side=side,
                                                 creation_date=(datetime.combine(cur_dt, time.min),cur_dt),
                                                 orig_order_id=None))
                    
                    if retry_orders < max_retry_orders:
                        retry_trade(trade)
                elif trade.fields['state'] == TradeInstruction.S_ABORTED:
                        retry_trade(trade)

    broker = context[PARAM_BROKER_INTERFACE]
        
    active_trades = TradeInstruction.get(context, state=TradeInstruction.non_terminal_states)

    for active_trade in active_trades:
        if active_trade.fields['close_out_date'] is not None and \
            datetime.now() > active_trade.fields['close_out_date']:
            orders = Order.get(context,trade_instruction_id=active_trade.fields['trade_instruction_id'])
            for order in orders:
                if order.fields['state'] in Order.non_terminal_states and \
                    order.fields['state'] != Order.S_PENDING_CANCEL and \
                    not (order.fields['side'] == ("B" if active_trade.fields['side'] == "S" else "S") and \
                         order.fields['order_type'] == Order.TYPE_MARKET):
                    if order.fields["broker_order_id"] is not None:
                        try:
                            broker.cancel_order(order.fields["broker_order_id"])
                            order.fields["state"] = Order.S_PENDING_CANCEL
                            order.update()
                        except:
                            logger.warning(f"Unable to cancel order: {pprint.pformat(order.fields)}")
                            logger.warning(traceback.format_exc())
                    elif order.fields['state'] in [Order.S_OPEN,Order.S_HELD]:
                        order.fields['state'] = Order.S_CANCELLED
                        order.update()
                    elif order.fields['state'] == Order.S_NETTED_PENDING:
                        pass        # should cancel as part of child orders cancelling
                    else:
                        logger.warning(f"On TI close out, unable to cancel non-terminal order {order.fields['internal_order_id']} due to missing broker id.")
                        
    # go through active trades
    for i in range(len(active_trades)):
        try:
            if context.get("trading.test.live_test",False):
                if active_trades[i].fields["symbol"] != context["trading.test.test_symbol"]:
                    continue
            
            # we have to refresh as the trade could have been altered since it was queried (eg. was crossed by an earlier trade)
            active_trades[i] = TradeInstruction.get(context,trade_instruction_id=active_trades[i].fields['trade_instruction_id'])[0]
            
            if active_trades[i].fields["state"] == TradeInstruction.S_OPEN and \
                (active_trades[i].fields["activation_date"] is None or \
                 active_trades[i].fields["activation_date"] <= datetime.now()):
                if active_trades[i].fields["close_out_date"] is not None and \
                    active_trades[i].fields["close_out_date"] < datetime.now() or \
                    active_trades[i].fields["expire_at"] is not None and \
                    active_trades[i].fields["expire_at"] < datetime.now():
                    active_trades[i].fields["state"] = TradeInstruction.S_CANCELLED
                    active_trades[i].update()
                    logger.info("Open trade instruction {} closed as close out or expire_at date has passed.".format(active_trades[i].fields['trade_instruction_id']))
                    continue
                    
                logger.info( "Processing open trade instruction: {0}".format(active_trades[i].pprint()))
            
                if (target_open_pos_dt is not None and datetime.now() > target_open_pos_dt) and \
                    not context.get("trading.test.live_test",False) and not context.get("trading.immediate_trade_execution",False):
                    logger.error("Trade instruction {} cancelled as target open position dt {} has passed"\
                                 .format(active_trades[i].fields['trade_instruction_id'],
                                         target_open_pos_dt.strftime("%Y-%m-%d %H:%M:%S")))
                    active_trades[i].fields['state'] = TradeInstruction.S_CANCELLED
                    active_trades[i].update()
                    continue

                if active_trades[i].fields["state"] == TradeInstruction.S_OPEN:
                    if active_trades[i].fields['quantity'] == 0:
                        logger.info("Trade Instruction created with quantity 0. Cancelling.")
                        active_trades[i].fields['state'] = TradeInstruction.S_CANCELLED
                        active_trades[i].update()
                        continue

                    if active_trades[i].fields['close_out_date'] <= datetime.now():
                        logger.info("Open Trade Instruction past close out date: Cancelling.")
                        active_trades[i].fields['state'] = TradeInstruction.S_CANCELLED
                        active_trades[i].update()
                        continue
                    
                    if not check_bid_ask_spread(context, active_trades[i].fields['symbol']):
                        if context.get("trading.cancel_ti_for_wide_spreads",False):
                            logger.info("Spreads on symbol {} are too wide. Cancelling trade instruction {}".format(active_trades[i].fields['symbol'],
                                                                                                                    active_trades[i].fields['trade_instruction_id']))
                            active_trades[i].fields['state'] = TradeInstruction.S_CANCELLED
                            active_trades[i].update()
                        else:
                            logger.info("Trade instruction {} on symbol {} held due to wide spreads".format(active_trades[i].fields['trade_instruction_id'],
                                                                                                                    active_trades[i].fields['symbol']))
                        continue
                    
                    update_order = None
                    retry_orders = Order.get(context, trade_instruction_id=active_trades[i].fields["trade_instruction_id"],
                                                      side=active_trades[i].fields["side"],
                                                      creation_date=(datetime.combine(cur_dt, time.min),cur_dt),
                                                      orig_order_id=None )
                    if len(retry_orders) < context[PARAM_TRADE_CANCEL_MAX_RETRY_ORDERS]:
                        max_trade_exposure = context.get(PARAM_SINGLE_TRADE_EXPOSURE,context[PARAM_MAX_POSITION_EXPOSURE])
                        if active_trades[i].fields["expected_notional"] > max_trade_exposure and \
                            not context.get('trading.enforce_static_trade_quantity',False):
                            logger.error("Unable to process trade instruction {}. Expected notional {} is greater than limit {}"\
                                         .format(active_trades[i].fields['trade_instruction_id'],
                                                 active_trades[i].fields['expected_notional'],
                                                 max_trade_exposure))
                            active_trades[i].fields["state"] = TradeInstruction.S_CANCELLED
                        else:
                            if context.get("trading.immediate_trade_execution",False) == True:
                                execute_after = None
                            elif PARAM_TARGET_OPEN_POSITION_TIME in context:
                                execute_after = get_execution_datetime(context) - \
                                                        timedelta(seconds=context.get(PARAM_PRE_OPEN_OPEN_ORDER_TIMING,0))
                            else:
                                execute_after = None
                                
                            pre_open_expire_time = context.get(PARAM_PRE_OPEN_EXPIRE_ORDER_TIMING,None)
                            order = create_order( context, active_trades[i], is_open=True, execute_after=execute_after,
                                                  expire_at = None if pre_open_expire_time is None else \
                                                    get_execution_datetime(context) - timedelta(seconds=context[PARAM_PRE_OPEN_EXPIRE_ORDER_TIMING]) )
                            
                            if order is not None:
                                context[PARAM_LOGGER].info( "Created order: {0}".format(order.pprint()))
                                active_trades[i].fields["open_order_id"] = order.fields["internal_order_id"]
                                update_order = order
                    else:
                        if active_trades[i].fields["trade_instruction_id"] not in ignored_tis_for_retry_msg:
                            context[PARAM_LOGGER].error("Max retry orders reached for trade instruction {}".format(active_trades[i].fields["trade_instruction_id"]))
                            active_trades[i].fields["state"] = TradeInstruction.S_CANCELLED
                            ignored_tis_for_retry_msg.append(active_trades[i].fields["trade_instruction_id"])

                    if update_order is not None:
                        TradeInstruction.batch_update(context,[update_order,active_trades[i]])
                    else:
                        active_trades[i].update_state()
                        active_trades[i].update()

            elif active_trades[i].fields["state"] in [TradeInstruction.S_HOLDING,TradeInstruction.S_PARTIAL_HOLDING,TradeInstruction.S_PARTIAL_CLOSED] and \
                (not context.get('trading.continuous.enable',False) and \
                                 (active_trades[i].fields["close_out_date"]-timedelta(minutes=context.get(PARAM_PRE_CLOSE_INSTRUCTION_CLOSEOUT_TIMING,3))) <= datetime.now() \
                or \
                context.get('trading.continuous.enable',False) and datetime.now() >= active_trades[i].fields["close_out_date"] ): 
                
                logger.debug( "Processing trade instruction for close out: {0}".format(active_trades[i].pprint()))

                if broker.is_market_open():
                    open_orders = Order.get(context, trade_instruction_id=active_trades[i].fields["trade_instruction_id"],
                                                      state=Order.non_terminal_states)
                    for order in open_orders:
                        if order.fields['state'] != Order.S_PENDING_CANCEL:
                            try:
                                if order.fields["broker_order_id"] is not None:
                                    broker.cancel_order(order.fields["broker_order_id"])
                                order.fields["state"] = Order.S_PENDING_CANCEL
                                order.update()
                            except:
                                logger.error("Unable to cancel order id {0}".format(order.fields["internal_order_id"]))
                                logger.error(traceback.format_exc())
                    
                    if active_trades[i].fields["close_out_date"].date() < datetime.now().date() or \
                        context.get("trading.immediate_trade_execution",False) == True:
                        execute_after = None
                    else:
                        execute_after = active_trades[i].fields["close_out_date"]

                     
                    order = create_order( context, active_trades[i], is_open=False, execute_after=execute_after, do_not_expire=True )
                    
                    if order is not None:
                        context[PARAM_LOGGER].debug( "Created order: {0}".format(order.pprint()))
                    
                        # change trade instruction state
                        active_trades[i].fields["close_order_id"] = order.fields["internal_order_id"]
                        
                        TradeInstruction.batch_update(context,[order,active_trades[i]])
                    else:
                        context[PARAM_LOGGER].debug( "Failed to create order for instruction: {0}".format(active_trades[i].fields['trade_instruction_id']))
                        active_trades[i].fields['state'] = TradeInstruction.S_CANCELLED
                        active_trades[i].update()

                else:
                    logger.warning("Skipping trade instruction {} for closing. Market is closed".format(active_trades[i].fields["trade_instruction_id"]))
                    
        except:
            logger.error("Unable to process trade instruction id {0}.".format(active_trades[i].fields["trade_instruction_id"])) 
            logger.error(traceback.format_exc())
    
    return

    


        
