import pprint, traceback, copy, sys, logging
from datetime import datetime, timedelta, time
import time as ptime

import numpy as np
import pandas as pd

from math import copysign

from algotrading.trading.Objects import TradeInstruction, Order
from algotrading.trading.Common import get_execution_datetime, get_current_is_opening_order_flag

from algotrading.trading.Pricing import get_bidask, get_symbol_mark

from algotrading.ContextFields import *



def amend_order( context, order ):
    amended_order = context[PARAM_BROKER_INTERFACE].amend_order( order )
    
    # if order was replaced, persist the new order
    if amended_order != order:
        amended_order.fields['trade_instruction_id'] = order.fields['trade_instruction_id']
        new_int_order_id = amended_order.insert(last_insert_id=True)
        
        if order.fields['trade_instruction_id'] is not None:
            ti = TradeInstruction.get(context,trade_instruction_id=order.fields['trade_instruction_id'])[0]
            
            updated = False
            for field in ['open_order_id','close_order_id']:
                if order.fields['internal_order_id'] == ti.fields[field]:
                    ti.fields[field] = new_int_order_id
                    updated=True
            
            if updated:
                ti.update()

    return amended_order


def check_bracket_order_pricing( context, original_order, bracket_order ):
    
    if context.get("trading.disable_bracket_order_price_checks",True):
        return
    
    if original_order.fields["order_class"] is not None:
        try:
            if original_order.fields["state"] == Order.S_FILLED and bracket_order.fields["state"] == Order.S_WORKING:
                instructions = TradeInstruction.get(context,trade_instruction_id=original_order.fields['trade_instruction_id'])
                assert(len(instructions) == 1)
                
                tolerance = context.get('trading.bracket.amend_order_tolerance',0.0025)
                
                if bracket_order.fields['order_type'] == Order.TYPE_STOP:
                    factor = instructions[0].fields['stop_loss']
                else:
                    factor = instructions[0].fields['stop_gain']
                
                expected_bracket_price = get_bracket_price( context, original_order.fields["side"], 
                                                      original_order.fields["avg_fill_price"], factor,
                                                      bracket_order.fields["order_type"] == Order.TYPE_STOP )
                
                bracket_price_field = "stop_price" if bracket_order.fields["order_type"] == Order.TYPE_STOP else \
                                       "limit_price"
                
                if bracket_order.fields[bracket_price_field] < (1-tolerance)*expected_bracket_price or \
                    bracket_order.fields[bracket_price_field] > (1+tolerance)*expected_bracket_price:
                    
                    context[PARAM_LOGGER].info( "Order {} limit/stop price to be amended from {} to {}"\
                                                    .format(bracket_order.fields["broker_order_id"],
                                                            bracket_order.fields[bracket_price_field],
                                                            expected_bracket_price) )
                    amended_order = copy.copy(bracket_order)
                    amended_order.fields = copy.deepcopy(bracket_order.fields)
                    amended_order.fields[bracket_price_field] = expected_bracket_price
                    amended_order = amend_order( amended_order )
                    
                    # need to amend open order otherwise incoming trades may not be matched properly if 2 TIs are identical in quantity/side
                    open_order = Order.get(context,instructions[0].fields['open_order_id'])
                    if amended_order.fields['order_type'] == Order.TYPE_STOP:
                        open_order[0].fields['bkt_stop_loss'] = amended_order.fields['stop_price']
                    else:
                        open_order[0].fields['bkt_stop_gain'] = amended_order.fields['limit_price']
                    open_order[0].update
                    
                    bracket_order.fields["state"] = Order.S_PENDING_CANCEL
                    bracket_order.update()
                    
        except:
            context[PARAM_LOGGER].error("Exception in check_stop_order_pricing")
            context[PARAM_LOGGER].error(traceback.format_exc())
            

def get_bracket_price( context, side, price, factor, stop_loss = True, min_spread = 0 ):
    
    if side == "S":
        if stop_loss:
            ret_price = round(price * (1+factor),2)
        else:
            ret_price = round(price * (1-factor),2)
    else:
        if stop_loss:
            ret_price = round(price * (1-factor),2)
        else:
            ret_price = round(price * (1+factor),2) 
    
    min_stop_delta = context.get("trading.min_stop_price_delta",0.04)
    min_stop_delta = min_spread if min_spread > min_stop_delta else min_stop_delta
    
    if abs(ret_price-price) < min_stop_delta:
        if side == "S" and not stop_loss or \
            side == "B" and stop_loss:
            ret_price = price - min_stop_delta
        else:
            ret_price = price + min_stop_delta

    return np.round(ret_price,2)



def set_bracket_stop_prices( context, instruction, order ):

    symbol = order.fields['symbol']
        
    min_spread = 0
    current_price = None
    price = None
    if order.fields["order_type"] == Order.TYPE_LIMIT:
        price = order.fields["limit_price"]
        min_spread = 0
    elif order.fields["order_type"] == Order.TYPE_STOP:
        price = order.fields["stop_price"]
        min_spread = 0
        
    if PARAM_STREAMING_PRICES in context:
        bidask_df = get_bidask(context,symbol)
        if len(bidask_df) == 1:
            min_spread = abs(bidask_df.loc[symbol].bid - bidask_df.loc[symbol].ask) * context.get("trading.min_spread_multiple_for_stops",2)
            current_price = bidask_df.loc[symbol].bid if order.fields["side"] == "S" else bidask_df.loc[symbol].ask

    if current_price is None:
        current_price = context[PARAM_BROKER_INTERFACE].get_quote( symbol, order.fields["side"] )
        
    if price is None: # market orders
        price = current_price

    if (order.fields["order_type"] == Order.TYPE_STOP and \
         order.fields["side"] == 'S' or \
         order.fields["order_type"] == Order.TYPE_LIMIT and \
         order.fields["side"] == 'B') and \
         current_price <= price or \
         (order.fields["order_type"] == Order.TYPE_STOP and \
         order.fields["side"] == 'B' or \
         order.fields["order_type"] == Order.TYPE_LIMIT and \
         order.fields["side"] == 'S') and \
         current_price >= price:
        context[PARAM_LOGGER].info( f"Price {current_price} has moved outside of stop/limit bound {price}. Converting to order id {order.fields['internal_order_id']} to market order" )
        order.fields['order_type'] = Order.TYPE_MARKET
        order.fields['stop_price'] = None
        order.fields['limit_price'] = None
        order.fields['time_in_force'] = Order.TIF_DAY
        price = current_price

    stop_loss = instruction.fields.get("stop_loss",None)
    if stop_loss is not None:
        if stop_loss < 1:   # assume percentage
            stop_price = get_bracket_price( context, order.fields["side"], price, stop_loss, stop_loss = True, min_spread = min_spread )
        else:   # assume price
            stop_price = stop_loss
        order.fields["bkt_stop_loss"] = stop_price
        
    stop_gain = instruction.fields.get("stop_gain",None)
    if stop_gain is not None:
        if stop_gain < 1: # assume percentage
            stop_price = get_bracket_price( context, order.fields["side"], price, stop_gain, stop_loss = False, min_spread = min_spread )
        else:   # other absolute price
            stop_price = stop_gain
        order.fields["bkt_stop_gain"] = stop_price
    
    return order

    


def create_order( context, instruction, is_open = True, execute_after = None, expire_at = None, do_not_expire = False ):

    open_pos = instruction.get_open_position_quantity()
    
    if open_pos == 0 and not is_open:
        context[PARAM_LOGGER].warning("Attempted to create close order for instruction {} when no open position exists"\
                                      .format(instruction.fields["trade_instruction_id"]))
        return None
    
    # open an order
    order = Order.new( context, trade_instruction_id=instruction.fields["trade_instruction_id"] )
    order.fields["state"] = Order.S_OPEN
    order.fields["symbol"] = instruction.fields["symbol"]
    order.fields["algo_id"] = instruction.fields["algo_id"]

    if is_open and instruction.fields['limit_price'] is not None and \
        instruction.fields['stop_price'] is not None:
        order.fields["order_type"] = Order.TYPE_STOP_LIMIT
        order.fields['limit_price'] = instruction.fields['limit_price']
        order.fields['stop_price'] = instruction.fields['stop_price']
    elif is_open and instruction.fields['limit_price'] is not None:
        order.fields["order_type"] = Order.TYPE_LIMIT
        order.fields['limit_price'] = instruction.fields['limit_price']
    elif is_open and instruction.fields['stop_price'] is not None:
        order.fields["order_type"] = Order.TYPE_STOP
        order.fields['stop_price'] = instruction.fields['stop_price']
    else:
        order.fields["order_type"] = Order.TYPE_MARKET
        
    if is_open:
        order.fields["quantity"] = instruction.fields["quantity"] - np.abs(open_pos)
        order.fields["side"] = instruction.fields["side"]
    else:
        order.fields["quantity"] = np.abs(open_pos)
        if open_pos > 0:
            order.fields["side"] = "S"
        else:
            order.fields["side"] = "B"
            
    
    order.fields["is_opening_order"] = 1 if is_open else 0
    
    if "trading.test.off_market_orders" in context and \
        context["trading.test.off_market_orders"] == True and \
        is_open:
        order.fields["order_type"] = Order.TYPE_LIMIT
        current_price = get_symbol_mark(context, instruction.fields["symbol"])

        if order.fields["side"] == "B":
            order.fields["limit_price"] = current_price * 0.7
        else:
            order.fields["limit_price"] = current_price * 1.3
            
    order.fields["execute_after"] = execute_after
        
    assert( order.fields["quantity"] > 0 )

    if not do_not_expire:
        if instruction.fields['expire_at'] is not None:
            order.fields["expire_at"] = instruction.fields['expire_at']
        elif "trading.test.order_expiry_time" in context.keys():
            order.fields["expire_at"] = datetime.now() + timedelta(minutes=context["trading.test.order_expiry_time"])
        elif expire_at is None:
            order.fields['expire_at'] = get_execution_datetime(context,datetime.now(),PARAM_TARGET_CLOSE_POSITION_TIME)
        else:
            order.fields['expire_at'] = expire_at
        
    if PARAM_TRADE_ORDERS_TIME_IN_FORCE in context.keys():
        order.fields["time_in_force"] = context[PARAM_TRADE_ORDERS_TIME_IN_FORCE]
    elif order.fields['expire_at'] is not None and \
         order.fields['expire_at'] > datetime.now().replace(hour=23,minute=59,second=59):
        order.fields["time_in_force"] = Order.TIF_GTC
    else:
        order.fields["time_in_force"] = Order.TIF_DAY
        
    if instruction.fields['close_out_date'] is not None and \
        order.fields['expire_at'] is not None and \
        instruction.fields['close_out_date'] < order.fields['expire_at']:
        order.fields['expire_at'] = instruction.fields['close_out_date'] 
            
    if is_open:
        if instruction.fields.get('stop_loss',None) is not None and \
            instruction.fields.get('stop_gain',None) is not None:
            order.fields["order_class"] = Order.CLASS_BRACKET
        elif instruction.fields.get('stop_loss',None) is not None:
            order.fields["order_class"] = Order.CLASS_OTO
        elif instruction.fields.get('stop_gain',None) is not None:
            order.fields["order_class"] = Order.CLASS_OTO
        
        if order.fields["order_class"] is not None:
            set_bracket_stop_prices( context, instruction, order )

    if is_open and PARAM_OPEN_ORDER_EXECUTION_WINDOW in context:
        target_open_pos_dt = get_execution_datetime(context)
        if datetime.now() > \
            (target_open_pos_dt + timedelta(seconds=context[PARAM_OPEN_ORDER_EXECUTION_WINDOW])) and \
            not context.get("trading.test.live_test",False):
            order.fields['state'] = Order.S_REJECTED
            order.fields['state_detail'] = 'window for opening positions has passed'

    order.update()
    
    return order


def check_orders_is_opening_order_flag( context, positions = None ):
    
    logger = context[PARAM_LOGGER]
    active_orders = Order.get(context,state=[Order.S_WORKING])

    if positions is None:
        positions = context[PARAM_BROKER_INTERFACE].get_open_positions()
    
    for order in active_orders:
        try:
            correct_open_order_flag = get_current_is_opening_order_flag(context,order,positions)
            
            if order.fields['is_opening_order'] != correct_open_order_flag:
                logger.debug("Order {} has incorrect is_opening_order flag {} based on current positions. Amending to {}."\
                             .format(order.fields['internal_order_id'],order.fields['is_opening_order'],correct_open_order_flag))
                order.fields['is_opening_order'] = correct_open_order_flag
                
                try:
                    amend_order(context,order)
                except:
                    logger.error(f"Unable to amend is_opening_order flag on order {order.fields['internal_order_id']}. Flag will be updated internally to prevent looping.")
                    logger.error(traceback.format_exc())
                
                order.update()
        except:
            logger.error("Unable to check is_opening_order flag for order {}".format(order.fields['internal_order_id']))
            logger.error(traceback.format_exc())


def create_closing_leg( context, ti, order_type ):
    
    # get opening order
    open_order = Order.get(context,internal_order_id=ti.fields['open_order_id'])[0]
    
    stop_price = None
    limit_price = None
    if order_type == Order.TYPE_STOP:
        if open_order.fields['bkt_stop_loss'] is None:
            raise ValueError(f"Missing bkt_stop_loss on opening order {open_order.fields['internal_order_id']}")
        stop_price = open_order.fields['bkt_stop_loss']
    elif order_type == Order.TYPE_LIMIT: 
        if open_order.fields['bkt_stop_gain'] is None:
            raise ValueError(f"Missing bkt_stop_gain on opening order {open_order.fields['internal_order_id']}")
        limit_price = open_order.fields['bkt_stop_gain']
    else:
        raise ValueError(f"Invalid order type {order_type} specified")

    # create order
    new_order = create_order( context, ti, is_open = False, do_not_expire = True )
    new_order.fields['order_type'] = order_type
    new_order.fields['time_in_force'] = Order.TIF_GTC
    new_order.fields['bkt_stop_loss'] = stop_price
    new_order.fields['bkt_stop_gain'] = limit_price
    
    new_order.update()


def create_closing_oco( context, ti ):
    # get opening order
    open_order = Order.get(context,internal_order_id=ti.fields['open_order_id'])[0]
    
    # create order
    new_order = create_order( context, ti, is_open = False, do_not_expire = True )
    new_order.fields['order_type'] = Order.TYPE_LIMIT
    new_order.fields['order_class'] = Order.CLASS_OCO
    new_order.fields['time_in_force'] = Order.TIF_GTC
    new_order.fields['bkt_stop_loss'] = open_order.fields['bkt_stop_loss']
    new_order.fields['bkt_stop_gain'] = open_order.fields['bkt_stop_gain']
    new_order.fields['limit_price'] = open_order.fields['bkt_stop_gain']
    
    new_order.update()
    

def match_existing_order( context, broker_order ):
    logger = context[PARAM_LOGGER]
    
    if broker_order.fields['orig_order_id'] is not None:
        orders = Order.get(context,broker_order_id=broker_order.fields['orig_order_id'])
        if len(orders) == 1:
            new_order = Order.new( context )
            order_id = new_order.fields["internal_order_id"]
            
            if orders[0].fields['side'] == broker_order.fields['side']: # assume order amend if side is equal
                new_order.fields = orders[0].fields.copy()
                
            new_order.fields.update( {k: v for k,v in broker_order.fields.items() if v is not None} )
            new_order.fields["internal_order_id"] = order_id
            new_order.fields['trade_instruction_id'] = orders[0].fields['trade_instruction_id']

            if orders[0].fields['trade_instruction_id'] is not None:
                tis = TradeInstruction.get(context,trade_instruction_id=orders[0].fields['trade_instruction_id'])

                # for order amends, replace order_id if listed as open/close on TI                
                if orders[0].fields['side'] == new_order.fields['side']:
                    order_id_fields = ['open_order_id','close_order_id']
                    for order_id_field in order_id_fields:
                        if orders[0].fields['internal_order_id'] == tis[0].fields[order_id_field]:
                            tis[0].fields[order_id_field] = order_id
                            tis[0].update()
                            break
                    
            new_order.update()
                         
            return True
        elif len(orders) > 1:
            logger.warning("Attempting to match incoming order. More than 1 order found for orig_order_id {}".format(broker_order.fields['orig_order_id']))
            return False
        
    
    possible_orders = Order.get(context,
                                side=broker_order.fields["side"],
                                quantity=broker_order.fields["quantity"],
                                symbol=broker_order.fields["symbol"],
                                state=Order.S_WORKING,
                                broker_order_id=None,
                                order_type=broker_order.fields["order_type"],
                                creation_date=(datetime.combine(broker_order.fields["creation_date"].date(), time.min), 
                                               datetime.combine(broker_order.fields["creation_date"].date(), time.max)))
    
    
    def matching_order_types(order1,order2):
        return order1.fields['order_type'] == Order.TYPE_STOP and \
               order1.fields['stop_price'] == order2.fields['stop_price'] or \
               order1.fields['order_type'] == Order.TYPE_LIMIT and \
               order1.fields['limit_price'] == order2.fields['limit_price'] or \
               order1.fields['order_type'] == Order.TYPE_MARKET
        
    for order in possible_orders:
        if matching_order_types(order,broker_order):
            order.fields["broker_order_id"] = broker_order.fields["broker_order_id"]
            order.update()
            return True

    logger.info("Unable to match incoming broker order {}".format(broker_order.fields["broker_order_id"]))
    return False
    
    
    
def persist_unknown_order( context, broker_order ):
    context[PARAM_LOGGER].warning("Persisting unknown broker order {}".format(broker_order.fields["broker_order_id"]))
    order = Order.new( context )
    order_id = order.fields["internal_order_id"]
    order.fields = broker_order.fields
    order.fields["internal_order_id"] = order_id
    order.fields['trade_instruction_id'] = None
        
    order.update()    
    
    
def handle_unknown_order( context, broker_order ):
    logger = context[PARAM_LOGGER]
    
    # check if we are created bracket orders
    #if PARAM_GTC_STOP_LOSS_THRESHOLD in context.keys() or \
    #    PARAM_GTC_STOP_GAIN_THRESHOLD in context.keys():
    
    #logger.info("Attempting to match broker order {}".format(broker_order.fields['broker_order_id']))
    
    # order amends
    found_ti = None
    if broker_order.fields['orig_order_id'] is not None:
        orders = Order.get(context,broker_order_id=broker_order.fields['orig_order_id'])
        if len(orders) == 1:
            found_ti = TradeInstruction.get(context,trade_instruction_id=orders[0].fields['trade_instruction_id'])
            if found_ti is not None:
                found_ti = found_ti[0]
        
    # order stop/limit triggers
    if found_ti is None:
        # look for a matching order today with same quantity/symbol and opposite side
        opposite_side = "B" if broker_order.fields["side"] == "S" else "S"
        
        possible_tis = []
        possible_orders = None
        if broker_order.fields['stop_price'] is not None:
            possible_orders = Order.get(context,
                                                state=Order.non_terminal_states + [Order.S_FILLED],
                                                side=opposite_side,
                                                quantity=broker_order.fields['quantity'],
                                                bkt_stop_loss=broker_order.fields['stop_price'],
                                                creation_date=(
                                                    datetime.combine((datetime.now() - timedelta(days=context[PARAM_DATABASE_HISTORY_LOOKBACK])).date(), time.min),
                                                    datetime.combine(datetime.now().date(), time.max)),
                                                symbol=broker_order.fields["symbol"] )
        elif broker_order.fields['limit_price'] is not None:
            possible_orders = Order.get(context,
                                                state=Order.non_terminal_states + [Order.S_FILLED],
                                                side=opposite_side,
                                                quantity=broker_order.fields['quantity'],
                                                bkt_stop_gain=broker_order.fields['limit_price'],
                                                creation_date=(
                                                    datetime.combine((datetime.now() - timedelta(days=context[PARAM_DATABASE_HISTORY_LOOKBACK])).date(), time.min),
                                                    datetime.combine(datetime.now().date(), time.max)),
                                                symbol=broker_order.fields["symbol"] )
        
        if possible_orders is not None:
            # put filled orders first as they are the most likely candidates
            filled_orders = [o for o in possible_orders if o.fields['state'] == Order.S_FILLED]
            filled_ti_ids = list(set([possible_order.fields['trade_instruction_id'] for possible_order in filled_orders if possible_order.fields['trade_instruction_id'] is not None]))
            
            other_orders = [o for o in possible_orders if o.fields['state'] != Order.S_FILLED]
            notfilled_ti_ids = list(set([possible_order.fields['trade_instruction_id'] for possible_order in other_orders if possible_order.fields['trade_instruction_id'] is not None]))
            
            all_ti_ids = filled_ti_ids + notfilled_ti_ids
            
            for ti_id in all_ti_ids:
                ti = TradeInstruction.get(context, trade_instruction_id=ti_id)[0]
                if ti.fields['state'] in TradeInstruction.non_terminal_states:
                    possible_tis.append(ti)
        else:
            possible_tis = TradeInstruction.get(context,
                                                state=TradeInstruction.non_terminal_states,
                                                side=opposite_side,
                                                creation_date=(
                                                    datetime.combine((datetime.now() - timedelta(days=context[PARAM_DATABASE_HISTORY_LOOKBACK])).date(), time.min),
                                                    datetime.combine(datetime.now().date(), time.max)),
                                                symbol=broker_order.fields["symbol"] )
        
        # find the correct TI
        # convoluted -- needs to be cleaned up
        source_order_id = None
        for ti in possible_tis:
            ti_orders = Order.get(context,trade_instruction_id=ti.fields['trade_instruction_id'])
            
            ti_has_stop_open_order = False
            ti_has_this_order_type = False
            
            for ti_order in ti_orders:
                if ti_order.fields['side'] == opposite_side and \
                    ti_order.fields['order_class'] in [Order.CLASS_OTO, Order.CLASS_BRACKET] and \
                    ti_order.fields['quantity'] == broker_order.fields['quantity'] and \
                    ti_order.fields['state'] != Order.S_CANCELLED and \
                    (broker_order.fields['order_type'] == Order.TYPE_STOP and \
                        ti_order.fields['bkt_stop_loss'] is not None and \
                        broker_order.fields['stop_price'] == ti_order.fields['bkt_stop_loss'] or \
                    broker_order.fields['order_type'] == Order.TYPE_LIMIT and \
                        ti_order.fields['bkt_stop_gain'] is not None and \
                        broker_order.fields['limit_price'] == ti_order.fields['bkt_stop_gain']):
                    ti_has_stop_open_order = True
                    source_order_id = ti_order
                    
                if ti_order.fields['side'] == broker_order.fields['side'] and \
                    ti_order.fields['order_type'] == broker_order.fields['order_type'] and \
                    ti_order.fields['quantity'] == broker_order.fields['quantity'] and \
                    ti_order.fields['state'] != Order.S_CANCELLED:
                    ti_has_this_order_type = True
                    
            if ti_has_stop_open_order and not ti_has_this_order_type:
                found_ti = ti
                break
   
    # if order is rejected, it may have an originated order in the REJECTED state (eg. this order is a REJECTED HELD order representing a OTO stop loss
    if found_ti is None and \
        broker_order.fields['state'] == Order.S_REJECTED:
        
        # a rejected OTO stop/limit order would have same creation_date as it is generated on the broker at the same time as the originating order
        orig_orders = Order.get(context,state=Order.S_REJECTED,
                                             quantity=broker_order.fields['quantity'],
                                             side="B" if broker_order.fields["side"] == "S" else "S",
                                             creation_date=broker_order.fields['creation_date'].replace(microsecond=0))
        if orig_orders is not None and \
            len(orig_orders) == 1:
            found_ti = TradeInstruction.get(context,trade_instruction_id=orig_orders[0].fields['trade_instruction_id'])[0]
   
    if found_ti is not None:
        # open order flag on broker order is not necessarily reflective of this trade. 
        # it looks at the overall position across all trades so we don't overwrite it.
        is_opening_order = 1 if broker_order.fields['side'] == found_ti.fields['side'] else 0
        if broker_order.fields["is_opening_order"] is None:
            broker_order.fields["is_opening_order"] = is_opening_order
            
        # associate with the trade instruction of the original bracket order + set expire date/time
        broker_order.fields["trade_instruction_id"] = found_ti.fields["trade_instruction_id"]
        broker_order.fields["algo_id"] = found_ti.fields["algo_id"]
        
        if broker_order.fields['side'] == found_ti.fields['side']:
            broker_order.fields["expire_at"] = found_ti.fields["expire_at"]
            broker_order.fields["execute_after"] = found_ti.fields["execute_after"]
        elif broker_order.fields["order_type"] in [Order.TYPE_STOP, Order.TYPE_LIMIT]:
            broker_order.fields["expire_at"] = get_execution_datetime( context, found_ti.fields["creation_date"],
                                                                       PARAM_TARGET_CLOSE_POSITION_TIME )
        
        order_id = broker_order.insert( last_insert_id = True )
        
        if broker_order.fields['state'] in Order.non_terminal_states and \
            is_opening_order == 1:
                found_ti.fields['open_order_id'] = broker_order.fields['internal_order_id']
        elif is_opening_order == 0:
            do_not_replace = False
            if found_ti.fields['close_order_id'] is not None and \
                found_ti.fields['close_order_id'] != broker_order.fields['internal_order_id']:
                orig_close_order = Order.get(context,internal_order_id=found_ti.fields['close_order_id'])
                if len(orig_close_order) == 1 and \
                    orig_close_order[0].fields['state'] in Order.non_terminal_states:
                    do_not_replace = True
                
            if not do_not_replace:
                found_ti.fields['close_order_id'] = broker_order.fields['internal_order_id']
                
        logger.info("Persisting incoming unmatched broker order {}. Matched to trade instruction {}."\
                    .format(broker_order.fields["broker_order_id"],found_ti.fields["trade_instruction_id"]))

        TradeInstruction.batch_update(context, [found_ti,broker_order])        
        
        if broker_order.fields["order_type"] in [Order.TYPE_STOP, Order.TYPE_LIMIT]:
            check_bracket_order_pricing( context, source_order_id, broker_order )
        
        return True

    # check if this is associated with a netted order (eg. sell to close + sell to open netted into a single sell. Orders may be returned that are split up between the two.
    net_orders = Order.get(context,
                        side=broker_order.fields["side"],
                        symbol=broker_order.fields["symbol"],
                        state=Order.S_WORKING,
                        order_type=broker_order.fields["order_type"],
                        is_net_order=1,
                        creation_date=(datetime.combine(broker_order.fields["creation_date"].date(), time.min), 
                                       datetime.combine(broker_order.fields["creation_date"].date(), time.max)))
    
    if len(net_orders) == 1:
        constituent_orders = Order.get(context,
                                       netted_order_id = net_orders[0].fields['internal_order_id'])
        
        if len(constituent_orders) > 0:     
            decouple_order_id = None       
            for order in constituent_orders:
                if order.fields['quantity'] == broker_order.fields['quantity']:
                    # break the netting (remote system has de-netted the trades)
                    decouple_order_id = order.fields['internal_order_id']
                    ti_id = order.fields['trade_instruction_id']
                    order.fields = broker_order.fields
                    order.fields['internal_order_id'] = decouple_order_id
                    order.fields['trade_instruction_id'] = ti_id
                    order.update()
                    break

            if decouple_order_id is not None:
                for order in constituent_orders:
                    if order.fields['internal_order_id'] != decouple_order_id:
                        order.fields['netted_order_id'] = None
                        order.fields['state'] = Order.S_WORKING
                        order.update()
            
                net_orders[0].fields['state'] = Order.S_CANCELLED
                net_orders[0].update()
                
                return True
            else:
                logger.error("Could not match to a constituent of netted order {}".format(net_orders[0].fields['internal_order_id']))
        else:
            logger.error("Could not find constituent orders to netted order {}".format(net_orders[0].fields['internal_order_id']))
        
    elif len(net_orders) > 1:
        logger.warning("Multiple matches for possible netted order match for incoming broker order {}".format(broker_order.fields['broker_order_id']))
    
    logger.error("Unable to find matching order in database for broker order {0}".format(broker_order.fields["broker_order_id"]))
    persist_unknown_order(context,broker_order)
    return True
    



def cross_orders( context, orders, netted_order_id, crossing_price ):
    
    logger = context[PARAM_LOGGER]

    buy_orders = []
    sell_orders = []
        
    # sort orders into long and short
    for order in orders:
        if order.fields["side"] == "B":
            buy_orders.append(order)
        else:
            sell_orders.append(order)
            
    # try to fill all buy orders
    for order in buy_orders:
        provided_liq = order.fields["quantity"] - order.fields["filled_qty"]
        
        if provided_liq == 0:
            logger.warn("Unexpected provided liquidity is 0 on buy crossing order {0}".format(order.fields["internal_order_id"]))
        else:
            # go through sell orders    
            for cross_order in sell_orders:
                # check how much liquidity is needed
                needed_liq = cross_order.fields["quantity"] - cross_order.fields["filled_qty"]
                if needed_liq > 0:
    
                    # set fields
                    cross_order.fields["execution_date"] = datetime.now()
                    cross_order.fields["netted_order_id"] = netted_order_id
    
                    # can we fully cross the sell order?
                    if needed_liq < provided_liq:
                        provided_liq = provided_liq - needed_liq
                        cross_order.add_fill(needed_liq,crossing_price)
                        cross_order.fields["state"] = Order.S_CROSS
                        
                        logger.debug("Crossed order {0} with {1}".format(cross_order.fields["internal_order_id"],
                                                                               order.fields["internal_order_id"]))
                        logger.debug("Liquidity remaining on order {0} is {1}".format(order.fields["internal_order_id"],provided_liq))
                    # we should then fully cross the buy order
                    elif needed_liq >= provided_liq:
                        logger.debug("Crossed order {0} with {1}".format(order.fields["internal_order_id"],
                                                                               cross_order.fields["internal_order_id"]))
                        logger.debug("Liquidity remaining on order {0} is {1}".format(cross_order.fields["internal_order_id"],needed_liq-provided_liq))

                        cross_order.add_fill(provided_liq,crossing_price)

                        provided_liq = 0
                        
                        if cross_order.fields["quantity"] == cross_order.fields["filled_qty"]:
                            cross_order.fields["state"] = Order.S_CROSS
                        break
                
            # adjust the buy order's filled qty to reflect what was filled on other orders
            fill_qty = order.fields["quantity"] - order.fields["filled_qty"] - provided_liq
            order.add_fill(fill_qty,crossing_price)
            
            order.fields["filled_qty"] = order.fields["quantity"] - provided_liq
            if provided_liq == 0:
                order.fields["state"] = Order.S_CROSS
                order.fields["netted_order_id"] = netted_order_id
                order.fields["execution_date"] = datetime.now()
            else: # no more liquidity to needed on the sell side
                break
        
    return
            

def create_netted_order( context, orders, positions = None ):

    symbol = orders[0].fields["symbol"]
    
    # create netting order
    net_order = Order.new( context )

    # get market price    
    crossing_price = get_symbol_mark(context,symbol)
                
    # cross the orders
    cross_orders( context, orders, net_order.fields["internal_order_id"], crossing_price )

    # sort filled and unfilled orders
    filled_orders = []
    unfilled_orders = []
    for order in orders:
        if order.fields["state"] == Order.S_CROSS:
            filled_orders.append(order)
        else:
            unfilled_orders.append(order)
    
    # update the netted order
    net_order.fields["state"] = Order.S_WORKING
    net_order.fields["symbol"] = symbol
    net_order.fields["order_type"] = Order.TYPE_MARKET    
    net_order.fields["is_net_order"] = 1 

    if len(unfilled_orders) > 0:
        net_order.fields["state"] = Order.S_OPEN
        net_order.fields["side"] = unfilled_orders[0].fields["side"]
        
        total_qty = 0
        for order in unfilled_orders:
            total_qty = total_qty + order.fields["quantity"] - order.fields["filled_qty"]
            order.fields["netted_order_id"] = net_order.fields["internal_order_id"]
            order.fields["state"] = Order.S_NETTED_PENDING
        
        net_order.fields["quantity"] = total_qty
        
        if positions is None or symbol not in positions or \
            (positions[symbol].fields["quantity"] <= 0 and order.fields["side"] == "S" or \
             positions[symbol].fields["quantity"] >= 0 and order.fields["side"] == "B"):
            net_order.fields["is_opening_order"] = 1
        else:
            net_order.fields["is_opening_order"] = 0
            
        net_order.fields['execute_after'] = None
        
        if net_order.fields['symbol'] in positions:
            trade_qty = net_order.fields['quantity'] if net_order.fields['side'] == "B" else \
                         -1*net_order.fields['quantity']
            pos_qty = positions[net_order.fields['symbol']].fields['quantity']
            
            if abs(trade_qty) > abs(pos_qty) and \
                np.sign(trade_qty) != np.sign(pos_qty):
                context[PARAM_LOGGER].info("Adjusting order quantity on {} to avoid overbought/oversold".format(net_order.fields['internal_order_id']))
                net_order.fields['quantity'] = abs(pos_qty)
                
                for one_order in orders:
                    if net_order.fields['execute_after'] is None or \
                        one_order.fields['execute_after'] is not None and \
                        net_order.fields['execute_after'] > one_order.fields['execute_after']:
                        net_order.fields['execute_after'] = one_order.fields['execute_after']

        if net_order.fields['execute_after'] is None:
            net_order.fields['execute_after'] = datetime.now()
            for one_order in orders:
                if one_order.fields['execute_after'] is not None and \
                    one_order.fields['execute_after'] > net_order.fields['execute_after']:
                        net_order.fields['execute_after'] = one_order.fields['execute_after']
        
        latest_expire_at = None
        for order in orders:
            if order.fields['expire_at'] is not None:
                if latest_expire_at is None:
                    latest_expire_at = order.fields['expire_at']
                elif order.fields['expire_at'] > latest_expire_at:
                    latest_expire_at = order.fields['expire_at']
        
        net_order.fields['expire_at'] = latest_expire_at if latest_expire_at is not None else \
                                            datetime.now() + timedelta(seconds=context.get(PARAM_MAX_ORDER_LIFE,60))
                                                
    else:
        net_order.fields["execution_date"] = datetime.now()
        net_order.fields["quantity"] = 0
        net_order.fields["state"] = Order.S_FILLED
    
    return (orders, net_order)


def get_current_order_exposure( context ):
    logger = context[PARAM_LOGGER]
    broker = context[PARAM_BROKER_INTERFACE]
    positions = broker.get_open_positions()


    # get current total exposure by open position
    cur_total_exposure = 0
    for symbol,pos in positions.items():
        if symbol in context.get("trading.symbol_filter",[]):
            logger.info("Symbol {} filtered. Skipping for presubmission checks.".format(symbol))
            continue
        
        add_exposure = copysign(pos.fields["market_value"],-1) if pos.fields["quantity"] < 0 else pos.fields["market_value"]
        cur_total_exposure = cur_total_exposure + add_exposure

    new_total_exposure = cur_total_exposure

    # get potential exposure by open order
    processed_orders = []
    close_order_transaction_ids = []
    db_orders = Order.get(context,  state=[Order.S_OPEN, Order.S_WORKING, Order.S_HELD])
    for db_order in db_orders:
        processed_orders.append(db_order.fields["internal_order_id"])

        if db_order.fields["trade_instruction_id"] is not None:
            ti = TradeInstruction.get(context,trade_instruction_id = db_order.fields["trade_instruction_id"])[0]
            if ti.fields['side'] != db_order.fields['side']:
                if db_order.fields["trade_instruction_id"] in close_order_transaction_ids:
                    continue
                else:
                    close_order_transaction_ids.append(db_order.fields["trade_instruction_id"])

        if db_order.fields["order_type"] != Order.TYPE_STOP:
            cur_price = get_symbol_mark(context,db_order.fields["symbol"])
            add_exposure = (db_order.fields["quantity"] - db_order.fields["filled_qty"])*cur_price
            add_exposure = -1*add_exposure if db_order.fields["side"] == "S" else add_exposure
            new_total_exposure = new_total_exposure + add_exposure


    return cur_total_exposure, new_total_exposure, processed_orders
    

def check_for_sufficient_exposure( context, exposure_adjustment ):
    
    _, new_total_exposure, _ = get_current_order_exposure( context )
    new_total_exposure = new_total_exposure + exposure_adjustment
    
    return abs(new_total_exposure) < context[PARAM_MAX_NET_EXPOSURE]


def order_presubmission_checks( context, orders_by_symbol_dict ):
    
    logger = context[PARAM_LOGGER]
    broker = context[PARAM_BROKER_INTERFACE]
    positions = broker.get_open_positions()
    adj_max_exposure = context[PARAM_MAX_NET_EXPOSURE] * (1+context.get("trading.orders.presubmission_exposure_tolerance",0.01))
    
    symbols = []
    if not broker.is_market_open():
        for symbol, orders in orders_by_symbol_dict.items():
            for order in orders:
                order.fields['state'] = Order.S_REJECTED
                order.fields['state_detail'] = 'Market is closed'
                order.update()
            symbols.append(symbol)

        for symbol in symbols:
            del orders_by_symbol_dict[symbol]
            return orders_by_symbol_dict
   
    cur_total_exposure, new_total_exposure, processed_orders = get_current_order_exposure( context )

    price_cache = {}
    orders_by_side = { "S" : [], "B" : [] }
   
    # go through orders and adjust our expected exposure.    
    for symbol,orders in orders_by_symbol_dict.items():
        cur_price = get_symbol_mark(context,symbol)
        price_cache[symbol] = cur_price
    
        for order in orders:
            if order.fields["internal_order_id"] not in processed_orders:
                logger.warning("{} not in database during order_presubmission_checks".format(order.fields["internal_order_id"]))
                add_exposure = (order.fields["quantity"] - order.fields["filled_qty"])*cur_price
                add_exposure = -1*add_exposure if order.fields["side"] == "S" else add_exposure
                new_total_exposure = new_total_exposure + add_exposure
            
            orders_by_side[order.fields["side"]].append(order)

    # check if we need to remove orders
    rejected_orders = []
    excess_exposure = abs(new_total_exposure) - adj_max_exposure
    logger.info("order_presubmission_checks for exposure. Current: {0}, After Order Execution {1}".format(cur_total_exposure,round(new_total_exposure,2)))
      
    side_to_pare = None
    if excess_exposure > 0:
        logger.warn("excess exposure of {0}. pruning orders.".format(round(excess_exposure,2)))
        if new_total_exposure < 0:
            side_to_pare = "S"
        else: 
            side_to_pare = "B"
    elif context[PARAM_ALGO_TRADE_SIDE] == "long" and \
        new_total_exposure < (context[PARAM_MAX_NET_EXPOSURE]-adj_max_exposure):
        logger.warn("short exposure of {0} on long algo. pruning orders.".format(round(excess_exposure,2)))
        side_to_pare = "S"
        excess_exposure = abs(new_total_exposure)
    elif context[PARAM_ALGO_TRADE_SIDE] == "short" and \
        new_total_exposure > (adj_max_exposure-context[PARAM_MAX_NET_EXPOSURE]):
        logger.warn("long exposure of {0} on short algo. pruning orders.".format(round(excess_exposure,2)))
        side_to_pare = "L"
        excess_exposure = abs(new_total_exposure)
        
    if side_to_pare is not None:    
        # brute force, just cancel out the first orders
        for order in orders_by_side[side_to_pare]:
            order_exposure = (order.fields["quantity"] - order.fields["filled_qty"])*price_cache[order.fields["symbol"]]
            excess_exposure = excess_exposure - order_exposure
            rejected_orders.append(order)

            if excess_exposure <= 0:
                break

    for rejected_order in rejected_orders:
        logger.warn("rejected order {0} due to exceeding exposure limits".format(rejected_order.fields["internal_order_id"]))
        orders_by_symbol_dict[rejected_order.fields["symbol"]].remove(rejected_order)
        rejected_order.fields["state"] = Order.S_REJECTED
        rejected_order.fields["state_detail"] = "Order execution would result in exceeding max exposure limit of {0}".format(adj_max_exposure)
        rejected_order.update()
    
    # ensure there are no working orders open on this symbol
    cur_batch_order_ids = []
    for symbol,orders in orders_by_symbol_dict.items():
        for order in orders:
            cur_batch_order_ids.append(order.fields['internal_order_id'])
    
    working_orders = Order.get(context, state=Order.non_terminal_states, netted_order_id=None)  # netted orders are not street orders so ignore
    filtered_working_orders = []
    for order in working_orders:
        if order.fields['internal_order_id'] not in cur_batch_order_ids and \
            order.fields['state'] not in [Order.S_OPEN,Order.S_PENDING_CANCEL] and order.fields['order_type'] == Order.TYPE_MARKET:
            filtered_working_orders.append(order) 
    
    working_symbols = set([order.fields['symbol'] for order in filtered_working_orders])
    
    for symbol,orders in orders_by_symbol_dict.items():
        if symbol in working_symbols:
            logger.warn("rejected orders {} as the following orders with the same symbol ({}) are still working: {}"\
                        .format([order.fields['internal_order_id'] for order in orders],
                                symbol,
                                [order.fields['internal_order_id'] for order in filtered_working_orders if order.fields['symbol'] == symbol]))
            
            for order in orders:
                order.fields['state'] = Order.S_REJECTED
                order.fields['state_detail'] = "Working orders already exist on this symbol"
                order.update()
    
    for symbol in working_symbols:
        if symbol in orders_by_symbol_dict:
            del orders_by_symbol_dict[symbol]
    
    return orders_by_symbol_dict

"""
def eligible_for_trade_cross( ti_1, ti_2 ):
                                
    required_open_pos = ti_2.fields['quantity'] \
                            if ti_2.fields['side'] == "B" else -1*ti_2.fields['quantity']
                            
    return ti_1.get_open_position_quantity() == required_open_pos
"""

def expire_orders( context ):

    logger = context[PARAM_LOGGER]
    broker = context[PARAM_BROKER_INTERFACE]
    
    expire_orders = Order.get(context, state=Order.S_WORKING,
                                expire_at=(None,datetime.now()+timedelta(seconds=1)))
            
    if len(expire_orders) > 0:
        for order in expire_orders:
            try:
                if order.fields["broker_order_id"] is not None:
                    logger.info( "Expiring order {0}".format(order.fields["internal_order_id"]))
                    broker.cancel_order( order.fields["broker_order_id"] )
            except:
                logger.error("Unable to cancel internal order id {0}".format(order.fields["internal_order_id"]))
                logger.error(traceback.format_exc())
                
            order.fields["state"] = Order.S_PENDING_CANCEL
            order.update()


    
# execute orders in the database
def execute_orders( context ):
    
    order_state_changed = False
    logger = context[PARAM_LOGGER]
    broker = context[PARAM_BROKER_INTERFACE]
    
    open_orders = Order.get(context, state=Order.S_OPEN)

    disable_netting = context.get(PARAM_DISABLE_ORDER_NETTING,False)
    
    positions = context[PARAM_BROKER_INTERFACE].get_open_positions()
    
    # sort orders by symbol for netting purposes
    open_orders_by_symbol = {}
    for order in open_orders:
        symbol = order.fields['symbol']
        if symbol in open_orders_by_symbol.keys():
            open_orders_by_symbol[symbol].append(order)
        else:
            open_orders_by_symbol[symbol] = [order]
            
    def check_order_quantity( order, positions ):
        if order.fields['symbol'] in positions:
            trade_qty = order.fields['quantity'] if order.fields['side'] == "B" else \
                         -1*order.fields['quantity']
            pos_qty = positions[order.fields['symbol']].fields['quantity']
            
            if abs(trade_qty) > abs(pos_qty) and pos_qty != 0 and \
                np.sign(trade_qty) != np.sign(pos_qty):
                logger.info("Adjusting order quantity on {} from {} to {}, to avoid overbought/oversold"\
                                .format(order.fields['internal_order_id'],
                                        order.fields['quantity'],
                                        abs(pos_qty)))
                order.fields['quantity'] = abs(pos_qty) 
                                        
            
    if len(open_orders_by_symbol.keys()) > 0:
        approved_open_orders_by_symbol = order_presubmission_checks( context, open_orders_by_symbol )   
            
        # go through active orders
        for symbol, orders in approved_open_orders_by_symbol.items():
            if context.get("trading.test.live_test",False):
                if symbol != context["trading.test.test_symbol"]:
                    continue
            
            disable_netting_for_symbol = disable_netting
            try:
                if len(orders) > 0:
                    if len(orders) > 1:
                        for order in orders:
                            if order.fields["order_type"] != Order.TYPE_MARKET or \
                                order.fields["time_in_force"] != Order.TIF_DAY:
                                disable_netting_for_symbol = True
                    
                    if len(orders) == 1 or disable_netting_for_symbol:
                        for order in orders:
                            try:
                                if order.fields['execute_after'] is None or \
                                    order.fields['execute_after'] <= datetime.now():
                                    logger.info("Placing order: {0}".format(order.pprint()))
    
                                    if order.fields["order_class"] in [Order.CLASS_BRACKET, Order.CLASS_OTO]:
                                        instructions = TradeInstruction.get(context,trade_instruction_id=order.fields['trade_instruction_id'])
                                        assert(len(instructions) == 1)
                                        set_bracket_stop_prices( context, instructions[0], order )
                                    
                                    assert(order.fields["quantity"] > 0)
                                    
                                    check_order_quantity( order, positions )
                                    
                                    if context.get('trading.check_open_order_flag',False):
                                        order.fields['is_opening_order'] = get_current_is_opening_order_flag(context,order,positions,adj_positions=True)
                                    order.update()
                                    
                                    placed_order = broker.place_order(order)
                                    order_state_changed = True
                                    
                                    order.fields["state"] = placed_order.fields["state"]
                                    order.fields["broker_order_id"] = placed_order.fields["broker_order_id"]
                            except Exception as e:
                                logger.error("Unable to create broker order for market order {0}.".format(orders[0].fields["internal_order_id"]))
                                logger.error(traceback.format_exc())
                                
                                order.fields["state"] = Order.S_CANCELLED
                                order.fields["state_detail"] = traceback.format_exc()[:30]
                                            
                            order.update()
                            
                    else:   # trade netting
                        (orders, net_order) = create_netted_order(context, orders, positions)
        
                        logger.info("Placing netted order {0}".format(net_order.pprint()))
                        
                        # if the net order has a quantity (ie not a cross) then we place it on market
                        if net_order.fields["quantity"] > 0 and datetime.now() >= net_order.fields['execute_after']:
                            try:
                                check_order_quantity( net_order, positions )
                                
                                if context.get('trading.check_open_order_flag',False):
                                    order.fields['is_opening_order'] = get_current_is_opening_order_flag(context,order,positions,adj_positions=True)
                                
                                placed_order = broker.place_order( net_order )
                                order_state_changed = True

                                net_order.fields["broker_order_id"] = placed_order.fields["broker_order_id"]
                                net_order.fields["state"] = placed_order.fields["state"]
                                net_order.fields["quantity"] = placed_order.fields["quantity"]  # could be amended down based on broker availability
                            except Exception as e:
                                logger.error("Unable to create netted broker order for open orders with symbol {0}.".format(symbol)) 
                                logger.error(traceback.format_exc())
                                
                                for order in orders:
                                    if order.fields["state"] in Order.non_terminal_states:
                                        order.fields["state"] = Order.S_CANCELLED
                                        order.fields["state_detail"] = traceback.format_exc()    
                                
                                net_order.fields["state"] = Order.S_CANCELLED         
                                net_order.fields["state_detail"] = "Exception creating netted broker order"                   
                                
                        net_order.update()
                        Order.batch_update( context, orders )
            except:
                logger.error("Exception processing orders on symbol {0}.".format(symbol)) 
                logger.error(traceback.format_exc())
       
    return order_state_changed



def process_order_notification( context, broker_order, force_update = False, do_not_recur = False ):
    
    logger = context[PARAM_LOGGER]
    
    try:
        logger.info("Received order notification for order: {0}".format(broker_order.pprint()))
        
        if broker_order.fields["symbol"] is not None and \
            broker_order.fields["symbol"] in context.get("trading.symbol_filter",[]):
            logger.info("Symbol {} filtered. Skipping notification.".format(broker_order.fields["symbol"]))
            return
        
        db_orders = Order.get(context,broker_order_id=broker_order.fields["broker_order_id"])
        if len(db_orders) > 0:
            assert( len(db_orders) == 1 )
            db_order = db_orders[0]

            if db_order.fields["state"] in Order.terminal_states and \
                    db_order.fields["state"] != broker_order.fields["state"] and \
                    db_order.fields['state'] != Order.S_EXPIRED and \
                    (db_order.fields["state"] != Order.S_FILLED or \
                    db_order.fields["quantity"] == db_order.fields["filled_qty"]):
                # can occur if get_orders updates to terminal but intermediate notifications on the order have not been processed yet
                # ignore expired -- the order may have executed and we didnt know about it
                logger.info("Received notification on broker order {0} (state {1}), DB record has terminal state {2}. Order will be skipped."\
                               .format(broker_order.fields["broker_order_id"],broker_order.fields["state"],db_order.fields["state"]))
                return
            
            orig_state = db_order.fields["state"]

            field_updated = False
            skip_qty_price_updates = False
            broker_order.fields["internal_order_id"] = db_order.fields["internal_order_id"] # in case its not populated
            for field in Order.broker_populated_fields:
                if (broker_order.fields[field] != db_order.fields[field] and \
                    broker_order.fields[field] is not None) or \
                    (field == 'filled_qty' and broker_order.is_order_notification and \
                     broker_order.inc_fill_update): 
                    field_updated = True
                    
                    # Check for incremental partial fills (eg. TDA)
                    if field == 'filled_qty' and broker_order.is_order_notification and \
                        broker_order.inc_fill_update and \
                        broker_order.remaining_exec_qty is not None and \
                        broker_order.fields['filled_qty'] < broker_order.fields['quantity']:

                        new_fill_qty = db_order.fields['filled_qty'] + broker_order.fields['filled_qty']
                        
                        if db_order.fields['avg_fill_price'] is not None:
                            new_avg_price = (db_order.fields['filled_qty'] * db_order.fields['avg_fill_price'] + \
                                             broker_order.fields['filled_qty'] * broker_order.fields['avg_fill_price']) / \
                                             new_fill_qty
                        else:
                            new_avg_price = broker_order.fields['avg_fill_price']
                        
                        if (db_order.fields['quantity'] - new_fill_qty) != broker_order.remaining_exec_qty:
                            logger.info("Partial fill order notification quantities do not reconcile. Broker may not support partial fill notifications. "\
                                         "Quantity: {}, Current fill_qty: {}, New fill qty: {}, broker qty remaining: {}"\
                                         .format(db_order.fields['quantity'],
                                                 db_order.fields['filled_qty'],
                                                 new_fill_qty,
                                                 broker_order.remaining_exec_qty))
                            
                        db_order.fields['filled_qty'] = new_fill_qty
                        db_order.fields['avg_fill_price'] = new_avg_price
                        
                        logger.debug("Added fill. New filled_qty: {}, New Avg Price: {}. Remaining (according to broker): {}"\
                                        .format(new_fill_qty, new_avg_price, broker_order.remaining_exec_qty))
                        skip_qty_price_updates = True
                    else:
                        if (field not in ['filled_qty','avg_fill_price'] or \
                            not skip_qty_price_updates) and field != 'last_update_date' and \
                            broker_order.fields[field] is not None:
                            logger.debug("Updated field {0} from {1} to {2}".format(field,db_order.fields[field],broker_order.fields[field]))    
                            db_order.fields[field] = broker_order.fields[field]
                            
                            if field == 'avg_fill_price' and \
                                context.get("trading.log_prices_as_filled",False):
                                try:
                                    cur_price = get_symbol_mark(context, broker_order.fields["symbol"])
                                    logger.info("Price comparison on fill, Symbol: {}, Fill Price: {}, Current Market Price: {}"\
                                                        .format(broker_order.fields["symbol"],broker_order.fields[field],cur_price))
                                except:
                                    logger.warning("Failed to log prices for comparison")
                                    logger.warning(traceback.format_exc())
                                    

            
            if field_updated:
                order_updates = [db_order]

                if db_order.fields["is_net_order"] == 1 and db_order.fields["state"] != orig_state and \
                    orig_state in Order.non_terminal_states and db_order.fields["state"] in Order.terminal_states:
                    logger.debug("Netted order in terminal state. Allocating fills to parents.")
                    parent_orders = Order.get(context,netted_order_id=db_order.fields['internal_order_id'])
                    assert(len(parent_orders) > 0)
                    
                    filled_qtys_dict = db_order.allocate_parent_order_fills( context, db_order.fields["filled_qty"], parent_orders )
                    
                    logger.debug( "Fill allocations are: {0}".format(pprint.pformat(filled_qtys_dict)) )
                    
                    for parent_order in parent_orders:
                        filled_qty = filled_qtys_dict[parent_order.fields["internal_order_id"]]
                        parent_order.apply_netted_order_fill( db_order, filled_qty )
                        order_updates.append( parent_order )
                        
                        if not parent_order.is_fully_filled() and \
                            parent_order.fields['filled_qty'] == 0:
                            parent_order.fields['netted_order_id'] = None
                            parent_order.fields['state'] = Order.S_OPEN

                Order.batch_update(context,order_updates)
            
                if db_order.fields["state"] == Order.S_FILLED and \
                    db_order.fields["order_class"] in [Order.CLASS_OTO, Order.CLASS_BRACKET]:
                    
                    close_bracket_orders = Order.get(context,
                                                         trade_instruction_id=db_order.fields["trade_instruction_id"],
                                                         order_type=[Order.TYPE_STOP,Order.TYPE_LIMIT],
                                                         state=Order.S_WORKING)
                    assert(len(close_bracket_orders) <= 2)
                    
                    if len(close_bracket_orders) > 0:
                        for bracket_child_order in close_bracket_orders:
                            check_bracket_order_pricing( context, db_order, bracket_child_order )

        else:
            if match_existing_order( context, broker_order ):
                if do_not_recur:
                    logger.warning("Matched incoming broker order {} but unable to recur to reprocess".format(broker_order.fields["broker_order_id"]))
                else:
                    process_order_notification( context, broker_order, force_update, do_not_recur=True )
            elif not handle_unknown_order( context, broker_order ):
                logger.error('Invalid order notification received with broker_order_id: {0}'.format(broker_order.fields['broker_order_id']))
                    
    except:
        logger.error('Exception processing order notification with internal_order_id: {0}'.format(broker_order.fields['internal_order_id']))
        logger.error(traceback.format_exc())

    return



