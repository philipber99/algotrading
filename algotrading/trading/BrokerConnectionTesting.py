import os, sys, pprint, traceback, logging, json
from datetime import datetime, timedelta, time
import time as ptime

from algotrading.ContextFields import *
from algotrading.Common import load_configuration, import_module_attr
from algotrading.trading.Objects import Order
from algotrading.trading.TestBrokerInterface import TestBrokerInterface
from algotrading.ContextFields import *
from algotrading.refdata.ServerStreamingPrices import ServerStreamingPrices 


algo_home = os.getenv('ALGO_HOME')
if algo_home is None:
    print("ALGO_HOME environment variable not set")
    sys.exit(2)

if algo_home[-1] != "/":
    algo_home = algo_home + "/"

context = load_configuration( {}, algo_home, "prod-test" )
logger = logging.getLogger(__name__)
logger.setLevel( logging.DEBUG )
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
    
context[PARAM_LOGGER] = logger

context[PARAM_APP_NAME] = "TEST"
test_symbol = "NVDA"

refdata = import_module_attr(context[PARAM_REFDATA_PROVIDER_CLASS])( context )

context[PARAM_REPORTING_PATH] = os.path.join(algo_home,"logs")

#broker = TDABrokerInterface( context )
broker = TestBrokerInterface(context) 

#pricing = TDAStreamingPrices(context,[test_symbol])
ss_prices = ServerStreamingPrices(context,[])
context[PARAM_BROKER_INTERFACE] = broker

global_order_log = []
close_stop_margin = 0.10
wide_stop_margin = 5
order_id = 0 

def get_order( side, symbol = test_symbol, quantity = 1, limit_price = None, stop_price = None, time_in_force = Order.TIF_DAY,
                    bkt_stop_loss = None, bkt_stop_gain = None, open_order = 1 ):
    global context, order_id
    
    order_type = Order.TYPE_MARKET
    if limit_price is not None:
        order_type = Order.TYPE_LIMIT
    elif stop_price is not None:
        order_type = Order.TYPE_STOP
        
    if bkt_stop_loss is not None and \
        bkt_stop_gain is not None:
        order_class = Order.CLASS_BRACKET
    elif bkt_stop_loss is not None:
        order_class = Order.CLASS_OTO
    else:
        order_class = None
    
    order_id = order_id + 1 
    
    order_obj = Order(context)
    order_obj.fields.update( {
                                     "internal_order_id" : order_id,
                                     "algo_id" : context[PARAM_APP_NAME],
                                     "side" : side,
                                     "quantity" : quantity,
                                     "state" : Order.S_OPEN,
                                     "symbol" : symbol,
                                     "limit_price" : limit_price if limit_price is not None else None,
                                     "order_type" : order_type,
                                     "time_in_force" : Order.TIF_DAY,
                                     "execute_after" : None,
                                     "stop_price" : stop_price if stop_price is not None else None,
                                     "order_class" : order_class,
                                     "bkt_stop_loss" : bkt_stop_loss if bkt_stop_loss is not None else None,
                                     "bkt_stop_gain" : bkt_stop_gain if bkt_stop_gain is not None else None,
                                     "is_opening_order" : open_order
                                    } )

    return order_obj

def get_responses( timeout=3):
    global global_order_log
    
    order_updates = []
    
    while True:
        order_update = broker.retrieve_order_update(block=True, timeout=timeout)
        if order_update is None:
            break
        else:
            order_updates.append(order_update)
            global_order_log.append(order_update.fields)
    
    return order_updates


def check_for_response( state, blocking = True, extra_orders = False ):

    orders_found = []
    extras_found = []
        
    while True:
        resp_orders = get_responses()
        for order in resp_orders:
            if order.fields['state'] == state:
                orders_found.append(order)
            else:
                extras_found.append(order)
                
        if not blocking or len(orders_found) > 0:
            break
    
    if len(orders_found) == 0:
        print(f"FAILED to find state {state}")
        return None
            
    if extra_orders:
        return (orders_found, extras_found)
            
    return orders_found


def wait_for_order_fill( blocking = True ):
    
    resp_orders = check_for_response( Order.S_FILLED, blocking )

    return resp_orders

def print_orders( orders ):
    for order in orders:
        pprint.pprint(order.fields)

def print_positions( positions ):
    for k,v in positions:
        print(f"{k}: {pprint.pformat(v.fields)}")


ptime.sleep(2)

print(f"Market is open: {broker.is_market_open()}")

# TC1: Market Orders
# check positions as well
def TC1():
    try:
        # place the order
        order = get_order( symbol=test_symbol, side="B", quantity=1, open_order=1 )
        order.fields['execute_after'] = datetime.now() + timedelta(minutes=1)
        broker.place_order(order)
        
        # check the order is in the list
        order_list = broker.get_orders( from_dt = datetime.now() - timedelta(seconds=60), to_dt = datetime.now() )
        #print_orders(order_list)
    
        # wait for execution        
        wait_for_order_fill()
        
        # should see positions
        print("\n")
        print_positions(broker.get_open_positions())
        
        # close the order
        ptime.sleep(5)
        order = get_order( symbol=test_symbol, side="S", quantity=1, open_order=0 )
        broker.place_order(order)
        
        # check response orders        
        wait_for_order_fill()
    
        # should see no position
        print("\n")
        print_positions(broker.get_open_positions())
    except:
        traceback.print_exc()


# TC2: Limit Orders
def TC2():
    try:
        # place the order
        bidask = ss_prices.get_bid_ask([test_symbol])
        
        order = get_order( symbol=test_symbol, side="B", limit_price=bidask.iloc[0].ask-0.15, quantity=1, open_order=1 )
        src_order = broker.place_order(order)
        
        # wait for execution        
        wait_for_order_fill()
    
        # close the order
        ptime.sleep(5)
        order = get_order( symbol=test_symbol, side="S", quantity=1, open_order=0 )
        broker.place_order(order)
        
        # check response orders        
        wait_for_order_fill()
    
        # should see no position
        print("\n")
        print_positions(broker.get_open_positions())
    except:
        traceback.print_exc()


# TC3: Stop Orders
def TC3():
    try:
        # place the order
        bidask = ss_prices.get_bid_ask([test_symbol])
        
        order = get_order( symbol=test_symbol, side="B", stop_price=bidask.iloc[0].ask+0.15, quantity=1, open_order=1 )
        broker.place_order(order)
        
        # wait for execution        
        wait_for_order_fill()
    
        # close the order
        ptime.sleep(5)
        order = get_order( symbol=test_symbol, side="S", quantity=1, open_order=0 )
        broker.place_order(order)
        
        # check response orders        
        wait_for_order_fill()
    
        # should see no position
        print("\n")
        print_positions(broker.get_open_positions())
    except:
        traceback.print_exc()


    

# TC13 + 14: Cancel orders
def TC13_14():
    try:
        # place the order
        bidask = ss_prices.get_bid_ask([test_symbol])
        
        order = get_order( symbol=test_symbol, side="B", stop_price=bidask.iloc[0].ask+1, quantity=1, open_order=1 )
        broker.place_order(order)

        # get the broker id
        broker_order_id = None
        resp_orders = check_for_response(Order.S_WORKING)
        if resp_orders is None:
            print("FAIL. No broker order id.")
        
        ptime.sleep(3)

        # cancel the order
        broker.cancel_order(resp_orders[0].fields['broker_order_id'])
        check_for_response(Order.S_CANCELLED)
        
        # cancel the order again
        try:
            broker.cancel_order(resp_orders[0].fields['broker_order_id'])
        except:
            # No exception is expected
            traceback.print_exc()
        
    except:
        traceback.print_exc()


# TC12: bracket order with stop/limit execution
def TC12():
    try:
        bidask = ss_prices.get_bid_ask([test_symbol])
        stopgain = bidask.loc[test_symbol]['bid'] + 0.10
        stoploss = bidask.loc[test_symbol]['bid'] - 0.10
        
        # place the order
        order = get_order( symbol=test_symbol, side="B", quantity=1, open_order=1, bkt_stop_loss = stoploss, bkt_stop_gain = stopgain )
        broker.place_order(order)
        
        # wait for execution        
        resp_orders,resp_extras  = check_for_response(Order.S_FILLED, extra_orders = True)
        
        # wait for bracket execution        
        stop_orders,stop_extras = check_for_response(Order.S_FILLED, extra_orders = True)
        
        # check working orders -- should be none
        working_orders = broker.get_orders( from_dt = datetime.combine(datetime.now().date(),time.min),
                                        to_dt = datetime.now(),
                                        status = Order.S_WORKING )
        if len(working_orders) > 0:
            print("Working orders left -- not expected")
            
        # should see no position
        print("\n")
        print(broker.get_open_positions())
    except:
        traceback.print_exc()


def create_double_stop_orders( stop_margin = close_stop_margin, bracket = True ):
    bidask = ss_prices.get_bid_ask([test_symbol])
    b_stop_price = bidask.loc[test_symbol]['ask'] + stop_margin
    b_stop_stopgain = bidask.loc[test_symbol]['ask'] + 2*stop_margin
    b_stop_stoploss = bidask.loc[test_symbol]['ask']
    
    s_stop_price = bidask.loc[test_symbol]['bid'] - stop_margin
    s_stop_stopgain = bidask.loc[test_symbol]['bid'] - 2*stop_margin
    s_stop_stoploss = bidask.loc[test_symbol]['bid']

    if bracket:    
        b_order = get_order( symbol=test_symbol, side="B", quantity=1, stop_price = b_stop_price, open_order=1, 
                             bkt_stop_loss = b_stop_stoploss, bkt_stop_gain = b_stop_stopgain )
        
        s_order = get_order( symbol=test_symbol, side="S", quantity=1, stop_price = s_stop_price, open_order=1, 
                             bkt_stop_loss = s_stop_stoploss, bkt_stop_gain = s_stop_stopgain )
    else:
        b_order = get_order( symbol=test_symbol, side="B", quantity=1, stop_price = b_stop_price, open_order=1 )
        
        s_order = get_order( symbol=test_symbol, side="S", quantity=1, stop_price = s_stop_price, open_order=1 )
        
    print("Placing buy order")
    broker.place_order(b_order)
    b_resp_orders = check_for_response(Order.S_WORKING)

    print("Placing sell order")
    broker.place_order(s_order)
    s_resp_orders = check_for_response(Order.S_WORKING)
    
    return b_resp_orders[0],s_resp_orders[0]


# TC4: Dual stop orders
def TC4():
    global close_stop_margin
    
    try:
        b_order, s_order = create_double_stop_orders( close_stop_margin, bracket=True )
        
        # wait for execution of either side        
        orders = wait_for_order_fill()
        
        if orders[0].fields['side'] == 'B':
            broker.cancel_order(s_order.fields['broker_order_id'])
            close_side = 'S'
        else:
            broker.cancel_order(b_order.fields['broker_order_id'])
            close_side = 'B'
        
        check_for_response(Order.S_CANCELLED)
        
        # wait for stop gain/loss to execute
        check_for_response(Order.S_FILLED)
    except:
        traceback.print_exc()


# TC5: ToOpen/ToClose Flag Adjustments
# make sure amendment of both source orders and bracket stop loss/gain orders are amended and work correctly
# ensure bracket orders execute AFTER the is_open amendment to see if the other side still cancels
def TC5():
    global close_stop_margin, wide_stop_margin
    
    try:
        b_order, s_order = create_double_stop_orders( close_stop_margin, bracket=True )
        
        # wait for execution of either side        
        found_orders,extra_orders = check_for_response(Order.S_FILLED,extra_orders=True)

        # adjust to the other side of flag (eg, if long 1, adj pos to short 1)        
        if found_orders[0].fields['side'] == 'B':
            target_order = b_order
            broker.cancel_order(s_order.fields['broker_order_id'])

            adj_order1 = get_order( symbol=test_symbol, side="S", quantity=1, open_order=0 )    # need 2 orders because need to flatten then open pos
            adj_order2 = get_order( symbol=test_symbol, side="S", quantity=1, open_order=1 )
        else:
            target_order = s_order
            broker.cancel_order(b_order.fields['broker_order_id'])
            
            adj_order1 = get_order( symbol=test_symbol, side="B", quantity=1, open_order=0 )
            adj_order2 = get_order( symbol=test_symbol, side="B", quantity=1, open_order=1 )
            
        broker.place_order(adj_order1)
        check_for_response(Order.S_FILLED)
        broker.place_order(adj_order2)
        check_for_response(Order.S_FILLED)

        # IMPORTANT QUESTION: WILL THE STOPGAIN/LOSS FILL IF THE TO_OPEN FLAGS ARE NOW WRONG.
        # PAUSE HERE TO CHECK. OTHERWISE PROCEED TO AMENDMENT
        
        # amend to_open/to_close
        working_orders = broker.get_orders( from_dt = datetime.combine(datetime.now().date(),time.min),
                                        to_dt = datetime.now(),
                                        status = Order.S_WORKING )
        test_pos = broker.get_open_positions()[test_symbol].fields['quantity']
        
        for order in working_orders:
            is_open = test_pos >= 0 and order.fields['side'] == 'B' or \
                        test_pos <= 0 and order.fields['side'] == 'S'
            
            if is_open != order.fields['is_opening_order']:
                print("amending is_open flag on order")
                order.fields['is_opening_order'] = is_open
                broker.amend_order(order)
        
        resp_orders = get_responses()
        
        # wait for stopgain/loss of executed double-stop to fill
        check_for_response(Order.S_FILLED)

        working_orders = broker.get_orders( from_dt = datetime.combine(datetime.now().date(),time.min),
                                        to_dt = datetime.now(),
                                        status = Order.S_WORKING )
        """
        # should be no open orders
        for order in working_orders:
            try:
                broker.cancel_order(order.fields['broker_order_id'])
            except:
                traceback.print_exc()
        """
    except:
        traceback.print_exc()

def place_bracket( side, order_type, exec_margin, stop_gain_margin, open_order = 1 ):
    bidask = ss_prices.get_bid_ask([test_symbol])
    
    open_side = 'ask' if side == 'B' else 'bid'
    close_side = 'bid' if open_side == 'ask' else 'ask'
    
    if order_type == 'market':
        stop_gain = bidask.iloc[0][close_side] + stop_gain_margin if open_side == 'ask' else bidask.iloc[0][close_side] - stop_gain_margin 
        stop_loss = bidask.iloc[0][close_side] - stop_gain_margin if open_side == 'ask' else bidask.iloc[0][close_side] + stop_gain_margin 
    else:
        if order_type == 'stop':
            stop_price = bidask.iloc[0][open_side] + exec_margin if open_side == 'ask' else bidask.iloc[0][open_side]-exec_margin
        else:
            stop_price = bidask.iloc[0][open_side] - exec_margin if open_side == 'ask' else bidask.iloc[0][open_side]+exec_margin
        
        stop_gain = stop_price + stop_gain_margin if open_side == 'ask' else stop_price - stop_gain_margin 
        stop_loss = stop_price - stop_gain_margin if open_side == 'ask' else stop_price + stop_gain_margin
    
    if order_type == 'stop':
        order = get_order( symbol=test_symbol, side=side, stop_price=stop_price, quantity=1, 
                           bkt_stop_loss = stop_loss, bkt_stop_gain = stop_gain, open_order =open_order)
    elif order_type == 'limit':
        order = get_order( symbol=test_symbol, side=side, limit_price=stop_price, quantity=1, 
                           bkt_stop_loss = stop_loss, bkt_stop_gain = stop_gain, open_order=open_order)
    elif order_type == 'market':
        order = get_order( symbol=test_symbol, side=side, quantity=1, 
                           bkt_stop_loss = stop_loss, bkt_stop_gain = stop_gain, open_order=open_order)
    else:
        print("unknown order type")
        return
    
    broker.place_order(order)

place_bracket( "S", 'market', 0.10, 2.00 )

place_bracket( "B", 'market', 0.10, 0.20)


found_orders,extra_orders = check_for_response(Order.S_FILLED,extra_orders=True)



"""
1) Bracket order executed. -1 position, OCO buy/to_close outstanding.
place_bracket( "S", 'market', 2.00, 4.00 )
- buy market/limit/stop order to_open fails (no boxed positions)
- buy market/limit/stop order to_close fails (may result in overbought)

2) placing second bracket, opposite direction.
place_bracket( "B", 'market', 2.00, 4.00 )
- order executes (position becomes 0) but the OCO legs immediately fail.
- sell stop to_close fails (may result in overbought)
- sell stop to_open works (replacement bkt leg)
- sell limit to_open works (replacement bkt leg)

3) Wait for execution of TO_CLOSE legs of (1) without placing orders to cover (2). Position is flat.
- The TO_CLOSE orders will execute even though the flag isn't correct in respect to the current position.

"""


# PASSED
#TC1()

# PASSED
#TC2()

# PASSED
#TC3()

# PASSED
#TC13_14()

# PASSED
# TC12()

# PASSED
#TC4()

TC4()



print("END")




""" 
sys.exit(0)

try:
    price = pricing.get_streaming_last_price("MSFT")
    order = get_order( "B", bkt_stop_loss=price - 0.04, bkt_stop_gain=price+0.04,open_order=1)
    broker.place_order(order)
except:
    traceback.print_exc()

while True:
    ptime.sleep(2)
    
o = broker.get_orders(from_dt=datetime.now(),to_dt=datetime.now())
for order in o:
    if order.fields["state"] == Order.S_WORKING and \
        order.fields["side"] == "S":
        pprint.pprint(order.fields)
        break

a_order = get_order( "S", stop_price = 195, open_order=0 )
a_order.fields["broker_order_id"] = order.fields["broker_order_id"]
broker.amend_order(a_order)

ptime.sleep(2)


try:
    order = get_order( "B", limit_price = 259.50, open_order=1 )
    broker.place_order(order)
except:
    traceback.print_exc()

ptime.sleep(2)

"""
    
"""
PART A) ORDER PLACEMENT / CANCEL, NO EXECUTION

0A) LOGIC FOR OVERBUYING / OVERSELLING

PASSED

Test by overselling / buying after execution a trade

0B) Amend stop order on an OTO

Place OTO, try to amend order.


1) Place LIMIT order
2) Cancel LIMIT order
order = place_order( "B", limit_price = 200 )
print(pprint.pformat(order.fields))
pprint.pprint(broker.get_orders(from_dt=datetime.now(),to_dt=datetime.now()))

PASSED


3) Place STOP order
4) Cancel STOP order

order = place_order( "S", stop_price = 231 )
o = broker.get_orders(from_dt=datetime.now(),to_dt=datetime.now())
for order in o:
    if order.fields["stop_price"] == 231 and \
        order.fields["status"] == Order.S_WORKING:
            use_order = order
            break

if use_order != None:
    use_order.fields["stop_price"] = 221
    broker.amend_order(use_order)
    
PASSED


5) PLACE STOP order and amend STOP price.

PASSED


6) Place OTO: S 1 MSFT LIMIT @300, TRIGGER B 1 MSFT STOP 310.
7) Cancel OTO.

PASSED

8) Place OCO: B 1 MSFT LIMIT 200, TRIGGER OCO { S 1 MSFT STOP 180, S 1 MSFT LIMIT 300 }
9) Cancel OCO

PASSED


PART B) ORDER EXECUTION

1) Place MARKET order to buy, then to sell (close)

PASSED

2) Place MARKET order to sell, then to buy (close)

PASSED

3) Place MARKET order to buy 10, sell 20, buy 10 (close). Need to cross 0 threshold in case it has same behavior as Alpaca.

PASSED

4) Place STOP order, have it execute. Close position after checking everything.

PASSED

a) Amend the STOP order before execution

PASSED


5) Place LIMIT order, have it execute. Close position after checking everything.

PASSED

6) Place OTO: B 1 MSFT MARKET, TRIGGER S 1 MSFT STOP XX.

PASSED


7) Cancel remaining leg.

PASSED


8) Place OCO B 1 MSFT MARKET TRIGGER OCO S 1 MSFT STOP, S 1 MSFT LIMIT
- set a tight bracket so it will execute


PASSED


"""








