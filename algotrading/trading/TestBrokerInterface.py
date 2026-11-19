import random, pprint, pytz, copy, traceback, os, pickle, atexit, inspect
import queue, threading

import pandas as pd
import time as ptime
from datetime import datetime, timedelta, time

from dateutil.tz import tzlocal
import numpy as np

from algotrading.ContextFields import *

from .BrokerInterface import BrokerInterface
from .Objects import Order, Position
from algotrading.refdata.MarketData import prices_sql, get_trading_calendar
from algotrading.trading.Pricing import get_price_data, get_symbol_mark
from algotrading.refdata.ReferenceData import RefData
from algotrading.trading.Common import get_current_is_opening_order_flag

alpaca_date_format = "%Y-%m-%d"
iso_datetime_format = '%Y-%m-%dT%H:%M:%S.%fZ'


class TestBrokerInterface(BrokerInterface):
            
    PARAM_TEST_BROKER_FILL_PROBABILIY = "TestBrokerInterface.FillProbability"
    
    fill_probability = {
                        #Order.S_FILLED : (0,1.00)
                        Order.S_WORKING : (0,1.00),     # partial fill
                        #Order.S_PENDING_CANCEL : (0,1.00),
                        #Order.S_CANCELLED : 0,    # always use PENDING_CANCEL
                        #Order.S_REJECTED : 0,
                        #Order.S_EXPIRED : 0
                    }
        
    def __init__(self, context, streaming = True):
        
        self.logger = context[PARAM_LOGGER]
        self.reporting_path = context[PARAM_REPORTING_PATH]
        
        self.event_q = queue.Queue()    # for sending events back to the trading manager
        self.context = context
        self.writer_lock = threading.Lock()  
        self.current_orders = {}        # maps broker order id to order
        
        self.last_order_state = {}      # stores last order state, used to update orders_by_state
        self.orders_by_state = {}       # stores stores by state, but only updated as part of dispatch
        self.shutdown_writer_thread = False
        
        for order_state in Order.terminal_states+Order.non_terminal_states:
            self.orders_by_state[order_state] = []
        
        self.todays_market_hours = None
        
        self.fill_all_orders = context.get("trading.test_broker.fully_fill_orders",True)
        
        if context.get('trading.test_broker.disable_dispatch',False):
            self.logger.warning("Test broker has event dispatching disabled. No test broker message will be received")
        
        fname = self.context[PARAM_REPORTING_PATH] + "TestBroker.pickle"
        
        if os.path.exists(fname):
            with open(fname,'rb') as f:
                new_orders = pickle.load(f)
                
                if context.get('trading.test_broker.daily_reset',False):
                    earliest_dt = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
                    for order_id,order in new_orders.items():
                        if order.fields['creation_date'] >= earliest_dt:
                            self.current_orders[order_id] = order
                else:
                    self.current_orders = new_orders
                                
                for order_id,order in self.current_orders.items():
                    self.orders_by_state[order.fields['state']].append(order_id)
                    self.last_order_state[order_id] = order.fields['state']
    
                # add context object back for this session
                for k,order in self.current_orders.items():
                    order.context = context
        
        self.update_open_orders()

        threading.Thread(target=self.broker_writer_thread, args=()).start()
    
    
    def broker_writer_thread(self):
        next_write_delay = self.context.get("trading.test_broker.data_file_write_delay",5)

        while not self.shutdown_writer_thread:
            self.write_broker_data()
            ptime.sleep(next_write_delay)
            
        
        
    def write_broker_data(self):
        fname = self.context[PARAM_REPORTING_PATH] + "TestBroker.pickle"
        
        with self.writer_lock:
            working_current_orders = copy.deepcopy(self.current_orders)
        
        with open(fname,'wb') as f:
            
            # remove context object from orders to prevent unnecessary serialization
            for k,order in working_current_orders.items():
                order.context = None
    
                for exec_order in order.exec_legs:
                    exec_order.context = None
            
            pickle.dump(working_current_orders,f)        
        
        
    def close(self):
        self.write_broker_data()
        self.shutdown_writer_thread = True

    
    @staticmethod        
    def linear_hl__get_exec_time( row, s_price ):
        
        if row.low == row.high:
            return 0
        
        if row.open > row.close:
            t = (s_price-row.high) / (row.low - row.high)
        else:
            t = (s_price-row.low) / (row.high - row.low)
    
        if t < 0:
            return 0
        elif t > 1:
            return 1
        
        return t if not np.isnan(t) else 0 

        

    def _check_for_execution( self, order, order_fills, row, price ):
        
        if row is None:
            low_price = price
            high_price = price
        else:
            low_price = row.low
            high_price = row.high
            
        def get_execution_dt( price ):
            if row is None:
                return datetime.now()
            return row.name + timedelta(seconds=int(59*TestBrokerInterface.linear_hl__get_exec_time(row,price)))
        
        new_fill = None            
        if order.fields["side"] == "B":
            if order.fields["order_type"] == Order.TYPE_STOP_LIMIT and \
                    not order.stop_limit_activate:
                if high_price >= order.fields["stop_price"]:
                    order.stop_limit_activate = True

            if order.fields["order_type"] == Order.TYPE_LIMIT or \
                order.fields["order_type"] == Order.TYPE_STOP_LIMIT and \
                order.stop_limit_activate:
                if low_price <= order.fields["limit_price"]:
                    new_fill = [order.fields["broker_order_id"],
                                        None,
                                        order.fields["limit_price"] if price is None else price]
                    
            elif order.fields["order_type"] == Order.TYPE_STOP:
                if high_price >= order.fields["stop_price"]:
                    new_fill = [order.fields["broker_order_id"],
                                        None,
                                        order.fields["stop_price"] if price is None else price]
        else:   # side is S
            if order.fields["order_type"] == Order.TYPE_STOP_LIMIT and \
                    not order.stop_limit_activate:
                if low_price <= order.fields["stop_price"]:
                    order.stop_limit_activate = True

            if order.fields["order_type"] == Order.TYPE_LIMIT or \
                order.fields["order_type"] == Order.TYPE_STOP_LIMIT and \
                order.stop_limit_activate:
                if high_price >= order.fields["limit_price"]:
                    new_fill = [order.fields["broker_order_id"],
                                        None,
                                        order.fields["limit_price"] if price is None else price]
                    
            elif order.fields["order_type"] == Order.TYPE_STOP:
                if low_price <= order.fields["stop_price"]:
                    new_fill = [order.fields["broker_order_id"],
                                        None,
                                        order.fields["stop_price"] if price is None else price]
        
        if new_fill is not None:
            new_fill[1] = get_execution_dt(new_fill[2])
            if new_fill[1] < order.fields['creation_date']:
                return False
            
            order_fills.append(tuple(new_fill))
            return True
        
        return False   
                
                
    def update_open_orders(self, dispatch=False,iter_count=0):
        
        # used to determine if we need to repeat (ie. new stop/limits created)
        pre_update_order_ids = copy.copy(self.orders_by_state[Order.S_WORKING])
        
        # go through working orders
        order_fills = []
        for order_id in self.orders_by_state[Order.S_WORKING]: #+self.orders_by_state[Order.S_PENDING_CANCEL]:
            order = self.current_orders[order_id] 
            try:
                if order.fields['state'] != Order.S_WORKING:
                    continue    # state may have changed
                
                # pending cancels get cancelled
                #if order.fields["state"] == Order.S_PENDING_CANCEL:
                #    order.fields["state"] = Order.S_CANCELLED
                # check working orders
                #elif order.fields["state"] == Order.S_WORKING:
                # if its from a previous day and wasnt GTC then cancel
                if order.fields["time_in_force"] != Order.TIF_GTC and \
                    order.fields["creation_date"].date() < datetime.now().date():
                    order.fields["state"] = Order.S_CANCELLED
                else: # check for order execution
                    if self.context.get("trading.test_broker.live_price_order_updates",False):
                        cur_price = self.get_quote(order.fields["symbol"],order.fields["side"])
                        self._check_for_execution( order, order_fills, price=cur_price, row=None )
                    else:
                        start_date = datetime.combine(order.fields["creation_date"].date(), 
                                    time(order.fields["creation_date"].hour, order.fields["creation_date"].minute)) 
                        
                        prices_df = prices_sql( self.context, order.fields["symbol"], start_date, 
                                                datetime.now(), RefData.BARS_1MIN )
                        trade_cal_df = get_trading_calendar( self.context, start_date, datetime.now() ).set_index("date")
                        
                        for idx,row in prices_df.iterrows():
                            if idx >= trade_cal_df.at[idx.date(),"open"] and \
                                idx <= trade_cal_df.at[idx.date(),"close"]:
                                if self._check_for_execution( order, order_fills, price = None, row = row ):
                                    break
            except:
                self.logger.error("Unable to update broker order id {}",order.fields["broker_order_id"])
                self.logger.error(traceback.format_exc())

        order_fills_df = pd.DataFrame(order_fills,columns=["broker_order_id","time","price"]).set_index("time").sort_index()
            
        # fill the open orders
        for idx,row in order_fills_df.iterrows():
            try:
                internal_order = self.current_orders[row.broker_order_id]
                # check if order is still in the working state (OCO processing may close out other orders)
                if internal_order.fields["state"] == Order.S_WORKING:
                    try:
                        created_order = self._create_response_order( internal_order, fully_fill_order = True, 
                                                                     dispatch_order = dispatch, override_price = row.price,
                                                                     override_exec_dt = idx.to_pydatetime() )
 
                    except ValueError as e:
                        broker_order = internal_order
                        broker_order.fields['state'] = Order.S_REJECTED
                        broker_order.fields['state_detail'] = str(e)
                        self._dispatch_broker_order(broker_order)
                        
            except:
                self.logger.error("Unable to fill broker order id {}",row.broker_order_id)
                self.logger.error(traceback.format_exc())
        
        repeated_update = False
        new_order_ids = list(set(self.orders_by_state[Order.S_WORKING]) - set(pre_update_order_ids))
        for order_id in new_order_ids:
            if self.current_orders[order_id].fields['order_type'] in [Order.TYPE_STOP,Order.TYPE_LIMIT]:
                repeated_update = True
        
        if repeated_update:
            if iter_count > self.context.get("trading.test_broker.max_update_orders_iterations",5):
                self.logger.error("Maximum iterations for test broker update_open_orders reached. Iterations will stop.")
            else:
                self.update_open_orders(dispatch,iter_count+1)
    
        
    def is_shortable(self, symbols):
        if type(symbols) is list:
            return {symbol: True for symbol in symbols}
        else:
            return True      
        
    def _new_order(self):
        new_order = Order(self.context)
        new_order.oco_order_id = None
        new_order.exec_legs = []
        new_order.remaining_exec_qty = 0
        new_order.stop_limit_activate = False
        
        return new_order
    
    def _get_total_filled_qty(self, order):
        if len(order.exec_legs) > 0:
            return int(np.sum([exec_leg.fields["filled_qty"] for exec_leg in order.exec_legs]))
            
        return order.fields["filled_qty"]
        
    def _get_random_state(self):
        rand = random.randint(0,100) / 100 - 0.001
        
        for k,v in self.fill_probability.items():
            start,stop = v
            
            if rand >= start and rand < stop:
                return k
            
        return Order.S_FILLED
    
    def _get_determined_next_state(self, current_state):
        state_map = {
                        Order.S_OPEN : Order.S_WORKING,
                        Order.S_WORKING : Order.S_FILLED
                    }
        return state_map[current_state]
        
        
    # for testing only
    def _change_order_state(self, broker_order_id, state, state_detail = ""):
        
        order = self.current_orders[broker_order_id]
        order.fields['state'] = state
        order.fields['state_detail'] = ""
        
        self._dispatch_broker_order(order)
        
        
    def _dispatch_broker_order(self, order, validate_order=False):
        
        if not self.context.get('trading.test_broker.disable_dispatch',False):
            if validate_order:
                try:
                    self._validate_order(order)
                except Exception as e:
                    self.logger.info(f"Order validation failed: {order.fields['broker_order_id']}")
                    self.logger.info(traceback.format_exc())
                    
                    order.fields['state'] = Order.S_REJECTED
                    order.fields['state_detail'] = str(e)
            
            broker_order_id = order.fields['broker_order_id']
            
            last_state = self.last_order_state.get(broker_order_id,None)
            if last_state is not None:
                self.orders_by_state[last_state].remove(broker_order_id)
            
            self.orders_by_state[order.fields['state']].append(broker_order_id)
            self.last_order_state[broker_order_id] = order.fields['state']
            
            order._notify_object_load()
            
            dispatch_order = copy.copy(order)
            dispatch_order.fields = copy.copy(order.fields)
            
            self.event_q.put(self._sanitize_order(dispatch_order))
            
    
    def _manual_load_orders_into_queue(self, fname ):
        with open(fname,'rb') as f:
            new_orders = pickle.load(f)
            
        for new_order in new_orders:
            self.event_q.put(new_order)
        
    
    def _sanitize_order(self, order ):
        allowed_fields = ["algo_id","avg_fill_price","broker_order_id","creation_date",
                          "execution_date","filled_qty","limit_price","order_type",
                          "quantity","side","state","state_detail","stop_price",
                          "symbol","time_in_force","is_opening_order","orig_order_id"]
        
        fields = list(order.fields.keys())
        for field in fields:
            if field not in allowed_fields:
                order.fields[field] = None
        
        return order
        
    def _create_new_order(self, submitted_order, do_not_store = False):
        broker_order = self._new_order()
        
        broker_order.fields = copy.deepcopy(submitted_order.fields)
        
        broker_order.from_db = False
        broker_order.is_order_notification = True
        broker_order.remaining_exec_qty = broker_order.fields["quantity"]
        broker_order.exec_legs = []
        broker_order.inc_fill_update = False
        
        if submitted_order.fields['broker_order_id'] is None:
            broker_order.fields["broker_order_id"] = str(submitted_order.fields["internal_order_id"])
        else:
            broker_order.fields["broker_order_id"] = submitted_order.fields["broker_order_id"]
            
        broker_order.fields['state'] = Order.S_WORKING
        
        if not do_not_store:
            with self.writer_lock:
                self.current_orders[broker_order.fields["broker_order_id"]] = broker_order
        
        return broker_order
    
    def _create_response_order( self, submitted_order, fully_fill_order = True, 
                                dispatch_order = True, override_price = None, override_exec_dt = None ):
        
        assert(submitted_order.fields["state"] in [Order.S_WORKING,Order.S_OPEN])
        
        if submitted_order.fields['broker_order_id'] is None:
            if submitted_order.fields['internal_order_id'] is not None:
                submitted_order.fields['broker_order_id'] = str(submitted_order.fields['internal_order_id'])
            else:
                submitted_order.fields['broker_order_id'] = str(random.randint(1,100000000))
                                                                                       
        # get a broker order for this order
        if submitted_order.fields["state"] == Order.S_OPEN:
            broker_order = self._create_new_order(submitted_order)
        else:
            broker_order = self.current_orders[submitted_order.fields["broker_order_id"]]
            
        if self.context.get("trading.test_broker.hang_orders",False):
            return None

        new_state = Order.S_FILLED if (self.fill_all_orders or fully_fill_order) else \
                                        self._get_determined_next_state( submitted_order.fields["state"] )#_get_random_state()
        
        do_not_fill = False
        if submitted_order.fields["order_type"] in [Order.TYPE_STOP, Order.TYPE_STOP_LIMIT, Order.TYPE_LIMIT] and \
            new_state in [Order.S_FILLED, Order.S_WORKING]:
            price = override_price if override_price is not None else \
                                        self.get_quote(broker_order.fields["symbol"],broker_order.fields["side"])
            
            new_state = None
            if submitted_order.fields["order_type"] == Order.TYPE_STOP and \
                 (submitted_order.fields["side"] == "B" and price >= submitted_order.fields["stop_price"] or \
                  submitted_order.fields["side"] == "S" and price <= submitted_order.fields["stop_price"]):
                    if submitted_order.fields['state'] == Order.S_OPEN:
                        broker_order.fields['state_detail'] = "The stop price must be above the current ask for buy stop orders and below the bid for sell stop orders."
                        new_state = Order.S_REJECTED
                    else:
                        new_state = Order.S_FILLED
                        
            if new_state is None:
                if submitted_order.fields["order_type"] == Order.TYPE_LIMIT and \
                  (submitted_order.fields["side"] == "B" and price <= submitted_order.fields["limit_price"] or \
                   submitted_order.fields["side"] == "S" and price >= submitted_order.fields["limit_price"]):
                    new_state = Order.S_FILLED
                else:
                    new_state = Order.S_WORKING
                    do_not_fill = True
        
        is_fill_order = False
        if submitted_order.fields["state"] in [Order.S_OPEN,Order.S_WORKING] and \
            new_state in [Order.S_WORKING, Order.S_FILLED] and not do_not_fill:
            
            if override_exec_dt is None:
                broker_order.fields["execution_date"] = datetime.now()
            else:
                broker_order.fields["execution_date"] = override_exec_dt
            
            is_fill_order = True
            
            fill_qty = broker_order.remaining_exec_qty
            if new_state == Order.S_WORKING:
                if broker_order.fields["order_type"] not in \
                        [Order.TYPE_LIMIT, Order.TYPE_STOP,Order.TYPE_STOP_LIMIT]:
                    partial_fill_qty = fill_qty // 2 - 1
                    if partial_fill_qty <= 0:
                        partial_fill_qty = fill_qty // 2
                    if partial_fill_qty != 0 and partial_fill_qty < fill_qty:
                        fill_qty = partial_fill_qty

            price = override_price if override_price is not None else \
                                        self.get_quote(broker_order.fields["symbol"],broker_order.fields["side"])
            if override_price is None and \
                PARAM_TIME_MACHINE in self.context and \
                broker_order.fields['order_type'] != Order.TYPE_LIMIT:
                penalty = self.context.get("trading.test_broker.trade_penalty",0.01)
                if broker_order.fields['side'] == 'S':
                    price = price - penalty
                else:
                    price = price + penalty
                                        
            #broker_order.add_fill(fill_qty,price)    # not doing aggregate order fill updates
            broker_order.fields["filled_qty"] = fill_qty
            broker_order.fields["avg_fill_price"] = price
            broker_order.remaining_exec_qty = broker_order.remaining_exec_qty - fill_qty
            
            if broker_order.remaining_exec_qty == 0:
                broker_order.fields["state"] = Order.S_FILLED
            else:
                broker_order.fields["state"] = Order.S_WORKING
            
            # OCOs, we need to drop the quantity on the other order if we have a fill
            if broker_order.oco_order_id is not None and fill_qty > 0:
                oco_broker_order = self.current_orders[broker_order.oco_order_id]
                oco_broker_order.fields["state"] = Order.S_CANCELLED
        
                """                
                oco_broker_order.fields["quantity"] = oco_broker_order.fields["quantity"] - fill_qty
                if oco_broker_order.fields["quantity"] <= oco_broker_order.fields["filled_qty"]:
                    if oco_broker_order.fields["quantity"] == 0:
                        oco_broker_order.fields["state"] = Order.S_CANCELLED
                    else:
                        oco_broker_order.fields["state"] = Order.S_FILLED
                """
                
                if dispatch_order:
                    self._dispatch_broker_order( oco_broker_order )
        else:
            broker_order.fields["state"] = new_state

        if is_fill_order:
            copy_order = copy.copy(broker_order)
            copy_order.fields = copy.copy(broker_order.fields)
            broker_order.exec_legs.append(copy_order)
        
        if dispatch_order:
            self._dispatch_broker_order( broker_order )
        
        # if this is a bracet order that is fully filled, then create the bracket order legs
        if broker_order.fields["state"] == Order.S_FILLED and \
                broker_order.fields["order_class"] in [Order.CLASS_BRACKET, Order.CLASS_OTO]:
            
            # Create legs if its a bracket order
            def copy_order_to_leg( source_order, leg_id ):
                order_leg = self._new_order()
                copy_fields = ["algo_id","side","symbol", "time_in_force"]
                for field in copy_fields:
                    order_leg.fields[field] = source_order.fields[field]
            
                order_leg.fields["broker_order_id"] = str(source_order.fields["broker_order_id"]) + "-" + leg_id
                order_leg.fields["state"] = Order.S_WORKING
                order_leg.fields["quantity"] = self._get_total_filled_qty(source_order)
                order_leg.fields["side"] = "B" if source_order.fields["side"] == "S" else "S"
                order_leg.fields["is_opening_order"] = 0
                
                # TDA does not give this so we have to do without
                #order_leg.fields["orig_order_id"] = source_order.fields["broker_order_id"]
                                                                                     
                order_leg.remaining_exec_qty = order_leg.fields["quantity"]
                
                order_leg.fields['creation_date'] = source_order.fields['execution_date']
                
                with self.writer_lock:
                    self.current_orders[order_leg.fields["broker_order_id"]] = order_leg
                
                return order_leg

            stop_leg = None
            limit_leg = None
            if broker_order.fields["bkt_stop_loss"] is not None:
                stop_leg = copy_order_to_leg(broker_order, leg_id="stop")
                stop_leg.fields["stop_price"] = broker_order.fields["bkt_stop_loss"]
                stop_leg.fields["order_type"] = Order.TYPE_STOP
            
            if broker_order.fields["bkt_stop_gain"] is not None:
                limit_leg = copy_order_to_leg(broker_order, leg_id="limit")
                limit_leg.fields["limit_price"] = broker_order.fields["bkt_stop_gain"]
                limit_leg.fields["order_type"] = Order.TYPE_LIMIT

            if stop_leg is not None and limit_leg is not None:
                stop_leg.oco_order_id = limit_leg.fields["broker_order_id"]
                limit_leg.oco_order_id = stop_leg.fields["broker_order_id"]

            if dispatch_order:
                if stop_leg is not None:
                    self._dispatch_broker_order( stop_leg, validate_order=True )
                    
                if limit_leg is not None:
                    self._dispatch_broker_order( limit_leg, validate_order=True )

        return broker_order

    
    
    def get_account_balance(self):
        return 1000000
    
    def is_market_open(self):
        if not self.context.get('trading.test_broker.always_open',True):
            mkt_open,mkt_close = self.get_market_hours(datetime.now())
            
            if mkt_open is not None and mkt_close is not None:
                return datetime.now() >= mkt_open and datetime.now() < mkt_close
        
        return True
    
        
    def get_market_hours(self, query_date):

        if self.context.get('trading.test_broker.always_open',True):
            return datetime.combine(datetime.now().date(),time.min),\
                   datetime.combine(datetime.now().date(),time.max)
        
        trade_cal_df = get_trading_calendar( self.context, datetime.combine(query_date.date(),time.min), 
                                             datetime.combine(query_date,time.max) )
        
        if len(trade_cal_df) == 0:
            return None,None
        
        return trade_cal_df.iloc[0].open.to_pydatetime(),trade_cal_df.iloc[0].close.to_pydatetime()
    
    
    def get_todays_close(self):
        
        open_dt, close_dt = self.get_market_hours( datetime.now() )
        
        if PARAM_TIME_MACHINE in self.context:
            close_dt = datetime.combine(datetime.now().date(),close_dt.time())
        
        return close_dt
    
            
    def _dt_to_ISO(self, dt ):
        if dt is not None:
            ts = pd.Timestamp(dt)
            return ts.tz_localize(self.context[PARAM_LOCAL_TIMEZONE]).astimezone("UTC").strftime(iso_datetime_format)
        
        return None
        
    def _to_dt(self, ts):
        if ts is not None:
            if type(ts) is pd.Timestamp:
                dt = ts.astimezone(tzlocal())
                return datetime(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond)
            elif type(ts) is str:
                dt = datetime.strptime(ts,self.iso_datetime_format)
                dt = pytz.timezone("UTC").localize(dt).astimezone(tzlocal())
                return datetime(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond)
        
        return None    
    
    def get_trading_calendar(self, start_dt, end_dt):
        
        trade_cal_df = get_trading_calendar( self.context, datetime.combine(start_dt.date(),time.min), 
                                             datetime.combine(end_dt,time.max) )
        
        return trade_cal_df
        
        
    def get_quote(self, ticker, side = 'B'):
    
        price_data = get_price_data(self.context,ticker)
        if price_data is not None:
            if side == 'B':
                return price_data['ask']
            else:
                return price_data['bid']
            
        if self.context.get('trading.test_broker.use_live_bidask',False):
            if PARAM_STREAMING_PRICES in self.context:
                try:
                    bidask_df = self.context[PARAM_STREAMING_PRICES].get_bid_ask([ticker])
                    
                    if len(bidask_df) == 1:
                        return bidask_df.loc[ticker].bid if side == 'S' else bidask_df.loc[ticker].ask 
                    else:
                        self.logger.warning("Unable to retrieve bid/ask for symbol {}".format(ticker))
                except:
                    self.logger.error("Exception when getting bid/ask: {}".format(traceback.format_exc()))
            
        return get_symbol_mark(self.context, ticker)
    
        
    # Order Management
    def get_orders(self, broker_order_id = None, status = None, from_dt = None, to_dt = None, ):

        order_list = [value for key,value in self.current_orders.items()]
        
        return_orders = []        
        for order in order_list:
            if broker_order_id is not None and order.fields["broker_order_id"] == broker_order_id or \
                status is not None and (order.fields["state"] == status or status == "all") or \
                from_dt is not None and to_dt is not None and \
                    order.fields["creation_date"] >= from_dt and order.fields["creation_date"] <= to_dt:
                
                q_order = copy.copy(order)
                q_order.is_order_notification = False
                
                if len(q_order.exec_legs) > 0:
                    order_fill_qtys = [exec_leg.fields["filled_qty"] for exec_leg in q_order.exec_legs]
                    order_fill_prices = [exec_leg.fields["avg_fill_price"] for exec_leg in q_order.exec_legs]
                    
                    q_order.fields["filled_qty"] = int(np.sum(order_fill_qtys))
                    if q_order.fields["filled_qty"] != 0:
                        q_order.fields["avg_fill_price"] = (np.sum(np.array(order_fill_qtys) * np.array(order_fill_prices))) / q_order.fields["filled_qty"]
                    else:
                        q_order.fields["avg_fill_price"] = None
                         
                    q_order.remaining_exec_qty = q_order.fields["quantity"] - q_order.fields["filled_qty"]
                
                q_order._notify_object_load()
                
                return_orders.append(q_order)

        return return_orders
    
    def _validate_order(self, order):
        symbol = order.fields['symbol']
        
        if order.fields['order_class'] is not None:
            if order.fields['is_opening_order'] == 0:
                raise ValueError("LT_STOP and LT_BRACKET orders must have is_opening_position=1")
        else:
            open_positions = self.get_open_positions()
            if symbol in open_positions:
                to_open_side = "B" if open_positions[symbol].fields['quantity'] > 0 else "S"
                if open_positions[symbol].fields['quantity'] == 0:
                    to_open_side = None
                order_side = order.fields['side']
                
                if order.fields['is_opening_order'] == 1 and \
                    to_open_side is not None and to_open_side != order_side or \
                    order.fields['is_opening_order'] == 0 and \
                    (to_open_side is None or to_open_side == order_side):
                    raise ValueError("incorrect to_open/to_close flag. opening_order flag is {} when position is {} and order side is {}."\
                                                .format(order.fields['is_opening_order'],
                                                       open_positions[symbol].fields['quantity'],
                                                       order.fields['side']))
                
                if (open_positions[symbol].fields['quantity'] > 0 and \
                    order.fields['side'] == 'S' or 
                   open_positions[symbol].fields['quantity'] < 0 and \
                    order.fields['side'] == 'B') and \
                    abs(order.fields['quantity']) > abs(open_positions[symbol].fields['quantity']):
                    raise ValueError("order would result in an overbought/oversold condition")
            elif order.fields['is_opening_order'] == 0:
                raise ValueError("closing orders not permitted when position is flat")
        
    
    def place_order(self, order ):
        self.record_api_call(f"{order.fields['internal_order_id']}")

        # Non bracket orders must have a flag reflecting the current position
        if order.fields['order_class'] is None:
            order.fields['is_opening_order'] = get_current_is_opening_order_flag( self.context, 
                                                                                  order, 
                                                                                  positions = None, 
                                                                                  adj_positions = False )

        self._validate_order(order)
        
        broker_order = self._create_response_order( order, fully_fill_order = self.fill_all_orders ) #order.fields["time_in_force"] == order.TIF_DAY )            
        
        ret_order = copy.copy(order)
        if order.fields["internal_order_id"] is not None:
            ret_order.fields["broker_order_id"] = str(order.fields["internal_order_id"])
        else:
            ret_order.fields["broker_order_id"] = broker_order.fields['broker_order_id']
            
        ret_order.fields["state"] = Order.S_WORKING
            
        return ret_order
    
    
    def amend_order(self,order):
        self.logger.info("amend_order called: {0}".format(pprint.pformat(order)))
        self.record_api_call(f"{order.fields['internal_order_id']}")
        
        if order.fields['order_class'] is not None:
            self.logger.info("Amending order with child orders will require cancel / place.")
            self.cancel_order(order.fields['broker_order_id'])
            
            new_order = Order(self.context)
            new_order.from_db = False
            
            new_order.fields = copy.deepcopy(order.fields)
            new_order.fields['broker_order_id'] = None
            new_order.fields['internal_order_id'] = None
            new_order.fields['quantity'] = new_order.fields['quantity'] - new_order.fields['filled_qty']
            new_order.fields['state'] = Order.S_OPEN
            
            return self.place_order(new_order)
        else:
            self._validate_order(order)
            
            broker_order = self.current_orders[order.fields["broker_order_id"]]
            new_broker_order = self._create_new_order(broker_order, do_not_store=True)
            new_broker_order.oco_order_id = broker_order.oco_order_id
            new_broker_order.exec_legs = broker_order.exec_legs
            new_broker_order.fields["creation_date"] = datetime.now()
            new_broker_order.remaining_exec_qty = new_broker_order.remaining_exec_qty + order.fields["quantity"] - \
                                                    broker_order.fields["quantity"]
            new_broker_order.stop_limit_activate = broker_order.stop_limit_activate 
            new_broker_order.fields['orig_order_id'] = broker_order.fields['broker_order_id']
    
            broker_order.fields["state"] = Order.S_CANCELLED
            new_broker_order.fields["state"] = Order.S_WORKING
            
            if "am" in broker_order.fields["broker_order_id"]:
                split_id = broker_order.fields["broker_order_id"].split("-")
    
                try:
                    version_number = int(split_id[-1]) + 1
                    split_id[-1] = str(version_number)
                    new_broker_order.fields["broker_order_id"] = "-".join(split_id)
                except:
                    new_broker_order.fields["broker_order_id"] = broker_order.fields["broker_order_id"] + "-am-0"
            else:
                new_broker_order.fields["broker_order_id"] = broker_order.fields["broker_order_id"] + "-am-0"
                
            with self.writer_lock:
                if new_broker_order.oco_order_id is not None:
                    self.current_orders[new_broker_order.oco_order_id].oco_order_id = new_broker_order.fields["broker_order_id"]
                
                self.current_orders[new_broker_order.fields["broker_order_id"]] = new_broker_order
            
            amendable_fields = ["quantity","limit_price","order_type","stop_price",
                                "bkt_stop_loss","bkt_stop_gain","is_opening_order"]
            
            for field in amendable_fields:
                new_broker_order.fields[field] = order.fields[field] 
    
            broker_order.fields['filled_qty'] = 0           # prevents double counting when calculating active positions
            broker_order.exec_legs = []
            
            self._dispatch_broker_order(broker_order)
            self._dispatch_broker_order(new_broker_order)
        
        return order

    
    def cancel_order(self, broker_order_id ):
        self.record_api_call(f"{broker_order_id}")
        broker_order = self.current_orders[broker_order_id]
        
        if broker_order.fields["state"] == Order.S_PENDING_CANCEL or \
            broker_order.fields["state"] == Order.S_CANCELLED:
            return
            #raise Exception("Order already cancelled")
        elif broker_order.fields["state"] in Order.terminal_states:
            raise Exception("Order in terminal state. Can not be cancelled")

        broker_order = self.current_orders[broker_order_id]
        broker_order.fields["state"] = Order.S_CANCELLED
        self._dispatch_broker_order(broker_order)
        
        # if the order we just cancelled is OCO then cancel the other side
        if broker_order.oco_order_id is not None:
            # get oco order and remove from original to avoid a loop
            oco_broker_order = self.current_orders[broker_order.oco_order_id]
            
            # create cancel order for alternate leg
            if oco_broker_order.fields["state"] not in [Order.S_CANCELLED, Order.S_PENDING_CANCEL]:
                oco_broker_order.fields["state"] = Order.S_CANCELLED
                self._dispatch_broker_order(oco_broker_order)

    
    def cancel_all_orders(self):
        pass
    
        
    # Position Management
    def get_open_positions(self):
        
        orders = [value for key,value in self.current_orders.items()]
        
        positions = {}
        for order in orders:
            if order.fields["filled_qty"] > 0:
                cur_pos = Position(self.context)
                            
                if order.fields["symbol"] not in positions.keys():
                    positions[order.fields["symbol"]] = cur_pos
                    cur_pos.fields["symbol"] = order.fields["symbol"]
                    cur_pos.fields["avg_price"] = 0
                    cur_pos.fields["quantity"] = 0
                    cur_pos.fields["market_value"] = 0
                    cur_pos.fields["cost_basis"] = 0
                else:
                    cur_pos = positions[order.fields["symbol"]]
                
                update_qty = order.fields["filled_qty"]
                if order.fields["side"] == "S":
                    update_qty = -1*update_qty
    
                if cur_pos.fields["quantity"] == 0 or \
                    np.sign(update_qty) == np.sign(cur_pos.fields["quantity"]):
                    
                    cur_pos.fields["avg_price"] = (cur_pos.fields["avg_price"]*cur_pos.fields["quantity"] + \
                                                    order.fields["avg_fill_price"] * update_qty) / \
                                                    (cur_pos.fields["quantity"]+update_qty)
                    cur_pos.fields["quantity"] = cur_pos.fields["quantity"] + update_qty
                    cur_pos.fields["cost_basis"] = cur_pos.fields["cost_basis"] + \
                                                    update_qty * order.fields["avg_fill_price"]
                else:   
                    cur_pos.fields["quantity"] = cur_pos.fields["quantity"] + update_qty
                    cur_pos.fields["cost_basis"] = cur_pos.fields["cost_basis"] + \
                                                    update_qty * order.fields["avg_fill_price"]
                    
                if cur_pos.fields["quantity"] == 0: 
                    del positions[cur_pos.fields["symbol"]]


        tickers = [k for k,v in positions.items()]
        
        # set market value
        if len(tickers) > 0:
            for key,pos in positions.items():
                pos.fields["market_value"] = pos.fields["quantity"] * self.get_quote(key,"S" if pos.fields['quantity'] > 0 else "B")
                
        return positions
        
        
    def close_positions(self, symbol):
        pass
    
    def close_all_positions(self):
        pass
        
    def retrieve_order_update(self, block=False, timeout=None):
        try:
            if self.event_q.qsize() > 0:
                outgoing_order = self.event_q.get(block=block,timeout=timeout)

                if self.context.get('trading.test_broker.create_response_orders',True):
                    # Only create a response if this is the latest version of the order
                    latest_order = len(self.current_orders[outgoing_order.fields["broker_order_id"]].compare(outgoing_order)) == 0
                    if latest_order and outgoing_order.fields["state"] == Order.S_WORKING and \
                         outgoing_order.fields["order_type"] == Order.TYPE_MARKET:
                        self._create_response_order( outgoing_order )

                return outgoing_order
        except:
            traceback.print_exc()

        
    def get_reason_type(self, broker_order):
        if broker_order.fields['state_detail'] is not None and \
            "stop price must be above the current ask" in broker_order.fields['state_detail']:
            return BrokerInterface.R_PRICE_MOVED_PAST_STOP
        
        return BrokerInterface.R_NONE
        
        
    
    
        
    
            
                            
                            