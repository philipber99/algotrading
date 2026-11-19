import pprint, pytz, logging, queue, traceback, asyncio
from datetime import datetime, time, timedelta, date
from threading import Thread
from dateutil.tz import tzlocal
import pandas as pd

from algotrading.ContextFields import *
from algotrading.trading.BrokerInterface import BrokerInterface
from algotrading.trading.Objects import Order, Position
from algotrading.Common import register_global_shutdown_handler

from alpaca.trading.stream import TradingStream
from alpaca.trading.client import TradingClient
from algotrading.refdata.MarketData import get_trading_calendar
from alpaca.trading.requests import *
from alpaca.common.exceptions import *

    
ALPACA_KEY = 'alpaca.key'
ALPACA_S_KEY = 'alpaca.s_key'
ALPACA_PAPER = 'alpaca.paper_trading'


class AlpacaBrokerInterface(BrokerInterface):
        
    alpaca_date_format = "%Y-%m-%d"
    iso_datetime_format = '%Y-%m-%dT%H:%M:%S.%fZ'


    def _conn_thread_func(self):
        self.conn.run(['trade_updates'])        

        
    def __init__(self, context, streaming = True ):
        
        self.logger = logging.getLogger(AlpacaBrokerInterface.__name__)
        
        self.key = context[ALPACA_KEY]
        self.secret_key = context[ALPACA_S_KEY]
        self.paper = context[ALPACA_PAPER]
            
        self.api = TradingClient( api_key = self.key, secret_key = self.secret_key, paper = self.paper )
        self.context = context
        self.event_q = queue.Queue()
                
        self.acct = self.api.get_account()
        self.acct_configs = self.api.get_account_configurations()
                
        # initialize streaming in events
        if streaming:
            self.trade_stream_client = TradingStream( api_key = self.key, secret_key = self.secret_key, paper = self.paper )
            
            async def handle_trade_updates(data):
                self.logger.info("on_msg: {0}".format(pprint.pformat(data)))
                try:
                    self.event_q.put( data.order, block=False )
                except:
                    pass
                
            self.trade_stream_client.subscribe_trade_updates(handle_trade_updates)
            self.stream_thread = Thread(target=self.trade_stream_client.run)
            self.stream_thread.start()
        else:
            self.trade_stream_client = None
            
        # register shutdown handle
        register_global_shutdown_handler(self.close)

        
    def close(self):
        self.logger.debug( "Closing Alpaca API" )
        self.trade_stream_client.stop()
        

    def __getstate__(self):
        return (self.logger, self.key, self.secret_key, self.base_url, self.api, self.context, self.acct, self.acct_configs)
    
    def __setstate__(self, state):
        self.logger, self.key, self.secret_key, self.base_url, self.api, self.context, self.acct, self.acct_configs = state
        
            
    _state_map = {
                    "new" : Order.S_WORKING,
                    "partially_filled" : Order.S_WORKING,
                    "filled" : Order.S_FILLED,
                    "done_for_day" : Order.S_WORKING,
                    "canceled" : Order.S_CANCELLED,
                    "expired" : Order.S_EXPIRED,
                    "replaced" : Order.S_CANCELLED,
                    "pending_cancel" : Order.S_PENDING_CANCEL,
                    "pending_replace" : Order.S_WORKING,
                    "pending_review" : Order.S_WORKING,
                    "accepted" : Order.S_WORKING,
                    "pending_new" : Order.S_WORKING,
                    "accepted_for_bidding" : Order.S_WORKING,
                    "stopped" : Order.S_CANCELLED,
                    "rejected" : Order.S_REJECTED,
                    "suspended" : Order.S_WORKING,
                    "calculated" : Order.S_WORKING,
                    "held" : Order.S_HELD
                }
    
    _state_rev_map = {v : k for k,v in _state_map.items()}
    
    _type_map = {
                    "market" : Order.TYPE_MARKET,
                    "limit" : Order.TYPE_LIMIT,
                    "stop" : Order.TYPE_STOP,
                    "stop_limit" : Order.TYPE_STOP_LIMIT,
                    "trailing_stop" : Order.TYPE_TRAILING_STOP,
                }
    
    _type_rev_map = {v : k for k,v in _type_map.items()}
    
    _class_map = {
                            "simple" : Order.CLASS_SIMPLE,
                            "bracket" : Order.CLASS_BRACKET,
                            "oco" : Order.CLASS_OCO,
                            "oto" : Order.CLASS_OTO
                    }
    
    _class_rev_map = {v : k for k,v in _class_map.items()}

    _tif_map = {
                            "day" : Order.TIF_DAY,
                            "gtc" : Order.TIF_GTC,
                            "opg" : Order.TIF_OPG,
                            "cls" : Order.TIF_CLS,
                            "ioc" : Order.TIF_IOC,
                            "fok" : Order.TIF_FOK
                        }
    _tif_rev_map = {v : k for k,v in _tif_map.items()}

    
    def _create_order_record( self, broker_order ):
        order_obj = Order(self.context)

        """       
        if type(broker_order) is dict:
            class order_props:
                pass
            
            new_broker_order = order_props()
            for key,value in broker_order.items():
                new_broker_order.__setattr__(key,value)
            
            broker_order = new_broker_order
        """
        
        try:
            order_obj.fields["internal_order_id"] = int(broker_order.client_order_id)
        except:
            order_obj.fields["internal_order_id"] = None

        order_obj.from_db = False
        order_obj.fields["broker_order_id"] = broker_order.id.hex
        order_obj.fields["creation_date"] = broker_order.created_at.astimezone(tzlocal()).replace(tzinfo=None) if broker_order.created_at is not None else None
        order_obj.fields["execution_date"] = broker_order.filled_at.astimezone(tzlocal()).replace(tzinfo=None)if broker_order.filled_at is not None else None
        order_obj.fields["side"] = broker_order.side[0].upper() 
        order_obj.fields["quantity"] = None if broker_order.qty is None else int(broker_order.qty) 
        order_obj.fields["symbol"] = broker_order.symbol 
        order_obj.fields["state"] = self._state_map[broker_order.status]
        order_obj.fields["limit_price"] =  None if broker_order.limit_price is None else float(broker_order.limit_price)
        order_obj.fields["order_type"] = self._type_map[broker_order.type] 
        order_obj.fields["order_class"] = self._class_map[broker_order.order_class] 
        order_obj.fields["filled_qty"] = None if broker_order.filled_qty is None else int(broker_order.filled_qty)
        order_obj.fields["avg_fill_price"] = None if broker_order.filled_avg_price is None else float(broker_order.filled_avg_price) 
        order_obj.fields["stop_price"] = None if broker_order.stop_price is None else float(broker_order.stop_price)
        order_obj.fields["time_in_force"] = None if broker_order.time_in_force is None else self._tif_map[broker_order.time_in_force] 
        
        #if hasattr(broker_order,"reason"):  #broker_order._raw:
        #    order_obj.fields["state_detail"] = broker_order.reason

        # kludge fix as stop orders can come in with a filled_qty even if there is none
        if order_obj.fields["order_type"] == Order.TYPE_STOP and \
            order_obj.fields["state"] == Order.S_CANCELLED:
            order_obj.fields["filled_qty"] = 0
        
        return order_obj
        
    def is_shortable(self, symbols):
        
        if type(symbols) is list:
            shortables_dict = {}
            for symbol in symbols:
                asset = self.api.get_asset(symbol)
                shortables_dict[symbol] = asset.shortable == True and asset.easy_to_borrow == True
                
            return shortables_dict
        else:
            asset = self.api.get_asset(symbols)
            return asset.shortable == True and asset.easy_to_borrow == True

    
    def get_account_balance(self):
        
        self.logger.info("get_account_balance called")
        account = self.api.get_account()
        self.logger.info("get_account_balance resp: {0}".format(pprint.pformat(account)))
        
        return float(account.equity)
    
    def is_market_open(self):
        
        open_dt, close_dt = self.get_market_hours( datetime.now() )
        
        if open_dt is not None and close_dt is not None:
            return datetime.now() >= open_dt and datetime.now() < close_dt
        
        self.logger.error("is_open_market called but no trading hours found for today. Defaulting to closed.")
        return False
    
    
    def get_todays_close(self):
        
        open_dt, close_dt = self.get_market_hours( datetime.now() )
        
        return close_dt
    
            
        
    def get_market_hours(self, query_date):
        
        trade_cal_df = get_trading_calendar( self.context, datetime.combine(query_date.date(),time.min), 
                                             datetime.combine(query_date,time.max) )
        
        if len(trade_cal_df) == 0:
            return None,None
        
        return trade_cal_df.iloc[0].open.to_pydatetime(),trade_cal_df.iloc[0].close.to_pydatetime()
    
    
    def _dt_to_ISO(self, dt ):
        if dt is not None:
            ts = pd.Timestamp(dt)
            return ts.tz_localize(self.context[PARAM_LOCAL_TIMEZONE]).astimezone("UTC").strftime(self.iso_datetime_format)
        
        return None
        
    def _to_dt(self, ts):
        if ts is not None:
            if type(ts) is pd.Timestamp:
                dt = ts.astimezone(tzlocal())
                return datetime(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
            elif type(ts) is str:
                ts = ts.split(".")[0]
                dt = datetime.strptime(ts,self.iso_datetime_format.split(".")[0])
                dt = pytz.timezone("UTC").localize(dt).astimezone(tzlocal())
                return datetime(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
        
        return None
    
    def get_trading_calendar(self, start_dt, end_dt):

        start_dt = datetime.combine(start_dt.date(), time.min)
        end_dt = datetime.combine(end_dt.date(), time.max)

        self.logger.info("get_calendar called. {0}, {1}".format(start_dt.strftime(self.alpaca_date_format),
                                                                          end_dt.strftime(self.alpaca_date_format)))
        
        calendars = self.api.get_calendar(self._dt_to_ISO(start_dt), self._dt_to_ISO(end_dt))
        
        self.logger.info("get_calendar resp: {0}".format(pprint.pformat(calendars)))
        
        dates = [c.date for c in calendars if c.date <= end_dt.date()]
        opens = [c.open for c in calendars if c.date <= end_dt.date()]
        closes = [c.close for c in calendars if c.date <= end_dt.date()]
        
        return pd.DataFrame( { "date" : dates, "open" : opens, "close" : closes } ).set_index("date")
    
    
    # Order Management
    def get_orders(self, broker_order_id = None, status = None, from_dt = None, to_dt = None, ):

        req_params = {}

        if from_dt is not None:
            req_params['after'] = self._dt_to_ISO(from_dt)

        if to_dt is not None:
            req_params['until'] = self._dt_to_ISO(to_dt)
            
        if status == None:
            req_params['status'] = "all"

        req = GetOrdersRequest(**req_params)
        
        #self.logger.info(f"list_orders called: status: {status}, from_dt: {from_dt}, to_dt: {to_dt}")
    
        broker_orders = self.api.get_orders(req)

        #self.logger.info("list_orders resp: {0}".format(pprint.pformat(broker_orders)))
    
        orders = []
        for broker_order in broker_orders:
            try:
                order = self._create_order_record( broker_order )
                orders.append(order)
            except:
                self.logger.error("failed to create order record for broker order {0}".format(broker_order.fields["broker_order_id"]))
                self.logger.error(traceback.format_exc())
                
        return orders
    
    def place_order(self, order):
        
        type_to_req_class_map = {
                                    'market' : MarketOrderRequest,
                                    'limit' : LimitOrderRequest,
                                    'stop' : StopOrderRequest,
                                    'stop_limit' : StopLimitOrderRequest,
                                    'trailing_stop' : TrailingStopOrderRequest
                                }
        
        def submit_order( order_obj ):
            req_dict = {
                            "symbol" : order_obj.fields["symbol"],
                            "qty" : order_obj.fields["quantity"],
                            "side" : "buy" if order_obj.fields["side"] == "B" else "sell",
                            "type" : self._type_rev_map[order_obj.fields["order_type"]],
                            "time_in_force" : order_obj.fields["time_in_force"].lower(),
                            "limit_price" : order_obj.fields["limit_price"],
                            "stop_price" : order_obj.fields["stop_price"],
                            "client_order_id" : None if order_obj.fields["internal_order_id"] is None else str(order_obj.fields["internal_order_id"]),
                            "order_class" : "simple" if order_obj.fields["order_class"] is None else self._class_rev_map[order_obj.fields["order_class"]] 
                        }

            take_profit = None
            stop_loss = None

            if req_dict['order_class'] in ["bracket","oco"]:
                req_dict["take_profit"] = TakeProfitRequest( limit_price = order_obj.fields["bkt_stop_gain"] )
                req_dict["stop_loss"] = StopLossRequest( stop_price = order_obj.fields["bkt_stop_loss"] )
            elif req_dict['order_class'] == 'oto':
                req_dict["stop_loss"] = StopLossRequest( stop_price = order_obj.fields["bkt_stop_loss"] )

            if req_dict['order_class'] == 'oco':              
                req_dict["limit_price"] = order_obj.fields['bkt_stop_gain']

            order_req = type_to_req_class_map[req_dict['type']](**req_dict)

            return self.api.submit_order( order_req )
 
        try:
            broker_order = submit_order(order)
        except APIError as e:
            err_msg = json.loads(e._error)
            e_msg = err_msg['message']
            if e_msg.startswith("insufficient qty available for order"):
                start_available_idx = e_msg.index("available: ") + len("available: ")
                end_available_idx = e_msg[start_available_idx:].index(")") + start_available_idx
                available_qty = int(e_msg[start_available_idx:end_available_idx])

                order.fields["quantity"] = available_qty
                broker_order = submit_order(order)
            else:
                raise
        
        self.logger.info("submit_order resp: {0}".format(pprint.pformat(broker_order)))
        
        return self._create_order_record( broker_order )
    
    def amend_order(self,order):
        self.logger.info("amend_order called: {0}".format(pprint.pformat(order)))
        
        req_dict = {
                        'qty' : order.fields["quantity"],
                        'time_in_force' : self._rev_tif_map[order.fields["time_in_force"]],
                        'limit_price' : order.fields["limit_price"],
                        'stop_price' : order.fields["stop_price"]
                    }
        
        return self.api.replace_order_by_id( order_id=order.fields["broker_order_id"],
                                             order_data=ReplaceOrderRequest(**req_dict))        
    
    def cancel_order(self, broker_order_id ):
        self.logger.info("cancel_order called: {0}".format(broker_order_id))
        
        self.api.cancel_order_by_id( broker_order_id )
        
        
        
    def cancel_all_orders(self):
        pass
    
    
    # Position Management
    def get_open_positions(self):
        
        #self.logger.info("list_positions called")
        positions = self.api.get_all_positions()
        #self.logger.info("list_positions resp: {0}".format(pprint.pformat(positions)))
        
        pos_dict = {}
        for pos in positions:
            new_pos = Position(self.context)
            new_pos.fields["symbol"] = pos.symbol
            new_pos.fields["avg_price"] = float(pos.avg_entry_price)
            new_pos.fields["quantity"] = int(pos.qty)
            new_pos.fields["market_value"] = float(pos.market_value)
            new_pos.fields["cost_basis"] = float(pos.cost_basis)
            
            pos_dict[pos.symbol] = new_pos
            
        return pos_dict
        
        
    def close_positions(self, symbol):
        pass
    
    def close_all_positions(self):
        pass
        
        
    def retrieve_order_update(self, block=False, timeout=None):
        try:
            al_order = self.event_q.get(block=block,timeout=timeout)
            order_obj = self._create_order_record( al_order )
            order_obj.is_order_notification = True
            return order_obj
        except queue.Empty as e:
            pass
        except:
            raise
        
        return None


        
    
            
                            
                            