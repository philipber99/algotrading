import inspect
import numpy as np
from datetime import datetime
from collections import Counter

"""
     Specifies the generic broker interface to implement per provider
"""



class BrokerInterface:
    
    def close(self):
        pass
    
    def is_market_open(self):
        pass
    
    def is_shortable(self, symbol):
        pass
    
    def get_market_hours(self, query_date):
        pass

    def get_account_balance(self):
        pass
    
    def get_trading_calendar(self, start_dt, end_dt):
        pass
    
    # Order Management
    def get_orders(self, broker_order_id = None, status = None, from_dt = None, to_dt = None, ):
        pass
    
    def place_order(self, order):
        pass
    
    def amend_order(self, order):
        pass
    
    def cancel_order(self, broker_order_id):
        pass
    
    def retrieve_order_update(self, block, max_time):
        pass
    
    def get_reason_type(self, broker_order):
        pass

    # Position Management
    def get_open_positions(self):
        pass

    
    # Circuit breakers
    api_call_dts = []
    api_calls = []
    state_changed = False
    
    def reset_state_changed_flag(self):
        self.state_changed = False
        
    def has_state_changed(self):
        return self.state_changed
    
    def record_api_call(self,call_id = "",state_change=False):
        self.api_call_dts.append(datetime.now())
        self.api_calls.append(f"{inspect.stack()[1][3]}|{call_id}")

        if state_change and not self.state_changed:
            self.state_changed = True
    
    def get_recent_api_calls(self,dt_start):
        return np.sum(np.array(self.api_call_dts) > dt_start)
    
    def check_for_api_loop(self,dt_start,max_calls,req_id = False):
        call_ids = np.array(self.api_calls)[np.array(self.api_call_dts) > dt_start]
        counts = Counter(call_ids)
        
        for label,num_calls in counts.items():
            if num_calls > max_calls and \
                (not req_id or len(label.split("|")) == 2):
                return (label,num_calls)
            
        return None
    
    # reason types for order state (usually rejection)
    R_PRICE_MOVED_PAST_STOP = "R_PAST_STOP_LIMIT"     # market price exceededthreshold stop/limit price at time of order placement
    R_NONE = "R_NONE"                                       # no rejection state
            
        
        
    