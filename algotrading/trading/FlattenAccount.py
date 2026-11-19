import os, logging, sys, pprint, warnings, traceback, argparse, copy, atexit
from datetime import datetime, time, timedelta

import pandas as pd

from algotrading.ContextFields import *
from algotrading.Common import load_configuration,  import_module_attr, global_shutdown

from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor

from algotrading.ContextFields import *
from algotrading.trading.Objects import Order

warnings.filterwarnings(action='error',module='pandas.*')


            
            
def get_responses( broker, timeout=3 ):
    
    order_updates = []
    
    while True:
        order_update = broker.retrieve_order_update(block=True, timeout=timeout)
        if order_update is None:
            break
        else:
            order_updates.append(order_update)
    
    return order_updates


# Determines order of loading files in to the cluster 
if __name__ == "__main__":

    algo_home = os.getenv('ALGO_HOME')
    if algo_home is None:
        print("ALGO_HOME environment variable not set")
        sys.exit(2)

    if algo_home[-1] != "/":
        algo_home = algo_home + "/"
            
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.min_rows', 40)
    pd.set_option('display.max_columns', 20)

    #------------------------------- 
    # APP PARAMETERS
    #-------------------------------
    context = {
                PARAM_APP_NAME : "eod",
                PARAM_ALGO_HOME : algo_home,
                PARAM_REPORTING_PATH : os.path.join(algo_home, "logs/"),   
                PARAM_LOG_PATH : os.path.join(algo_home,"logs/"),
                PARAM_DEFAULT_LOG_LEVEL : logging.DEBUG,
                PARAM_DEFAULT_DASK_LOG_LEVEL : logging.DEBUG,
            }
    

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', help='specify the environment (unit,qa,prod)' )
    
    options_ns = parser.parse_args(sys.argv[1:])
    
    if options_ns.e is not None:
        print("Environment specified: {0}".format(options_ns.e))
        context = load_configuration( context, algo_home, [options_ns.e] )
        context["app.env"] = options_ns.e
    else:
        raise ValueError("Environment argument -e must be specified")
    
    nagios_monitor = NagiosSystemMonitor(context) if "nagios.service_id" in context else None
    
    # init broker connection
    logger = logging.getLogger(context[PARAM_BROKER_INTERFACE_CLASS].split(".")[-1])
    logger.handlers = []
    broker_clean_name = context[PARAM_BROKER_INTERFACE_CLASS].split(".")[-1]
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(context.get('trading.broker.log_level',10))
    
    num_closing_trades = 0
    num_positions_still_open = 0
    try:
        try:
            context[PARAM_LOGGER] = logger
    
            logger.info("Loading {} ".format(broker_clean_name))
            broker = import_module_attr(context[PARAM_BROKER_INTERFACE_CLASS])(context)
            context[PARAM_BROKER_INTERFACE] = broker
            
            logger.info("Cancelling all orders")
            orders = broker.get_orders( from_dt=datetime.combine(datetime.now().date(),time.min)-timedelta(days=5),
                                        to_dt=datetime.combine(datetime.now().date(),time.max) )    
        
            for order in orders:
                try:
                    if order.fields["state"] in Order.non_terminal_states and \
                        order.fields['state'] != Order.S_PENDING_CANCEL:
                        broker.cancel_order( order.fields["broker_order_id"] )
                except:
                    logger.warning("Error cancelling broker order {}".format(order.fields["broker_order_id"]))
                    logger.warning(traceback.format_exc())
            
            logger.info("Retrieving positions and cancelling")    
    
            # flatten open positions    
            open_positions = broker.get_open_positions()
            for symbol,position in open_positions.items():
                if symbol in context.get("trading.symbol_filter",[]):
                    logger.info("Symbol {} filtered. Position will not be closed.".format(symbol))
                    continue
                
                order = Order.new( context, 0, create_rec=False )
                order.fields["side"] = "B" if position.fields["quantity"] < 0 else "S"
                order.fields["order_type"] = Order.TYPE_MARKET
                order.fields["quantity"] = abs(position.fields["quantity"])
                order.fields["is_opening_order"] = 0
                order.fields["symbol"] = position.fields["symbol"]
                order.fields["state"] = Order.S_OPEN
                
                num_closing_trades += 1
                
                try:
                    broker.place_order( order )
                except:
                    logger.warning("unable to place closing order for symbol: {}".format(order.fields["symbol"]))
                    logger.warning(traceback.format_exc())
                
            # wait and then check that positions are closed
            if num_closing_trades > 0:
                responses = get_responses(broker, timeout=context.get('broker.flatten_account.wait_time',10))
                logger.info(f'Responses:\n----------------------\n{pprint.pformat([response.fields for response in responses])}\n----------------------\n')
                open_positions = broker.get_open_positions()
                for symbol,position in open_positions.items():
                    if position.fields['quantity'] > 0:
                        num_positions_still_open += 1
                        logger.error(f"Position still open: {pprint.pformat(position.fields)}")
    
        except:
            logger.fatal("Flatten failed: {}".format(traceback.format_exc()))
            if nagios_monitor is not None:
                nagios_monitor.report_event(NagiosSystemMonitor.MONITOR_CRITICAL,f"Exception encountered in FlattenAccount")
            sys.exit(2)
        
        if num_closing_trades > 0:
            logger.error(f"Number of positions flattened: {num_closing_trades}")
            if nagios_monitor is not None:
                nagios_monitor.report_event(NagiosSystemMonitor.MONITOR_CRITICAL,f"Unclosed positions at end of day: {num_closing_trades}. Still open: {num_positions_still_open}")
            sys.exit(1)
        
        logger.info("No positions to close.")
        sys.exit(0)
    finally:
        global_shutdown()
        if PARAM_BROKER_INTERFACE in context:
            context[PARAM_BROKER_INTERFACE].close()
        



