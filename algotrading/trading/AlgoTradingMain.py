import os, logging, sys, pprint, warnings, traceback, argparse, atexit, faulthandler, signal, asyncio
from datetime import datetime, time, timedelta
import time as ptime

import cProfile

import pandas as pd

from algotrading.ContextFields import *
from algotrading.Common import load_configuration, init_parallel_processing, import_module_attr, \
                                analysis_load_data, BufferingSMTPHandler, get_new_analysis_filename, \
                                apply_configuration_overrides, set_dask_workers_by_symbol, global_shutdown, \
                                register_global_shutdown_handler
                                
from algotrading.util.NagiosSystemMonitor import NagiosHandler
                                
import time_machine
from dateutil.tz import tzlocal

from algotrading.refdata import MarketData, StreamingPrices
from algotrading.analysis.lib.Execution import clear_node_caches
from algotrading.trading.Validations import perform_startup_validations, perform_shutdown_validations, check_valid_algo_runtime
from algotrading.trading.Processing import start_processing, close_all_trades, cancel_all_orders, close_all_positions, analyze_and_create_trades, \
                                                PARAM_POLL_ORDERS_INTERVAL
from algotrading.trading.Common import get_execution_datetime 
from algotrading.trading.Objects import init_sql_caches
from algotrading.trading.Reporting import generate_post_trading_report

warnings.filterwarnings(action='error',module='pandas.*')


# Determines order of loading files in to the cluster 
if __name__ == "__main__":

    algo_home = os.getenv('ALGO_HOME')
    if algo_home is None:
        print("ALGO_HOME environment variable not set")
        sys.exit(2)
            
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.min_rows', 40)
    pd.set_option('display.max_columns', 20)

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)   # kill -SIGUSR1 will dump stack
    
    #------------------------------- 
    # APP PARAMETERS
    #-------------------------------
    context = {
                                PARAM_ALGO_HOME : algo_home,
                                PARAM_REPORTING_PATH : os.path.join(algo_home, "logs/"),   
                                PARAM_LOG_PATH : os.path.join(algo_home,"logs/"),
                                #PARAM_DATADUMP_PATH : str(pathlib.Path().absolute()) + "/node_data/",   
                                PARAM_DASK_WORKER_PATH : os.path.join(algo_home, "tmp/"),
                                PARAM_ALGO_CACHE_PATH : os.path.join(algo_home, "cache/"),
                                PARAM_DEFAULT_LOG_LEVEL : logging.DEBUG,
                                PARAM_DEFAULT_DASK_LOG_LEVEL : logging.DEBUG,
                                
                                PARAM_FEATURES : [],        # features generated in graph

                                PARAM_DISTRIBUTED_MODULES : [],
                                
                                PARAM_MAX_NET_EXPOSURE : 100, 
                                PARAM_MAX_NET_EXPOSURE : 500,   
                                PARAM_MAX_POSITION_EXPOSURE: 500,
                                                        
                                #PARAM_GLOBAL_TEST_BATCH_SIZE : 5,
                                PARAM_ALGO_PREDICTION_MODE : True,
                                PARAM_RESULTS_AUTO_POST_PROC : True,

                                PARAM_ORDER_NOTIFICATION_TIMEOUT : 1,
                                PARAM_TRADE_CANCEL_MAX_RETRY_ORDERS : 3,
                                
                                PARAM_FORCE_THREAD_EXECUTION : False,
                                PARAM_PERSIST_RESULTS_DATA : True,
                                
                                "smtp_logging" : False,
                                "trading.exposure_tolerance" : 0.05,
                                "trading.hold_positions" : False,
                                "trading.suppress_algo_testrun" : False,
                                "secondary_algos" : None,
                                "trading.neutral_long_short_ratio" : None,
                                'refdata.load_intraday_for_today' : True,
                                'trading.open_position_time_window' : 20,
                                "trading.min_free_exposure_for_ti_creation" : 0.05,
                                
                                PARAM_POLL_ORDERS_INTERVAL: 5,
                            }
    
    
    validation_context = {
                                PARAM_MIN_EQUITY_REQUIREMENT : 50000,       # minimum account equity required (validation)
                                PARAM_MAX_REPORT_EXPOSURE_THRESHOLD : 300000,       # maximum open exposure for new orders to be created
                                
                                PARAM_MAX_PNL_LOSS : 0.05,                  # max pnl loss on a position before an alert is added to the pnl report
                                
                                PARAM_MIN_TRADING_DAY_MINUTES : 0,
                                PARAM_PRICE_RECON_LOOKBACK : 1000,
                                PARAM_DATABASE_HISTORY_LOOKBACK : 5,
                            }

    run_context = {}

    # Update main parameter dict
    context.update(validation_context)
    
    # !!!!!! ALL TEST MODE PARAMETERS CHANGES GO HERE !!!!!!!
    test_mode = os.getenv('TEST_MODE')
    if test_mode is not None and int(test_mode) == 1:
        run_context[PARAM_TEST_MODE] = True
        context[PARAM_BROKER_INTERFACE_CLASS] = "algotrading.trading.TestBrokerInterface.TestBrokerInterface"
        context[PARAM_SIMULATE_LIVE_DATA] = True
        context[PARAM_DEFAULT_LOG_LEVEL] = logging.DEBUG
        context[PARAM_LOAD_DATA] = None
        context[PARAM_MAX_REPORT_EXPOSURE_THRESHOLD] = 300000
        run_context[PARAM_FORCE_THREAD_EXECUTION] = True
        

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', help='validation mode')
    parser.add_argument('-ex', action='store_true', help='order execution mode')
    parser.add_argument('-p', action='store_true', help='generate pnl report only')
    parser.add_argument('-xv', action='store_true', help='skip validations')
    parser.add_argument('-xp', action='store_true', help='skip pnl')
    parser.add_argument('-pkg_verify', action='store_true', help='used only by the package installer to verify algos')
    parser.add_argument('-d', action='store', help='specify a fake trade datetime (YYYY-MM-DD HH:MM:SS)' )
    parser.add_argument('-a', action='store', help='specify the algo module to load (eg. algotrading.algos.XYZ)' )
    parser.add_argument('-e', action='store', help='specify the environment (unit,qa,prod)' )
    parser.add_argument('-hyper', action='store', help='override to hyper params of the algo' )
    parser.add_argument('-live_test', action='store_true', help='performs a live test of a single trade instruction' )
    parser.add_argument('-cancel_orders', action='store_true', help='cancels all open orders and quits' )
    parser.add_argument('-close_trades', action='store_true', help='closes out all open trade instructions' )
    parser.add_argument('-close_positions', action='store_true', help='closes out all open positions directly on the broker' )
    parser.add_argument('-profile', action='store_true', help='enable profile mode' )
    
    options_ns = parser.parse_args(sys.argv[1:])
    
    if options_ns.cancel_orders or options_ns.close_positions or options_ns.close_trades:
        options_ns.xv = True    
        options_ns.ex = True
        
    if options_ns.e is not None:
        print("Environment specified: {0}".format(options_ns.e))
        context = load_configuration( context, algo_home, [options_ns.e, options_ns.a] )
        context["app.env"] = options_ns.e
    
    #run_context[PARAM_FORCE_THREAD_EXECUTION] = PARAM_DASK_SCHEDULER not in context.keys()
    
    if options_ns.d is not None:
        fake_dt = datetime.strptime(options_ns.d,"%Y-%m-%d %H:%M:%S").astimezone(tzlocal())
        
        # this has to be done twice or it gets DST wrong. No idea why
        traveller = time_machine.travel(fake_dt)
        traveller.start()
        traveller = time_machine.travel(fake_dt)
        coords = traveller.start()
        atexit.register(traveller.stop)
        
        context[PARAM_TIME_MACHINE] = coords

    context['trading.process_start_time'] = datetime.now()

    if options_ns.a is not None:
        module_name = options_ns.a
        context["secondary_algos"] = None
    else:
        module_name = context["algo"]   
        
    if options_ns.live_test:
        context["trading.test.live_test"] = True
        context["trading.immediate_trade_execution"] = True
        context["trading.test.max_instructions"] = 1
        context["trading.test.test_symbol"] = context.get("trading.test.test_symbol","AAPL")
        
    if "trading.live_date" in context and \
        PARAM_DATABASE_HISTORY_LOOKBACK in context:
        live_date = datetime.strptime(context["trading.live_date"],"%Y-%m-%d").date()
        days_since_live = (datetime.now().date() - live_date).days
        if context[PARAM_DATABASE_HISTORY_LOOKBACK] > days_since_live:
            context[PARAM_DATABASE_HISTORY_LOOKBACK] = days_since_live 

    # set up the logger
    logger = logging.getLogger(__name__)
    logger.handlers = []
    new_format = logging.Formatter('%(asctime)s|%(levelname)s| %(message)s',datefmt='%m/%d/%Y %H:%M:%S')
    
    stdout_logging = os.getenv('STDOUT_LOGGING')
    if stdout_logging is not None and int(stdout_logging) == 1:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(new_format)
        logger.addHandler(handler)
    
    if (test_mode is None or int(test_mode) == 0) and not options_ns.pkg_verify and \
        context["smtp_logging"] == True:
        smtpbuffhandler = BufferingSMTPHandler( context, "Alerts generated from {0}".format(module_name.split(".")[-1]) )
        smtpbuffhandler.setFormatter(new_format)
        logger.addHandler(smtpbuffhandler)
    
    fname = "trading.{0}.{1}".format(module_name.split(".")[-1],datetime.now().strftime("%Y-%m-%d"))
    fpath = get_new_analysis_filename( context, filename = fname, use_path = context[PARAM_LOG_PATH], file_ext = ".log" )

    handler = logging.FileHandler(fpath)
    handler.setFormatter(new_format)
    logger.addHandler(handler)

    if "nagios.service_id" in context:
        handler = NagiosHandler(context)
        logger.addHandler(handler)

    logger.setLevel(context[PARAM_DEFAULT_LOG_LEVEL])
    context[PARAM_LOGGER] = logger

    if module_name is None and not test_mode:
        logger.fatal("No algo specified on command line using -a option or in configuration. Exiting...")
        ptime.sleep(5)
        os._exit(1)
    
    if options_ns.v == True:
        logger.info("Running in validations mode...")

    if options_ns.ex == True:
        logger.info("Running in order execution mode...")
        force_execution_mode = True
    else:
        force_execution_mode = False
    
    if options_ns.p == True:
        logger.info("Running in PnL report mode...")
    
    if options_ns.d is not None:
        logger.info("Running with false date {0}...".format(options_ns.d))
        #run_context[PARAM_FORCE_THREAD_EXECUTION] = True
    
    if options_ns.a is not None:
        logger.info("Algo specified: {0}".format(options_ns.a))

    if context.get("trading.test.live_test",False):
        logger.warning("*** Running in live test mode ***")
        
    #-------------------------------
    # Algo module loaded
    #-------------------------------
    
    params_override = None
    if options_ns.hyper is not None:
        params_override = {}
        overrides = options_ns.hyper.split(';')
        for override in overrides:
            params_override[override.split('!')[0]] = override.split('!')[1]
    try:
        # init parallel processing
        if not context.get("disable_dask",True):
            init_parallel_processing( context )
            
        try:
            # init broker connection
            broker_logger = logging.getLogger(context[PARAM_BROKER_INTERFACE_CLASS].split(".")[-1])
            broker_logger.handlers = []
            
            broker_clean_name = context[PARAM_BROKER_INTERFACE_CLASS].split(".")[-1]
            
            log_filename = "{0}.{1}.{2}".format(broker_clean_name,module_name.split(".")[-1],datetime.now().strftime("%Y-%m-%d"))
            log_path = get_new_analysis_filename( context, filename = log_filename, use_path = context[PARAM_LOG_PATH], file_ext = ".log" )
            
            handler = logging.FileHandler(log_path)
            new_format = logging.Formatter('%(asctime)s|%(levelname)s| %(message)s',datefmt='%m/%d/%Y %H:%M:%S')
            handler.setFormatter(new_format)
            broker_logger.addHandler(handler)
            
            stdout_logging = os.getenv('STDOUT_LOGGING')
            if stdout_logging is not None and int(stdout_logging) == 1:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(new_format)
                broker_logger.addHandler(handler)
    
            broker_logger.setLevel(context.get('trading.broker.log_level',10))
    
            orig_logger = context.get(PARAM_LOGGER,None)
            
            context[PARAM_LOGGER] = broker_logger
            logger.info("Loading {} ".format(broker_clean_name))
            broker = import_module_attr(context[PARAM_BROKER_INTERFACE_CLASS])(context )
            context[PARAM_BROKER_INTERFACE] = broker
            
            register_global_shutdown_handler(context[PARAM_BROKER_INTERFACE].close)
            
            context[PARAM_LOGGER] = orig_logger
            
        except:
            logger.fatal("Initialization failed: {}".format(traceback.format_exc()))
            sys.exit(1)
        
        if PARAM_TIME_MACHINE not in context.keys():
            if PARAM_STREAMING_PRICES_CLASS in context.keys():
                streaming_prices = import_module_attr(context[PARAM_STREAMING_PRICES_CLASS])(context,[])
                context[PARAM_STREAMING_PRICES] = streaming_prices
            elif isinstance(broker,StreamingPrices.StreamingPrices):
                streaming_prices = broker
                context[PARAM_STREAMING_PRICES] = context[PARAM_BROKER_INTERFACE]       
                     
        try:
            logger.info("Loading model file: " + module_name)
            algo_spec = import_module_attr( module_name + ".add_algo_specification")
            if not algo_spec( context, params_override = params_override ):
                logger.error("Algo has nothing to do today. Exiting...")
                sys.exit(0)
                
            active_stocks = import_module_attr(module_name + ".get_active_stocks")(context)
                
            secondary_algo_specs = []
            if context["secondary_algos"] is not None and not options_ns.pkg_verify:
                context["secondary_algos"] = context["secondary_algos"].replace(" ","").split(",")
                
                for algo_name in context["secondary_algos"]:
                    logger.info("Loading secondary model: " + algo_name)
                    secondary_algo_specs.append( import_module_attr( algo_name + ".add_algo_specification") )
                    active_stocks = active_stocks + import_module_attr(algo_name + ".get_active_stocks")(context)
                    
                active_stocks = list(set(active_stocks))
            if len(secondary_algo_specs) > 0:
                context["trading.secondary_algo_specs"] = secondary_algo_specs
            
            context['active_stocks'] = active_stocks
            
            if not context.get("disable_dask",True):
                set_dask_workers_by_symbol( context, active_stocks )
            
        except ImportError: 
            logger.fatal("Failed to import module: " + module_name)
            logger.fatal(f"{traceback.format_exc()}")
            sys.exit(1)
        except:
            logger.fatal("Error in algo init")
            logger.fatal(traceback.format_exc())
            sys.exit(1)
            
        context = apply_configuration_overrides( context )
        
        if not context.get("disable_dask",True) and \
            context.get("clear_node_caches",False):
            clear_node_caches(context)

        #-------------------------------
        # Main Code
        #-------------------------------
        
        
        # write the pid out in case we need to kill the process
        pid = os.getpid()
        with open(os.path.join(context[PARAM_LOG_PATH],f"trading_{pid}.pid"),'w') as f:
            f.write(f"{pid}")
        
        init_sql_caches( context )
        
        algo_logger = logging.getLogger("algo")
        algo_logger.handlers = []
        
        log_filename = "algo.{0}.{1}".format(context[PARAM_APP_NAME],datetime.now().strftime("%Y-%m-%d"))
        log_path = get_new_analysis_filename( context, filename = log_filename, use_path = context[PARAM_LOG_PATH], 
                                   file_ext = ".log" )
        
        handler = logging.FileHandler(log_path)
        new_format = logging.Formatter('%(asctime)s|%(levelname)s| %(message)s',datefmt='%m/%d/%Y %H:%M:%S')
        handler.setFormatter(new_format)
        algo_logger.addHandler(handler)
    
        if stdout_logging is not None and int(stdout_logging) == 1:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(new_format)
            algo_logger.addHandler(handler)
            
        """
        if (test_mode is None or int(test_mode) == 0) and not options_ns.pkg_verify:
            smtpbuffhandler = BufferingSMTPHandler(context, "Alerts generated from algo analysis for {0}".format(context[PARAM_APP_NAME]) )
            smtpbuffhandler.setFormatter(new_format)
            algo_logger.addHandler(smtpbuffhandler)
        """
        
        algo_logger.setLevel(context[PARAM_DEFAULT_LOG_LEVEL])
        context['algo_logger'] = algo_logger
        
        # ------------------------------------------------------------------
        traveller = None
        exit_code = 0    

        trade_cal_df = MarketData.get_trading_calendar( context, datetime.combine(datetime.now().date(),time.min), 
                                             datetime.combine(datetime.now(),time.max) )
        
        if not options_ns.v and \
            not options_ns.ex and \
            not options_ns.p and \
            not context.get('trading.test.live_test',False) and \
            not check_valid_algo_runtime(context):
            logger.info("No algo processing required at this time. Running in execution mode only")
            force_execution_mode = True
        
        if len(trade_cal_df) > 0 and trade_cal_df.iloc[0].date == datetime.now().date():
            if trade_cal_df.iloc[0].half_day:
                if datetime.now() > trade_cal_df.iloc[0].close:
                    logger.info("Running post-close on a half day. Exiting as success....")
                    sys.exit(0)

                if not context.get("trading.trade_halfdays",True):
                    logger.info("Today is a half day and algo does not support half days. Running in execution mode only")
                    force_execution_mode = True

        if options_ns.xv or options_ns.p or options_ns.pkg_verify:       # skip validations flag
            logger.warning("**************** !!! STARTUP VALIDATIONS HAVE BEEN SKIPPED !!! ****************")
        else:
            if not perform_startup_validations( context, options_ns.v ):
                if context.get('trading.startup.close_positions_on_error',False) and \
                    not options_ns.v:
                    logger.error("Startup validations failed. Closing all positions.")
                    options_ns.ex = True
                    options_ns.close_positions = True
                else:
                    raise Exception("Startup validations failed. Exiting...")
            else:
                logger.info("Startup validations complete")

            """
            # not useful                
            if options_ns.v:        # only do in validation mode otherwise runtime is slowed down
                # check that refdata has all prices for valid trading days
                if not prices_up_to_date( context ):
                    raise Exception("prices_up_to_date failed. Exiting...")
            """
            
        if not options_ns.p:    # PnL Mode
            try:
                if not (options_ns.ex or force_execution_mode):        #Order executionmode
                    if PARAM_LOAD_DATA in context.keys() and context[PARAM_LOAD_DATA] is not None:
                        logger.warning("**************** !!! Loading results from data file !!! ****************")
                        
                        new_context = analysis_load_data( context[PARAM_LOAD_DATA], context )
                        context[PARAM_ALGO_RESULTS] = new_context[PARAM_ALGO_RESULTS]
                        context[PARAM_BROKER_INTERFACE] = broker
                    else:
                        if options_ns.pkg_verify:
                            context[PARAM_ALGO_PREDICTION_MODE] = False
                            
                        try:
                            skip_analysis = False
                            if not context.get('trading.test.live_test',False):
                                if not context.get("trading.suppress_algo_testrun",False):
                                    logger.info("Algo pre-run....")
                                    analyze_and_create_trades( context, suppress_trades = True )
                            
                                if options_ns.pkg_verify:
                                    print( context[PARAM_ALGO_DB_FILENAME], flush=True )
                                    os._exit(0)
                                    
                                if not options_ns.v and context.get(PARAM_PRE_OPEN_ALGO_RUN_TIMING,None) is not None:
                                    target_algo_runtime_dt = get_execution_datetime( context, field=PARAM_TARGET_OPEN_POSITION_TIME ) - \
                                                                timedelta(seconds=context[PARAM_PRE_OPEN_ALGO_RUN_TIMING]) + \
                                                                timedelta(seconds=context.get("trading.adjust_algo_run_time",0))
                                    
                                    if target_algo_runtime_dt > datetime.now():
                                        logger.info("Target algo run time is {}. Sleeping...".format(target_algo_runtime_dt.strftime("%Y-%m-%d %H:%M:%S")))                                    
                                        if PARAM_TIME_MACHINE in context:
                                            context[PARAM_TIME_MACHINE].shift((target_algo_runtime_dt-datetime.now()).seconds)
                                        else:
                                            ptime.sleep((target_algo_runtime_dt-datetime.now()).seconds)
                                    else:
                                        logger.info("Pre-run overlaps algo run-time. Skipping algo run and using pre-run results")                                    
                        except:
                            raise
    
            except SystemExit as e:
                raise
            except:
                if options_ns.v:
                    raise
                else:   
                    logger.error( "Exception during analysis phase" )
                    logger.error( traceback.format_exc() )
                    logger.error( "Continuing to process_trades")
                    options_ns.ex = True
                               
            if not options_ns.v and not options_ns.pkg_verify:
                if options_ns.close_trades:
                    close_all_trades( context )
                    context["trading.ti_cancel_is_final"] = True
                if options_ns.cancel_orders:
                    cancel_all_orders( context )
                    context["trading.only_process_notifications"] = True
                elif options_ns.close_positions:
                    cancel_all_orders( context )
                    close_all_positions( context )
                    context["trading.only_process_notifications"] = True
                    
                if options_ns.profile:
                    cProfile.run('process_trades( context, analyze_and_create_trades, execution_mode_only = options_ns.ex or force_execution_mode )', sort=2)
                else:
                    start_processing( context, analyze_and_create_trades, execution_mode_only = options_ns.ex or force_execution_mode )
                logger.info("Trade execution completed")
        else:
            logger.warning("Operating in PnL reporting mode. Algo and trade execution skipped.")

        if not options_ns.xv and not options_ns.p and not options_ns.v and not options_ns.pkg_verify:
            if not perform_shutdown_validations( context ):
                raise Exception("Shutdown validations failed")
            logger.info("Shutdown validations complete")
        
        if not options_ns.v and not options_ns.pkg_verify and not options_ns.xp:
            generate_post_trading_report( context )
        
    except SystemExit as e:
        logger.critical("System exit encountered")
        logger.critical(traceback.format_exc())
        exit_code = e.code
    except:
        logger.fatal(traceback.format_exc())
        exit_code = 1
    finally:
        global_shutdown()
    
    logger.info("Trading process finished")
     
    #os._exit(exit_code)
    sys.exit(exit_code)
    
    
