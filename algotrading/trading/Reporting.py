import os, smtplib
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np


from algotrading.ContextFields import *
from algotrading.Common import get_sql_connection, fix_pandas_fp, get_new_analysis_filename
                                
from algotrading.refdata import MarketData, ReferenceData

from .Objects import PnlEffect, TradeInstruction, Order




def get_agg_pnl( context, start_dt, end_dt ):
    
    conn = get_sql_connection( context )
    cursor = conn.cursor()
    
    sql_query = "SELECT tis.symbol symbol, tis.creation_date trade_open, mos.execution_date trade_close, tis.expected_notional open_trade_value FROM pnl_effects pes "\
                "INNER JOIN trade_instructions tis on tis.trade_instruction_id = pes.trade_instruction_id " \
                "INNER JOIN market_orders mos on tis.close_order_id = mos.internal_order_id " \
                "WHERE mos.execution_date is not NULL and pes.creation_date BETWEEN '{}' AND '{}'".format(start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                                                                          end_dt.strftime("%Y-%m-%d %H:%M:%S"))
         
    cursor.execute(sql_query)

    held_pnls = []
    trade_pnls = []
    benchmark_pnls = {}
    
    granularity = 1 if context.get('trading.continuous.enable',False) else 5
    
    trades_df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    if len(trades_df) > 0:
        start_search = trades_df.trade_open.min()-timedelta(minutes=granularity) #datetime.combine(trades_df.trade_open.min().date(),time.min)
        end_search = trades_df.trade_close.max() #datetime.combine(trades_df.trade_close.max().date(),time.max)
        
        if type(end_search) is float:
            context[PARAM_LOGGER].warning("Unable to find end date for pnl agg {} to {}. Order linkage problem in trade instruction for close order.".format(start_dt,end_dt))
        else:
            benchmark_symbols = context.get('trading.reporting.benchmark_symbols','QQQ,SPY').replace(' ','').split(",")
            traded_symbols = context['active_stocks']
            
            held_basket_df = None
            sym_prices_dict = {}
            for symbol in (traded_symbols+benchmark_symbols):
                sym_prices_dict[symbol] = MarketData.prices_sql( context, symbol, 
                                                                 start_search, end_search, 
                                                                 frequency = "{}min".format(granularity))[['open','close']].loc[start_search:end_search]
                if symbol in traded_symbols:
                    if len(sym_prices_dict[symbol]) > 0:
                        working_df = sym_prices_dict[symbol].copy().dropna()
                        working_df = working_df / working_df.iloc[0].open
                        if held_basket_df is None:
                            held_basket_df = working_df
                        else:
                            held_basket_df = held_basket_df + working_df
                    else:
                        context[PARAM_LOGGER].warning("No price data returned on symbol {}. Held basket PnL may not be accurate.".format(symbol))
                        
            held_basket_df = held_basket_df / len(traded_symbols)
            held_basket_df = held_basket_df.dropna()
            
            total_notional = 0
            benchmark_pnls = { bench_sym : []  for bench_sym in benchmark_symbols}
            
            for idx,trade_row in trades_df.iterrows():
                #prices_df = sym_prices_dict[trade_row.symbol].loc[trade_row.trade_open:trade_row.trade_close]
                #trade_pnls.append((prices_df.iloc[-1].close - prices_df.iloc[0].open) / prices_df.iloc[0].open * trade_row.open_trade_value)
                
                adj_open = trade_row.trade_open-timedelta(minutes=granularity)
                if PARAM_TARGET_OPEN_POSITION_TIME in context:
                    ideal_trade_open_dt = MarketData.get_market_time(context,context[PARAM_TARGET_OPEN_POSITION_TIME],trade_row.trade_open.date())
                    use_open = trade_row.trade_open >= (ideal_trade_open_dt+timedelta(seconds=10))
                else:
                    use_open = False
                
                for bench_sym in benchmark_symbols:
                    prices_df = sym_prices_dict[bench_sym].loc[adj_open:trade_row.trade_close]
                    
                    if len(prices_df) == 0:
                        continue
                    
                    if use_open:
                        benchmark_pnls[bench_sym].append((prices_df.iloc[-1].close - prices_df.iloc[0].open) / prices_df.iloc[0].open * trade_row.open_trade_value)
                    else:
                        benchmark_pnls[bench_sym].append((prices_df.iloc[-1].close - prices_df.iloc[0].close) / prices_df.iloc[0].close * trade_row.open_trade_value)
            
                prices_df = held_basket_df.loc[adj_open:trade_row.trade_close]
                if len(prices_df) > 0:
                    if use_open:
                        held_pnls.append((prices_df.iloc[-1].close - prices_df.iloc[0].open) / prices_df.iloc[0].open * trade_row.open_trade_value)
                    else:
                        held_pnls.append((prices_df.iloc[-1].close - prices_df.iloc[0].close) / prices_df.iloc[0].close * trade_row.open_trade_value)

    pnl_effs = PnlEffect.get( context, creation_date=(start_dt,end_dt) )
    
    total_pnl = 0
    gains = 0
    losses = 0
    biggest_loss = 0
    biggest_gain = 0
    for pnl_eff in pnl_effs:
        pnl = pnl_eff.fields["pnl"]
        
        total_pnl = total_pnl + pnl
        if pnl > 0:
            if pnl > biggest_gain:
                biggest_gain = pnl
            gains = gains + pnl
        elif pnl < 0:
            if pnl < biggest_loss:
                biggest_loss = pnl
            losses = losses + pnl
    
    pnls = { "Total" : total_pnl,
             "Held" : round(np.sum(held_pnls),2) }
    
    if len(benchmark_pnls) > 0:
        for bench_sym in benchmark_symbols:
            pnls[bench_sym] = round(np.sum(benchmark_pnls[bench_sym]),2)
    
    pnls["Gains"] = gains
    pnls["Losses"] = losses
    pnls["Biggest Gain"] = biggest_gain
    pnls["Biggest Loss"] = biggest_loss
    
    return pnls


def get_capital_commit(context, dt):
    
    connection = get_sql_connection( context, refdata = False )
    
    cursor = connection.cursor()
    cursor.execute( f"SELECT * FROM market_orders WHERE date(creation_date) = '{dt.strftime('%Y-%m-%d')}'" )
    
    results_df = pd.DataFrame( cursor.fetchall(), columns=[i[0] for i in cursor.description] )
    cursor.close()
    connection.close()
    
    sod_dt = datetime.combine(dt.date(),time(6,30))
    eod_dt = datetime.combine(dt.date(),time(13,0))
    
    qty_cnt = np.zeros(((eod_dt-sod_dt).seconds,))
    
    filled_df = results_df[results_df['state'] == "FILLED"]
    
    for idx,row in filled_df.iterrows():
        movement = row['filled_qty'] if row['side'] == "B" else -row['filled_qty']
        qty_cnt[(row['execution_date'] - sod_dt).seconds:] += movement
            
    price = context[PARAM_STREAMING_PRICES].get_streaming_last_price( results_df.iloc[0]['symbol'] )
    qty_cnt *= price
    
    return np.min(qty_cnt),np.max(qty_cnt),qty_cnt    
    
    
    
def generate_post_trading_report(context):
    # - PnL generated today
    # - Composition of PnL (fully executed trade instructions)
    # - Open positions
    # - List of any validation exceptions from validatePostTrading

    logger = context[PARAM_LOGGER]
    
    fname = "PnLReport-{0}-{1}-{2}".format(context[PARAM_APP_NAME],datetime.now().strftime("%Y-%m-%d"),context["app.env"])
    use_filename = get_new_analysis_filename( context, filename = fname, use_path = context[PARAM_REPORTING_PATH], 
                                   file_ext = ".txt" )    
    
    f = open(use_filename,"w")
    
    try:
        cur_dt = datetime.now()
        stripped_cur_dt = datetime.combine(cur_dt, time.min)
        today_pnls = get_agg_pnl( context,stripped_cur_dt, datetime.now() )
        week_pnls = get_agg_pnl( context, stripped_cur_dt-timedelta(days=7), datetime.now() )
        month_pnls = get_agg_pnl( context, stripped_cur_dt-timedelta(days=30), datetime.now() )
        prev_week_pnls = get_agg_pnl( context, stripped_cur_dt-timedelta(days=14), stripped_cur_dt-timedelta(days=7) )
        prev_month_pnls = get_agg_pnl( context, stripped_cur_dt-timedelta(days=60), stripped_cur_dt-timedelta(days=30) )
        
        f.write("\n")
        title_str = "PnL Report for {0}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write(title_str)
        f.write("-"*len(title_str))
        f.write("\n\n")
        

        def print_pnl_dict( title, pnl_dict ):
            print_str = title + ": "
            
            first = True
            for k,v in pnl_dict.items():
                if not first:
                    print_str = print_str + ", "
                first = False
                print_str = print_str + "{0}: ${1}".format(k,round(v,2))
            
            return print_str        
        
        f.write( print_pnl_dict("Today",today_pnls) + "\n" )
        f.write( print_pnl_dict("Last 7 Days",week_pnls) + "\n" )
        f.write( print_pnl_dict("Prev 7 Days",prev_week_pnls) + "\n" )
        f.write( print_pnl_dict("Last 30 Days",month_pnls) + "\n" )
        f.write( print_pnl_dict("Prev 30 Days",prev_month_pnls) + "\n" )
        f.write("\n")
    
        pd.set_option('display.max_rows', None)
        pd.set_option('display.min_rows', None)
        pd.set_option('display.max_columns', None)    
        pd.set_option('display.width', 2000)    
        
        f.write("Open Positions\n")
        broker = context[PARAM_BROKER_INTERFACE]
        pos_dict = broker.get_open_positions()
        
        all_pos = []
        for k,v in pos_dict.items():
            if k in context.get("trading.symbol_filter",[]):
                logger.info("Ignoring position on filtered stock {}".format(k))
                continue
            else:
                all_pos.append(v.fields)
        pos_df = pd.DataFrame(all_pos)
        
        if len(pos_df) > 0:
            pos_df['pnl'] = pos_df['market_value'] - pos_df['cost_basis']
            pos_df = fix_pandas_fp( pos_df.round(2) )
            
            pos_df.set_index('symbol',inplace=True)
            f.write(pos_df[['quantity','avg_price','cost_basis','market_value','pnl']].to_string())
        else:
            f.write("No positions\n")
            
        f.write("\n\n")
        
        if len(pos_df) > 0:
            cur_long_exposure = round(pos_df['market_value'][pos_df['market_value'] > 0].sum(),2)
            cur_short_exposure = round(pos_df['market_value'][pos_df['market_value'] < 0].sum(),2)
            unrealized_pnl = round((pos_df['market_value'] - pos_df['cost_basis']).sum(),2)
        else:
            cur_long_exposure = 0
            cur_short_exposure = 0
            cur_exposure = 0
            unrealized_pnl = 0
            
        f.write("Long Market Exposure: {0}. Short Market Exposure: {1}. Unrealized PnL: {2}.\n"\
                        .format(cur_long_exposure,cur_short_exposure,unrealized_pnl))
        if cur_long_exposure > context[PARAM_MAX_REPORT_EXPOSURE_THRESHOLD] or \
            abs(cur_short_exposure) > context[PARAM_MAX_REPORT_EXPOSURE_THRESHOLD]:
            warn_str = "******* MARKET EXPOSURE EXCEEDS THRESHOLD OF {0} ******* ".format(context[PARAM_MAX_REPORT_EXPOSURE_THRESHOLD])
            f.write(warn_str+"\n")
            logger.critical(warn_str)
    
        if unrealized_pnl < -1*context[PARAM_MAX_PNL_LOSS]*context[PARAM_MAX_NET_EXPOSURE]:
            warn_str = "******* UNREALIZED PNL LOSS EXCEEDS THRESHOLD OF {0} ******* ".format(context[PARAM_MAX_PNL_LOSS])
            f.write(warn_str+"\n")
            logger.critical(warn_str)
        
        
        prev_trading_date = MarketData.get_relative_trading_date(context, cur_dt, -1)
        
        if PARAM_GTC_STOP_LOSS_THRESHOLD in context.keys() and \
            PARAM_GTC_STOP_GAIN_THRESHOLD in context.keys():
            gtc_orders = Order.get(context,creation_date=(datetime.combine(prev_trading_date, time.min),
                                                          datetime.combine(prev_trading_date, time.max)),order_type='STOP',
                                                          time_in_force='GTC',
                                                          order_class=None)
        else:
            gtc_orders = Order.get(context,creation_date=(datetime.combine(prev_trading_date, time.min),
                                                          datetime.combine(prev_trading_date, time.max)),order_type='STOP',
                                                          time_in_force='GTC')
        filled_orders = 0
        for order in gtc_orders:
            if order.fields["state"] == Order.S_FILLED:
                filled_orders = filled_orders + 1
        
        f.write("\nGTC Stop Order Execution: {0} executed out of {1} total GTC orders.\n".format(filled_orders,len(gtc_orders)))
        
        f.write("\n")
        f.write("Today's Trade Instructions\n")
        todays_instructions = TradeInstruction.get(context,creation_date=(datetime.combine(cur_dt, time.min),cur_dt))
        instruction_df = TradeInstruction.to_df(todays_instructions)
        if len(instruction_df) > 0:
            f.write(instruction_df[["side","quantity","symbol","state","expected_notional","close_out_date","creation_date"]].to_string())
        else:
            logger.info("No trade instructions today")
            f.write("No trade instructions today")
        f.write("\n\n")
            
        f.write("Today's Orders\n")
        todays_orders = Order.get(context,creation_date=(datetime.combine(cur_dt, time.min),cur_dt))
        order_df = Order.to_df(todays_orders)
        if len(order_df) > 0:
            order_df = order_df.sort_values('symbol')
            f.write(order_df[["trade_instruction_id","side","quantity","symbol","state","filled_qty","avg_fill_price","order_type","netted_order_id","is_net_order","execution_date"]].to_string())
        else:
            logger.info("No orders today")
            f.write("No orders today")
        f.write("\n\n")
        
        f.write("Position Effects\n")
        sql_query = \
"""
select ti.trade_instruction_id, side, quantity, symbol, pnl from pnl_effects pe
        inner join trade_instructions ti
        on ti.trade_instruction_id = pe.trade_instruction_id
        where date(pe.creation_date) = \"{0}\";
""".format(datetime.now().strftime("%Y-%m-%d"))

        conn = get_sql_connection(context)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        pnl_df = pd.DataFrame(data=cursor.fetchall(), index = None, columns = cursor.column_names)
        pnl_df = pnl_df.set_index("trade_instruction_id")
        pnl_df['pnl'] = pnl_df['pnl'].map('{:,.2f}'.format)
        f.write(pnl_df.to_string())
        f.write("\n\n")
        
    finally:
        f.close
    
    if context.get("trading.email_report",False):
        with open(use_filename,"r") as f:
            if PARAM_TEST_MODE not in context.keys() or not context[PARAM_TEST_MODE]: 
                # email via smtp    
                body = f.read()
                
                _header = {'fromaddr': context[PARAM_SMTP_FROM],
                            'toaddrs': context[PARAM_SMTP_TO],
                            'subject' : fname + " ({})".format(os.uname()[1])
                            }
            
                _header['header'] = 'From: {fromaddr}\r\nTo: {toaddrs}\r\nSubject: {subject}\r\n\r\n'.format(**_header)
                msg = _header['header'] + body
                            
                smtp = smtplib.SMTP_SSL(context[PARAM_SMTP_HOST])
                smtp.login(context[PARAM_SMTP_USER],context[PARAM_SMTP_PASSWORD])
                smtp.sendmail(_header['fromaddr'], _header['toaddrs'], msg)
                smtp.quit()
            else:
                print( f.read() )
    else:
        logger.info("SMTP mail info not configured. PnL report will not be mailed")
        
    return




