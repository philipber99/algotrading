import pprint, traceback, queue, threading, copy, warnings
import pandas as pd
import numpy as np

import time as ptime
from datetime import datetime, date, timedelta, time

import mysql.connector
from mysql.connector.constants import FieldType

from threading import Thread

from algotrading.ContextFields import *
from algotrading.Common import get_sql_connection, RWLock, register_global_shutdown_handler
from algotrading.trading.BrokerInterface import BrokerInterface
                                
sql_timestr_format = "%Y-%m-%d %H:%M:%S"
sql_datestr_format = "%Y-%m-%d"

db_updater = None

def flip_side( side ):
    side = side.upper()[0]
    if side == "B":
        return "S"
    return "B"


def init_sql_caches( context ):
    global db_updater
    
    cache_classes = [TradeInstruction, Order, PnlEffect]
    
    for cl in cache_classes:
        if issubclass(cl,CachedSQLObject):
            cl.init_cache(context)
            if db_updater is None:
                db_updater = DatabaseUpdater(context)
                db_updater.start()

    register_global_shutdown_handler(shutdown_sql_caches)


def shutdown_sql_caches():
    global db_updater
    
    if db_updater is not None and not db_updater.shutdown_thread:
        db_updater.shutdown()
    

class SQLObject:

    def __init__(self, context):
        self.context = context
        self.logger = context[PARAM_LOGGER]
        self.fields = { "last_update_date" : datetime.now() }
        self.from_db = False        # indicates if the information is sources from the database
        
    # deived object must set tablename attribute in the class, NOT the instance
    
    # used to notify descendents when we perform an update in the DB on this object
    def _notify_object_commit(self):
        pass

    def _notify_object_load(self):
        pass
            
    @staticmethod
    def _format_value( value ):
        formatted_str = ""
        
        if value is None:
            formatted_str = "NULL"
        elif type(value) is list:
            formatted_str = "("
            
            if len(value) > 0:
                formatted_str = formatted_str + SQLObject._format_value( value[0] )
                for i in range(1,len(value)):
                    formatted_str = formatted_str + "," + SQLObject._format_value(value[i])
            formatted_str += ")"
        elif type(value) is str:
            formatted_str = "\"{0}\"".format(value)
        elif type(value) in [datetime, pd.Timestamp]:
            formatted_str = "\"{0}\"".format(value.strftime(sql_timestr_format))
        elif type(value) is date:
            formatted_str = "\"{0}\"".format(value.strftime(sql_datestr_format))
        elif type(value) is float and np.isnan(value):
            return "NULL"
        else:
            formatted_str = str(value)
            
        return formatted_str
    
    @classmethod
    def _format_conditions( cls, is_update = False, seperator = " AND ", **kwargs):
        
        condition_str = ""
        
        conditions = []
        for (key,value) in kwargs.items():
            if type(value) is tuple:
                if value[0] is None:
                    conditions.append("{0} < {1}".format(key,cls._format_value(value[1])))
                elif value[1] is None:
                    conditions.append("{0} > {1}".format(key,cls._format_value(value[0])))
                else:
                    conditions.append("{0} BETWEEN {1} and {2}".format(key,cls._format_value(value[0]),cls._format_value(value[1])))
                    
            elif type(value) is list:
                conditions.append("{0} IN {1}".format(key,cls._format_value(value)))
            else:
                fmt_value = cls._format_value(value)
                if fmt_value == "NULL" and not is_update:
                    conditions.append("{0} is {1}".format(key,fmt_value))
                else:
                    conditions.append("{0} = {1}".format(key,fmt_value))
        
        if len(conditions) > 0:
            condition_str = conditions[0]
            for i in range(1,len(conditions)):
                condition_str = condition_str + seperator + conditions[i]
                
        return condition_str

    @classmethod
    def _create_objects_from_cursor( cls, context, cursor ):
        
        new_objs = []
        
        field_names = [i[0] for i in cursor.description]
        
        row = cursor.fetchone()
        while row is not None:
            new_obj = cls(context)
            
            fields = {}
            for i in range(len(row)):
                fields[field_names[i]] = row[i]
                       
            setattr(new_obj,"fields",fields)
            new_obj.from_db = True
            
            new_objs.append(new_obj)
                
            row = cursor.fetchone()

        return new_objs


    @classmethod
    def _new_object(cls, context):
        return cls(context)
    
    @staticmethod
    def _generate_insert_query( ins_class, fields ):
        return \
"""
INSERT INTO {0}{1} VALUES {2};
""".format( ins_class.tablename,
            SQLObject._format_value( list(fields.keys()) ).replace("\"",""),
            SQLObject._format_value( list(fields.values()) ))
        
        
        
    
    def insert(self, last_insert_id = False):
        cursor = None
        
        self.fields['last_update_date'] = datetime.now()
        
        sql_query = SQLObject._generate_insert_query( self.__class__, self.fields )
        
        try:
            conn = get_sql_connection( self.context )

            # get connection and execute the query
            cursor = conn.cursor()
            cursor.execute(sql_query)
            conn.commit()
            
            self.from_db = True
            
            if last_insert_id:
                if self.primary_attr[0] not in self.fields or \
                    self.fields[self.primary_attr[0]] is None:
                    cursor.execute("SELECT LAST_INSERT_ID();")
                    return cursor.fetchone()[0]
                else:
                    return self.fields[self.primary_attr[0]]
                
        finally:       
            if cursor is not None:
                cursor.close()

        return None
    
        
    @classmethod
    def get(cls, context, **kwargs):
        # args represents search parameters. the name of the argument is the column and the value is the saerch parameter
        # if an arg is a:
        #         tuple -- search between [0] and [1]
        #         list  -- search for a match to one of the values in the list
        #         str/int -- search for an exact match
        
        # formulate query
        sql_query = \
"""
SELECT * FROM {0} WHERE {1};
""".format(cls.tablename,cls._format_conditions(**kwargs))

        cursor = None
        
        try:
            # get connect and execute the query
            cursor = get_sql_connection( context ).cursor()
            cursor.execute(sql_query)
            
            ret_objects = cls._create_objects_from_cursor( context, cursor )
            for obj in ret_objects:
                obj._notify_object_load()
            
            return ret_objects        
        finally:       
            if cursor is not None:
                cursor.close()

        return None

    @staticmethod
    def _generate_delete_query( del_class, **kwargs ):

        sql_query = \
"""
DELETE FROM {0} WHERE {1};
""".format(del_class.tablename,SQLObject._format_conditions(**kwargs))

        return sql_query
       
       
        
    @classmethod
    def delete(cls, context, **kwargs):
        # args represents search parameters. the name of the argument is the column and the value is the saerch parameter
        # if an arg is a:
        #         tuple -- search between [0] and [1]
        #         list  -- search for a match to one of the values in the list
        #         str/int -- search for an exact match
        
        # formulate query
        sql_query = cls._generate_delete_query(cls, **kwargs)
        
        cursor = None
        
        try:
            conn = get_sql_connection( context )

            # get connect and execute the query
            cursor = conn.cursor()
            cursor.execute(sql_query)
            conn.commit()
        
        finally:       
            if cursor is not None:
                cursor.close()

        return None
        
    @staticmethod
    def _generate_update_query( upd_class, fields ):
        
        where_fields = {}
        set_fields = fields.copy()
        for p_attr in upd_class.primary_attr:
            where_fields[p_attr] = set_fields[p_attr]
            del set_fields[p_attr]
            
        set_conditions = SQLObject._format_conditions( is_update = True, seperator = ", ", **set_fields )
        where_conditions = SQLObject._format_conditions( is_update = True, **where_fields )
        
        sql_query = \
"""
UPDATE {0} SET {1} WHERE {2};
""".format(upd_class.tablename,set_conditions,where_conditions)

        return sql_query
    

    def update(self):
        
        cursor = None
        
        self.fields['last_update_date'] = datetime.now()
        
        sql_query = self._generate_update_query(self.__class__,self.fields)
        
        try:
            conn = get_sql_connection( self.context )

            # get connection and execute the query
            cursor = conn.cursor()
            cursor.execute(sql_query)
            conn.commit()
            
            self.from_db = True
            
            self._notify_object_commit()
        
        finally:       
            if cursor is not None:
                cursor.close()

        return None


    @classmethod
    def batch_update(cls, context, objects):
        
        # generate the queries for each update
        sql_queries = []
        for obj in objects:
            obj.fields['last_update_date'] = datetime.now()
            sql_queries.append(cls._generate_update_query(cls,obj.fields))
 
        try:
            # get the connection
            conn = get_sql_connection( context )
            cursor = conn.cursor()

            # execute each query            
            for sql_query in sql_queries:
                cursor.execute(sql_query)
            
            # commit at once
            conn.commit()
            
            for obj in objects:
                obj._notify_object_commit()
        
        finally:       
            if cursor is not None:
                cursor.close()

        return
    
    @classmethod
    def to_df(cls, objects):
        if len(objects) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame( objects[0].fields,index=[0] )
        
        append_list = []
        for i in range(1,len(objects)):
            append_list.append(objects[i].fields)
            #df = df.append(objects[i].fields,ignore_index=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = pd.concat([df,pd.DataFrame(append_list)])
        
        df.set_index(cls.primary_attr,inplace=True)
        return df

    def pprint(self):
        return "{0} Fields: {1}".format(type(self), pprint.pformat(self.fields))


class DatabaseUpdater(Thread):
    
    sql_cmd_queue = queue.Queue()
    shutdown_event = threading.Event()
    shutdown_thread = False
    
    def __init__(self, context):
        super().__init__()
        
        self.context = context
        self.logger = context[PARAM_LOGGER]
        self.connection = mysql.connector.connect(
            host=context[PARAM_DB_HOSTNAME],
            user=context[PARAM_DB_USERNAME],
            passwd=context[PARAM_DB_PASSWORD],
            database=context[PARAM_DB_DEFAULTDB])

        self.command_batch_size = self.context.get("trading.dbcache.command_batch_size",10)
        self.shutdown_event.clear()
        self.loop_delay = self.context.get("trading.dbcache.sql_updater_loop_delay",1)
        self.shutdown_timeout = self.context.get("trading.dbcache.shutdown_timeout",60)
        self.shutdown_start_dt = None
        
    def shutdown(self):
        
        if self.shutdown_start_dt is None:
            self.shutdown_start_dt = datetime.now()
            self.shutdown_thread = True
            self.shutdown_event.wait()
            self.connection.close()
        
        
    def check_connection(self):
        if not self.connection.is_connected():
            self.connection.connect()
            
        return self.connection.is_connected()

    
    def queue_sql_query(self, cls, sql_query):
        self.sql_cmd_queue.put((cls,sql_query))


    def process_queries(self):
        cursor = self.connection.cursor()
        
        num_commands = 0
        while not self.sql_cmd_queue.empty():
            num_commands = num_commands + 1
            try:
                (cls,query) = self.sql_cmd_queue.get()
                if self.context.get("trading.dbcache.log_queries",False):
                    self.logger.debug("Executing query: {}".format(query))
                cursor.execute(query)
            except:
                self.logger.error(traceback.format_exc())
                self.logger.error("Offending Query: {}".format(query))
                
            if num_commands > self.command_batch_size:
                #self.logger.debug("Committing transactions")
                self.connection.commit()
            
        #self.logger.debug("Committing transactions (out of loop)")
        self.connection.commit()
        cursor.close()

    def run(self):
        
        while True:
            try:
                self.check_connection()
                
                if self.shutdown_start_dt is not None:
                    if (datetime.now() - self.shutdown_start_dt).seconds > self.shutdown_timeout and \
                        not self.sql_cmd_queue.empty():
                        self.logger.critical("SQL command queue empty but shutdown timeout limit reached")
                        break
                    
                if not self.sql_cmd_queue.empty():
                    self.process_queries()
                    
                if self.loop_delay != 0:
                    ptime.sleep(self.loop_delay)
                
                if self.shutdown_thread: 
                    if self.sql_cmd_queue.empty():
                        break
                    else:
                        self.logger.debug("Shutdown received for database cache, command queue is not empty")
            except:
                self.logger.error("Exception in database cache updater")
                self.logger.error(traceback.format_exc())

        try:
            if not self.sql_cmd_queue.empty():
                self.logger.error("uncommitted contents of SQL command queue: ")
                while not self.sql_cmd_queue.empty():
                    self.logger.error(pprint.pformat(self.sql_cmd_queue.get()))
        except:
            self.logger.error(traceback.format_exc())
            
        self.logger.debug("Database cache thread is complete")
        self.shutdown_event.set()
        



class CachedSQLObject(SQLObject):

    _sql_type_map = {    
                        FieldType.DECIMAL : "float64",
                        FieldType.TINY : "float64",
                        FieldType.SHORT : "float64",
                        FieldType.LONG : "float64",
                        FieldType.FLOAT : "float64",
                        FieldType.DOUBLE : "float64",
                        FieldType.TIMESTAMP : "datetime64[ns]",
                        FieldType.LONGLONG : "float64",
                        FieldType.INT24 : "float64",
                        FieldType.DATE : "datetime64[ns]",
                        FieldType.TIME : "datetime64[ns]",
                        FieldType.DATETIME : "datetime64[ns]",
                        FieldType.NEWDATE : "datetime64[ns]",
                        FieldType.VARCHAR : "string",
                        FieldType.VAR_STRING : "string",
                        FieldType.STRING : "string"
                    }


    _obj_type_map = {    
                        FieldType.DECIMAL : "float64",
                        FieldType.TINY : "int",
                        FieldType.SHORT : "int",
                        FieldType.LONG : "int",
                        FieldType.FLOAT : "float64",
                        FieldType.DOUBLE : "float64",
                        FieldType.TIMESTAMP : "datetime64[ns]",
                        FieldType.LONGLONG : "float64",
                        FieldType.INT24 : "int",
                        FieldType.DATE : "datetime64[ns]",
                        FieldType.TIME : "datetime64[ns]",
                        FieldType.DATETIME : "datetime64[ns]",
                        FieldType.NEWDATE : "datetime64[ns]",
                        FieldType.VARCHAR : "string",
                        FieldType.VAR_STRING : "string",
                        FieldType.STRING : "string"
                    }

    _pandas_na_type = {
                        "float64" : np.nan,
                        "datetime64[ns]" : pd.NaT,
                        "string" : None
                        }
    
    _na_to_text = {
                        0 : "0",
                        np.nan : "np.nan",
                        pd.NaT : "pd.NaT",
                        None : "None"
                    }
    
    _df_caches = {}
    _df_cache_locks = {}
    
    _next_ids = {}
   
    _pandas_col_types = {}
    _real_col_types = {}
    
    _sql_col_types = {}
    

    def __init__(self, context):
        super(CachedSQLObject,self).__init__(context)
    
    @classmethod
    def init_cache(cls, context):

        cls.df = pd.DataFrame()
                
        cutoff_dt = datetime.combine(datetime.now().date(),time.min) - \
                            timedelta(days=context[PARAM_DATABASE_HISTORY_LOOKBACK])
        
        sql_query = "SELECT * FROM {} WHERE creation_date >= '{}';".format(cls.tablename,
                                                                            cutoff_dt.strftime("%Y-%m-%d %H:%M:%S"))

        cursor = None
        
        try:
            cursor = get_sql_connection( context ).cursor()
            cursor.execute(sql_query)
            
            cls._df_caches[cls.tablename] = pd.DataFrame( cursor.fetchall(), columns=[i[0] for i in cursor.description] )
            
            for i in cursor.description:
                target_type = cls._sql_type_map[i[1]]
                
                cls._sql_col_types[i[0]] = i[1] 
                cls._pandas_col_types[i[0]] = target_type
                cls._real_col_types[i[0]] = cls._obj_type_map[i[1]]
                
                cls._df_caches[cls.tablename][i[0]] = cls._df_caches[cls.tablename][i[0]].astype(target_type)

            cls._df_caches[cls.tablename][cls.primary_attr[0]] = cls._df_caches[cls.tablename][cls.primary_attr[0]].astype('int')
            cls._df_caches[cls.tablename].set_index([cls.primary_attr[0]],inplace=True)

            cls._df_cache_locks[cls.tablename] = RWLock()
            
            cursor.execute("""
SELECT *
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = '{}'
    AND TABLE_NAME = '{}'
    AND COLUMN_NAME = '{}'
    AND DATA_TYPE = 'int'
    AND COLUMN_DEFAULT IS NULL
    AND IS_NULLABLE = 'NO'
    AND EXTRA like '%auto_increment%';
""".format(context[PARAM_DB_DEFAULTDB],cls.tablename,cls.primary_attr[0]))
            
            if len(cursor.fetchall()) > 0:
                cursor.execute("""
    SELECT AUTO_INCREMENT
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = "{}"
    AND TABLE_NAME = "{}";
    """.format(context[PARAM_DB_DEFAULTDB],cls.tablename))
                
                cls._next_ids[cls.tablename] = int(cursor.fetchall()[0][0])
            else:
                if len(cls._df_caches[cls.tablename].index) > 0:
                    cls._next_ids[cls.tablename] = cls._df_caches[cls.tablename].index.max()+1
                else: 
                    cls._next_ids[cls.tablename] = 0
        except:
            self.logger.error(traceback.format_exc())
            raise
        finally:       
            if cursor is not None:
                cursor.close()

    
    def _notify_object_commit(self):
        pass
        
    @staticmethod
    def _format_value( key, value ):
        formatted_str = ""
        
        if value is None:
            formatted_str = CachedSQLObject._na_to_text[CachedSQLObject._pandas_na_type[CachedSQLObject._pandas_col_types[key]]]
        elif type(value) is list:
            formatted_str = "["
            if len(value) > 0:
                formatted_str = formatted_str + SQLObject._format_value( value[0] )
                for i in range(1,len(value)):
                    formatted_str = formatted_str + "," + SQLObject._format_value(value[i])
            formatted_str += "]"
        elif type(value) is str:
            formatted_str = "\"{0}\"".format(value)
        elif type(value) in [datetime, pd.Timestamp]:
            formatted_str = "\"{0}\"".format(value.strftime(sql_timestr_format))
        elif type(value) is date:
            formatted_str = "\"{0}\"".format(value.strftime(sql_datestr_format))
        else:
            formatted_str = str(value)
            
        return formatted_str
    
    @classmethod
    def _format_conditions( cls, is_update = False, seperator = "&", **kwargs):
        
        condition_str = ""
        
        conditions = []
        for (key,value) in kwargs.items():
            if type(value) is tuple:
                if value[0] is None:
                    conditions.append("read_df['{0}'] < {1}".format(key,cls._format_value(key,value[1])))
                elif value[1] is None:
                    conditions.append("read_df['{0}'] > {1}".format(key,cls._format_value(key,value[0])))
                else:
                    conditions.append("(read_df['{0}'] >= {1})&(read_df['{0}'] <= {2})".format(key,cls._format_value(key,value[0]),cls._format_value(key,value[1])))
                    
            elif type(value) is list:
                conditions.append("read_df['{0}'].isin({1})".format(key,cls._format_value(key,value)))
            elif pd.isna(value):
                conditions.append("read_df['{0}'].isna()".format(key))
            else:
                fmt_value = cls._format_value(key,value)
                conditions.append("read_df['{0}'] == {1}".format(key,fmt_value))
        
        if len(conditions) > 0:
            condition_str = conditions[0]
            for i in range(1,len(conditions)):
                condition_str = "(" + condition_str + ")" + seperator + "(" + conditions[i] + ")"
                
        return condition_str

    @classmethod
    def _create_objects_from_df( cls, context, df ):
        
        new_objs = []
        
        for idx,row in df.iterrows():
            new_obj = cls(context)

            new_obj.fields = row.to_dict()
            
            for k,v in new_obj.fields.items():
                if pd.isna(v):
                    new_obj.fields[k] = None
                
                if cls._real_col_types[k] == "int" and \
                    new_obj.fields[k] is not None:
                    new_obj.fields[k] = int(new_obj.fields[k])
                elif cls._real_col_types[k] == "datetime64[ns]" and \
                    new_obj.fields[k] is not None:
                    new_obj.fields[k] = new_obj.fields[k].to_pydatetime()
                
            new_obj.fields[cls.primary_attr[0]] = idx
            
            new_obj.from_db = True
            new_objs.append(new_obj)

        return new_objs

    @classmethod
    def _get_new_id(cls):
        ret_id = cls._next_ids[cls.tablename]
        cls._next_ids[cls.tablename] = ret_id + 1
        
        return ret_id
        

    @classmethod
    def _new_object(cls, context):
        return cls(context)
    
    @classmethod
    def _fix_df_types(cls, df):
        for col_name in df.columns:
            df[col_name] = df[col_name].astype(cls._pandas_col_types[col_name])
    
    @classmethod
    def _fields_to_df(cls,fields_dict):
        fields_dict = fields_dict.copy()
        for k,v in fields_dict.items():
            if v is None:
                fields_dict[k] = cls._pandas_na_type[cls._pandas_col_types[k]]
            elif type(v) is datetime:
                fields_dict[k] = v.replace(microsecond=0)
                
        idx = fields_dict[cls.primary_attr[0]]
        del fields_dict[cls.primary_attr[0]]

        ret_df = pd.DataFrame(fields_dict,index=[idx])
        ret_df.index.name = cls.primary_attr[0]
    
        return ret_df
                
    def insert(self, last_insert_id = False):
        cursor = None
        
        self.fields['last_update_date'] = datetime.now()
        
        try:
            self._df_cache_locks[self.tablename].acquire_write()
            if self.primary_attr[0] not in self.fields or \
                self.fields[self.primary_attr[0]] is None:
                self.fields[self.primary_attr[0]] = self._get_new_id()
            
            if len(self._df_caches[self.tablename]) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self._df_caches[self.tablename] = pd.concat([self._df_caches[self.tablename],self._fields_to_df(self.fields)])
            else:
                self._df_caches[self.tablename] = self._fields_to_df(self.fields)
            
            sql_query = SQLObject._generate_insert_query( self.__class__, self.fields )
            db_updater.queue_sql_query(self.__class__,sql_query)
        finally:
            self._df_cache_locks[self.tablename].release()
        
        self.from_db = True
        
        return self.fields[self.primary_attr[0]]
    
            
    @classmethod
    def query_df(cls, **kwargs):
        query = "query_df = read_df[" + cls._format_conditions(**kwargs) + "]"
        
        try:
            cls._df_cache_locks[cls.tablename].acquire_read()
            read_df = cls._df_caches[cls.tablename].reset_index()
        finally:
            cls._df_cache_locks[cls.tablename].release()
            
        exec( query, globals(), locals() )

        return locals()['query_df'].set_index(cls.primary_attr[0])
    
        
    @classmethod
    def get(cls, context, **kwargs):
        
        ret_objects = cls._create_objects_from_df( context, cls.query_df(**kwargs) )
        for obj in ret_objects:
            obj._notify_object_load()
        
        return ret_objects



    @classmethod
    def delete(cls, context, **kwargs):
        
        del_rows_df = cls.query_df(**kwargs)
        
        try:
            cls._df_cache_locks[cls.tablename].acquire_write()
            cls._df_caches[cls.tablename] = cls._df_caches[cls.tablename].drop(index=del_rows_df.index)
        finally:
            cls._df_cache_locks[cls.tablename].release()

        db_updater.queue_sql_query(cls,SQLObject._generate_delete_query(cls,**kwargs))

        return None


    def update(self,withhold_commit=False):

        if not self.from_db:
            raise ValueError("Unable to update object that is not from the database")
        
        self.fields['last_update_date'] = datetime.now()
        
        new_row = self._fields_to_df(self.fields)
        
        try:
            self._df_cache_locks[self.tablename].acquire_write()
            self._df_caches[self.tablename].drop(index=new_row.index,inplace=True)
            
            if len(self._df_caches[self.tablename]) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self._df_caches[self.tablename] = pd.concat([self._df_caches[self.tablename],new_row])
            else:
                self._df_caches[self.tablename] = new_row
        
            sql_query = SQLObject._generate_update_query(self.__class__,self.fields)
            db_updater.queue_sql_query(self.__class__,sql_query)
        finally:
            self._df_cache_locks[self.tablename].release()
        
        self.from_db = True
        
        if withhold_commit:
            return self._notify_object_commit
        else:
            self._notify_object_commit()
            
        return None
    
    @classmethod
    def batch_update(cls, context, objects):
        
        commit_funcs = []
        for obj in objects:
            commit_funcs.append(obj.update(withhold_commit=True))

        for commit_func in commit_funcs:
            if commit_func is not None:
                commit_func()

        return
        


class Order(CachedSQLObject):
    # ORDER STATES
    S_CREATED = "CREATED"
    S_OPEN = "OPEN"
    S_WORKING = "WORKING"
    S_FILLED = "FILLED"
    S_HELD = "HELD"
    S_EXPIRED = "EXPIRED"
    S_PENDING_CANCEL = "PENDING_CANCEL"
    S_CANCELLED = "CANCELLED"
    S_REJECTED = "REJECTED"
    S_NETTED_PENDING = "NETTED_PENDING"
    S_NETTED = "NETTED"
    S_NETTED_CANCELLED = "NETTED_CANCELLED"
    S_NETTED_PARTIAL = "NETTED_PARTIAL"
    S_CROSS = "CROSS"       # means this order was filled by another order internally
    
    # ORDER TYPES
    TYPE_MARKET = "MARKET"
    TYPE_LIMIT = "LIMIT"
    TYPE_STOP = "STOP"
    TYPE_STOP_LIMIT = "STOP_LIMIT"
    TYPE_TRAILING_STOP = "TRAILING_STOP"
    
    # ORDER Time in Force
    TIF_DAY = "DAY"
    TIF_GTC = "GTC"
    TIF_OPG = "OPG"
    TIF_CLS = "CLS"
    TIF_IOC = "IOC"
    TIF_FOK = "FOK"
    
    # ORDER Classes
    CLASS_SIMPLE = "SIMPLE"
    CLASS_BRACKET = "BRACKET"
    CLASS_OCO = "OCO"
    CLASS_OTO = "OTO"
        
    tablename = "market_orders"
    primary_attr = ["internal_order_id"]
    
    terminal_states = [S_FILLED,S_EXPIRED,S_CANCELLED,S_REJECTED,S_NETTED,S_NETTED_PARTIAL,S_NETTED_CANCELLED,S_CROSS]
    non_terminal_states = [S_OPEN,S_WORKING,S_HELD,S_PENDING_CANCEL,S_NETTED_PENDING]
    internal_only_states = [S_CREATED,S_OPEN,S_NETTED_PENDING,S_NETTED,S_NETTED_CANCELLED,S_NETTED_PARTIAL,S_CROSS]     # ie. no mirror order on the broker
    working_states = [S_CREATED,S_OPEN,S_WORKING,S_HELD,S_NETTED_PENDING]
    
    price_fields = ["limit_price","avg_fill_price","stop_price","bkt_stop_loss","bkt_stop_gain"]
    
    broker_populated_fields = ["internal_order_id","broker_order_id","execution_date","side","quantity","symbol","state",
                               "limit_price","order_type","filled_qty","avg_fill_price","state_detail"]
    
    def __init__(self, context):
        super().__init__(context)
        self.fields = {
                                 #"internal_order_id" : None,   # AUTO POPULATED, DO NOT SPECIFY
                                 "broker_order_id" : None,
                                 "internal_order_id" : None,
                                 "algo_id" : context[PARAM_APP_NAME],
                                 "creation_date" : datetime.now(),
                                 "execution_date" : None,
                                 "trade_instruction_id" : None,
                                 "side" : None,
                                 "quantity" : 0,
                                 "symbol" : "",
                                 "state" : Order.S_CREATED,
                                 "state_detail" : None,
                                 "limit_price" : None,
                                 "order_class" : None,
                                 "order_type" : Order.TYPE_MARKET,
                                 "filled_qty" : 0,
                                 "is_net_order" : 0,
                                 "execute_after" : None,
                                 "avg_fill_price" : None,
                                 "netted_order_id" : None,
                                 "time_in_force" : Order.TIF_DAY,
                                 "stop_price" : None,
                                 "expire_at" : None,
                                 "bkt_stop_loss" : None,
                                 "bkt_stop_gain" : None,
                                 "is_opening_order" : None,
                                 "orig_order_id" : None
                                }
        
        self.is_order_notification = False
        self.remaining_exec_qty = None
        self.inc_fill_update = False
    
    
    @classmethod
    def new( cls, context, trade_instruction_id = None, create_rec = True ):

        new_order = cls._new_object(context)
        new_order.fields["trade_instruction_id"] = trade_instruction_id
        new_order.fields["algo_id"] = context[PARAM_APP_NAME]
        
        if create_rec:
            insert_id = new_order.insert(last_insert_id = True)
            new_order.fields["internal_order_id"] = insert_id

        return new_order
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k != 'context':
                setattr(result, k, copy.deepcopy(v, memo))
            
        return result
    
    
    def _notify_object_commit(self):
        if self.fields["trade_instruction_id"] is not None:
            # update trade instruction state
            instruction = TradeInstruction.get(self.context,trade_instruction_id=self.fields["trade_instruction_id"])
            if len(instruction) != 1:
                self.context[PARAM_LOGGER].error("Unable to find trade instruction id {} in database".format(self.fields["trade_instruction_id"]))
            else:
                instruction[0].update_state()
                
        
    def _notify_object_load(self):
        for field in self.price_fields:
            if self.fields[field] is not None:
                self.fields[field] = np.round(self.fields[field],2)
    
    @classmethod
    def get_order(cls, context, order_id):
        
        latest_order_id = order_id
        
        while True:
            orders = Order.get(context,orig_order_id=latest_order_id)
            if len(orders) > 0:
                if latest_order_id != orders[0].fields['internal_order_id']: 
                    latest_order_id = orders[0].fields['internal_order_id']
                else:
                    self.context[PARAM_LOGGER].error("Infinite loop detected on latest order search for {}".format(latest_order_id))
                    break
            else:
                break    
        
        return Order.get(context,internal_order_id=latest_order_id)[0]

    def compare(self, order_obj, broker_fields=False):
        # first copy dictionaries and remove any Nones
        dict1 = { k:v for k,v in self.fields.items() if v is not None}
        dict2 = { k:v for k,v in order_obj.fields.items() if v is not None}

        field_faults = []
                
        # find whats missing from 2 thats in 1
        #missing_from_2 = list(set(dict1.keys()) - set(dict2.keys()))
        #if len(missing_from_2) != 0:
        #    field_faults = field_faults + missing_from_2 
        #    for fld in missing_from_2:
        #        del dict1[fld]
                
        # find whats missing from 1 in 2
        missing_from_1 = list(set(dict2.keys()) - set(dict1.keys()))
        if len(missing_from_1) != 0:
            field_faults = field_faults + missing_from_1 
            for fld in missing_from_1:
                del dict2[fld]
            
        # compare the remaining fields (exclude some fields because they are not on the broker side
        for key in dict1.keys():
            if (key in dict2) and \
                ((type(dict1[key]) is datetime and dict1[key].date() != dict2[key].date()) or \
                key == "avg_fill_price" and abs(dict1[key]-dict2[key]) > 0.02 or \
                 (type(dict1[key]) is not datetime and key != "avg_fill_price" and dict1[key] != dict2[key])):
                field_faults.append(key)

        ignore_field_list = ["trade_instruction_id","is_net_order","execute_after","expire_at","internal_order_id","order_class",
                                "bkt_stop_gain","bkt_stop_loss", "netted_order_id", "is_opening_order","time_in_force","algo_id",
                                "orig_order_id","last_update_date"]

        field_faults = list(set(field_faults) - set(ignore_field_list))
        
        if not broker_fields:
            field_faults = list(set(field_faults) - set(self.broker_populated_fields))
                               
        return field_faults
    
    def is_fully_filled(self):
        return self.fields['filled_qty'] == self.fields['quantity']
        
    def add_fill(self, quantity, price):
        if self.fields["filled_qty"] > 0:
            new_avg_price = (quantity*price + self.fields["filled_qty"]*self.fields["avg_fill_price"]) / \
                                (quantity + self.fields["filled_qty"])
        else:
            new_avg_price = price
                      
        self.fields["filled_qty"] = self.fields["filled_qty"] + quantity
        self.fields["avg_fill_price"] = new_avg_price
                                
    
    def apply_netted_order_fill(self, netted_order, add_filled_qty ):
        
        if add_filled_qty > 0:
            self.add_fill(add_filled_qty,netted_order.fields["avg_fill_price"])
        
        assert(self.fields["filled_qty"] <= self.fields["quantity"])
        
        # fully filled means we are complete, regardless of state of netted order
        if self.fields["filled_qty"] ==  self.fields["quantity"]:
            self.fields["state"] = Order.S_NETTED
            self.fields["execution_date"] = datetime.now()
            
        elif netted_order.fields["state"] in self.terminal_states and self.fields["filled_qty"] > 0:
            self.fields["state"] = Order.S_NETTED_PARTIAL
            self.fields["execution_date"] = datetime.now()
        
        elif netted_order.fields["state"] in self.terminal_states and self.fields["filled_qty"] == 0:
            self.fields["state"] = Order.S_NETTED_CANCELLED
        
        elif netted_order.fields["state"] in self.non_terminal_states:
            assert( self.fields["state"] == self.S_NETTED_PENDING )
            
        elif netted_order.fields["state"] in [Order.S_FILLED, Order.S_NETTED, Order.S_NETTED_PARTIAL, Order.S_NETTED_CANCELLED,
                                       Order.S_CROSS, Order.S_NETTED_PENDING]:
            self.logger.error( "Invalid state encountered when updating parent of netted order {0}. Invalid state of netted order is {1}."\
                                    .format(netted_order.fields["internal_order_id"],netted_order.fields["state"]))
        
        return
    
    @classmethod
    def allocate_parent_order_fills( cls, context, fill_qty, parent_orders):
        
        close_orders = []
        open_orders = []
        
        for order in parent_orders:
            instruction = TradeInstruction.get(context,trade_instruction_id=order.fields["trade_instruction_id"])
            if instruction[0].fields["open_order_id"] == order.fields["internal_order_id"]:
                open_orders.append(order)
            elif instruction[0].fields["close_order_id"] == order.fields["internal_order_id"]:
                close_orders.append(order)
            else:
                context[PARAM_LOGGER].warning("Parent order id {0} not tied to an active open or close order on trade instruction {1}."\
                            .format(order.fields["internal_order_id"],instruction.fields["trade_instruction_id"]))
                open_orders.append(order)
    
        fill_allocs = {}
            
        # priority is to fill close orders first
        ordering = close_orders + open_orders
        to_allocate = fill_qty
        for order in ordering:
            needed_liq = order.fields["quantity"] - order.fields["filled_qty"]
            if needed_liq >= to_allocate:
                fill_allocs[order.fields["internal_order_id"]] = to_allocate
                to_allocate = 0
            else: # needed_liq < to_allocate
                fill_allocs[order.fields["internal_order_id"]] = needed_liq
                to_allocate = to_allocate - needed_liq
                
        return fill_allocs
        


class TradeInstruction(CachedSQLObject):
    # TRADE INSTRUCTION STATUS
    S_CREATED = "CREATED"
    S_OPEN = "OPEN"
    S_PENDING_EXEC = "PENDING_EXEC"
    S_HOLDING = "HOLDING"
    S_PARTIAL_HOLDING = "PARTIAL_HOLDING"
    S_CLOSE_PENDING = "CLOSE_PENDING"
    S_CANCELLED = "CANCELLED"
    S_ABORTED = "ABORTED"           # similar to cancelled but the trade instruction can not move to any other state
    S_CLOSED = "CLOSED"
    S_PARTIAL_CLOSED = "PARTIAL_CLOSED"
    
    
    tablename = "trade_instructions"
    primary_attr = ["trade_instruction_id"]
    
    terminal_states = [S_ABORTED, S_CANCELLED, S_CLOSED]
    non_terminal_states = [S_CREATED,S_OPEN, S_PENDING_EXEC,S_HOLDING,S_PARTIAL_HOLDING,S_CLOSE_PENDING, S_PARTIAL_CLOSED]
    
    price_fields = ["limit_price","stop_price"]
    
    def __init__(self, context):
        super().__init__(context)
        self.fields = {
                                 #"trade_instruction_id" : None,   # AUTO POPULATED, DO NOT SPECIFY
                                 "algo_id" : context[PARAM_APP_NAME],
                                 "creation_date" : datetime.now(),
                                 "state" : TradeInstruction.S_CREATED,
                                 "side" : None,
                                 "quantity" : 0,
                                 "symbol" : None,
                                 "expected_notional" : 0,
                                 "open_order_id" : None,
                                 "close_out_date" : None,
                                 "close_order_id" : None,
                                 "stop_loss" : None,
                                 "stop_gain" : None,
                                 "limit_price" : None,
                                 "stop_price" : None,
                                 "notes" : None,
                                 "expire_at" : None,
                                 "activation_date" : None
                            }

    
    @classmethod
    def new(cls,context):
        new_trade = cls._new_object(context)
        
        insert_id = new_trade.insert(last_insert_id = True)
        new_trade.fields["trade_instruction_id"] = insert_id
        new_trade.fields["algo_id"] = context[PARAM_APP_NAME]

        return new_trade
      
    def _notify_object_load(self):
        for field in self.price_fields:
            if self.fields[field] is not None:
                self.fields[field] = np.round(self.fields[field],2)
        
    def _check_open_close_order_ids(self,cur_pos_qty):
        
        opening_side = self.fields['side']
        closing_side = "B" if self.fields['side'] == "S" else "S"
        
        if cur_pos_qty > 0 and closing_side == "B" or \
            cur_pos_qty < 0 and closing_side == "S":
            closing_side = "B" if closing_side == "S" else "S"
        
        def get_order_non_term_first( side ):
            orders = Order.get(self.context,trade_instruction_id=self.fields['trade_instruction_id'],side=side,
                                                state=Order.non_terminal_states)
            if len(orders) == 0:
                orders = Order.get(self.context,trade_instruction_id=self.fields['trade_instruction_id'],side=side,
                                                    state=Order.terminal_states)
        
            if len(orders) > 0:
                latest_order = None
                for order in orders:
                    if latest_order is None or \
                        order.fields['creation_date'] > latest_order.fields['creation_date']:
                        latest_order = order
                
                return latest_order

            return None

        update_self = False
        
        if opening_side != closing_side:             
            open_order = get_order_non_term_first(opening_side)
            if open_order is not None and \
                open_order.fields['internal_order_id'] != self.fields['open_order_id']:
                self.fields['open_order_id'] = open_order.fields['internal_order_id']
                update_self = True
            
        close_order = get_order_non_term_first(closing_side)
        if close_order is not None and \
            close_order.fields['internal_order_id'] != self.fields['close_order_id'] and \
            close_order.fields['internal_order_id'] != self.fields['open_order_id']:
            self.fields['close_order_id'] = close_order.fields['internal_order_id']
            update_self = True
            
        if update_self:
            self.update()

        
    def _get_correct_state(self):

        cur_pos_qty = self.get_open_position_quantity()
        self._check_open_close_order_ids(cur_pos_qty)

        if self.fields['open_order_id'] is None and \
            self.fields['state'] not in [TradeInstruction.S_CANCELLED,TradeInstruction.S_ABORTED] and \
            cur_pos_qty == 0:
            if self.fields["symbol"] is not None and \
                self.fields["symbol"] != "":
                return TradeInstruction.S_OPEN
            return TradeInstruction.S_CREATED
        
        orders = Order.get(self.context,trade_instruction_id=self.fields['trade_instruction_id'])
        working_orders = [order for order in orders if order.fields['state'] in Order.working_states]
        
        if len(working_orders) > 0:
            if self.fields['close_order_id'] is None or \
                (Order.get(self.context,internal_order_id=self.fields['close_order_id'])[0].fields['state'] == Order.S_HELD and \
                 cur_pos_qty == 0):
                return TradeInstruction.S_PENDING_EXEC

            return TradeInstruction.S_CLOSE_PENDING
        
        if cur_pos_qty == 0:
            if len([order for order in orders if order.fields['filled_qty'] > 0]) > 0:
                return TradeInstruction.S_CLOSED

            rejected_orders = [order for order in orders if order.fields['state'] == Order.S_REJECTED]
            if len(rejected_orders) == len(orders) and self.fields['state'] != TradeInstruction.S_CANCELLED:
                # if we haven't exceeded max orders then we may retry based on the last rejection reason
                if len(rejected_orders) > 0 and \
                    len(rejected_orders) < self.context[PARAM_TRADE_CANCEL_MAX_RETRY_ORDERS] and \
                    self.context[PARAM_BROKER_INTERFACE].get_reason_type(rejected_orders[-1]) == \
                            BrokerInterface.R_PRICE_MOVED_PAST_STOP:
                        return TradeInstruction.S_CANCELLED     # we can retry the TI
                return TradeInstruction.S_ABORTED

            return TradeInstruction.S_CANCELLED
        else:  
            if len([order for order in orders if order.fields['state'] in Order.working_states]) == 0:
                if cur_pos_qty == self.fields['quantity'] and self.fields['side'] == "B" or \
                    -cur_pos_qty == self.fields['quantity'] and self.fields['side'] == "S":
                    return TradeInstruction.S_HOLDING
                elif self.fields['close_out_date'] is None or \
                    self.fields['close_out_date'] > datetime.now():
                    return TradeInstruction.S_PARTIAL_HOLDING
                else:
                    return TradeInstruction.S_PARTIAL_CLOSED
                
            self.context[PARAM_LOGGER].critical("TI state should be PENDING_EXEC or CLOSE_PENDING.\n{}".format(pprint.pformat(self.fields)))    
            raise ValueError("Invalid TI state condition for {}. Should be PENDING_EXEC or CLOSE_PENDING"\
                                .format(self.fields['trade_instruction_id']))
        
        
        self.context[PARAM_LOGGER].critical("No TI state determined.\n{}".format(pprint.pformat(self.fields)))    
        raise ValueError("No TI state determined for {}".format(self.fields['trade_instruction_id']))
            
    
                      
    def update_state(self):
        
        # get current state
        cur_state = self.fields["state"]
        new_state = self._get_correct_state()
        
        if cur_state != new_state:
            self.context[PARAM_LOGGER].debug("Updating state for trade instruction {0} from {1} to {2}"\
                                                .format(self.fields["trade_instruction_id"],cur_state,new_state))            
            self.fields["state"] = new_state
            self.update()

            # if this instruction is finished, we generate PnL records            
            if new_state in self.terminal_states:
                PnlEffect.generate_pnl( self.context, self )
            
            # state updated
            return True

        # state not updated
        return False
    
    
    def get_open_position_quantity(self):
        # returns the number of shares existing as an open position related to this trade instruction
        
        trade_orders = Order.get(self.context,trade_instruction_id=self.fields["trade_instruction_id"])
        
        total_filled_qty = 0
        for order in trade_orders:
            side = order.fields["side"]
            if side is not None:
                if side == "B":
                    total_filled_qty = total_filled_qty + order.fields["filled_qty"]
                elif side == "S":
                    total_filled_qty = total_filled_qty - order.fields["filled_qty"]
        
        return total_filled_qty
    
    
    
class PnlEffect(CachedSQLObject):
    
    tablename = "pnl_effects"
    primary_attr = ["trade_instruction_id"]
    
    price_fields = ['pnl']
    
    @classmethod
    def new(cls,context,trade_instruction_id, pnl, algo_id):
        new_pnl = cls._new_object(context)
        new_pnl.fields = {
                             "trade_instruction_id" : trade_instruction_id,
                             "creation_date" : datetime.now(),
                             "algo_id" : algo_id,
                             "pnl" : pnl,
                            }
        
        new_pnl.insert()

        return new_pnl        

    def _notify_object_load(self):
        for field in self.price_fields:
            self.fields[field] = np.round(self.fields[field],2)
    
    @classmethod
    def generate_pnl(cls,context, instruction):
        
        open_orders = []
        close_orders = []
        orders = Order.get(context,trade_instruction_id=instruction.fields["trade_instruction_id"])
        for order in orders:
            if order.fields["side"] == instruction.fields["side"]:
                open_orders.append(order)
            else:
                close_orders.append(order)
        
        position = 0
        avg_price = 0
        for open_order in open_orders:
            if open_order.fields["filled_qty"] > 0:
                avg_price = (position*avg_price + open_order.fields["filled_qty"]*open_order.fields["avg_fill_price"]) / \
                                (position+open_order.fields["filled_qty"])
                position = position + open_order.fields["filled_qty"]
        
        pnl = 0
        for close_order in close_orders:
            if close_order.fields["filled_qty"] > 0:
                if close_order.fields["side"] == "S":
                    pnl = pnl + close_order.fields["filled_qty"] * (close_order.fields["avg_fill_price"] - avg_price)
                else:
                    pnl = pnl + close_order.fields["filled_qty"] * (avg_price - close_order.fields["avg_fill_price"])
        
        if pnl != 0:
            # remove existing pnl if it exists
            existing_pnl = PnlEffect.get(context,trade_instruction_id=instruction.fields["trade_instruction_id"])
            if len(existing_pnl) > 0:
                PnlEffect.delete(context,trade_instruction_id=instruction.fields["trade_instruction_id"])
                
            # add pnl
            pnleffect = PnlEffect.new(instruction.context,instruction.fields["trade_instruction_id"], np.round(pnl,2), instruction.fields['algo_id'])
            
            context[PARAM_LOGGER].debug("Generated PnL for trade instruction {0} ({1}): {2}"\
                                                .format(instruction.fields["trade_instruction_id"],
                                                        instruction.fields["symbol"],pnl))    
        else:
            pnleffect = None
            context[PARAM_LOGGER].debug("No PnL generated for trade instruction {0}"\
                                        .format(instruction.fields["trade_instruction_id"]))
        return pnleffect

        
        
            
    
    
class Position:
    
    def __init__(self, context):
        self.context = context
        self.fields = {
                            "symbol" : None,
                            "avg_price" : None,
                            "quantity" : None,
                            "market_value" : None,
                            "cost_basis" : None,
                        }




    
        