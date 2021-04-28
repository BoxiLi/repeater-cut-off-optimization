import logging
import os
import warnings
from datetime import datetime
import json
import copy
import pickle
import time

import numpy as np

logging.EXP = logging.INFO - 1
logging.TEST = 25

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    try:
        stage = printProgressBar.stage
    except:
        printProgressBar.stage = 0.
        stage = printProgressBar.stage
    ratio = iteration / float(total)
    if total > 10 and ratio < stage - 1.0e-5:
        return
    printProgressBar.stage += 0.1
    percent = ("{0:." + str(decimals) + "f}").format(100 * ratio)
    print(prefix, percent + "% Complete" + suffix)
    # Print New Line on Complete
    if iteration == total: 
        print()


def log_init(log_prefix, level=logging.WARN):
    """
    If level<=EXP, Initialze the log file with the current time as ID.
    Otherwise only retrun ID
    """
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    date = datetime.now()
    id = log_prefix + "-" + date.strftime('%Y%m%d-%H%M%S')
    if os.path.isfile("logging/" + id + '.log'):
        warnings.warn("ID repeatition.")
        id += "r"
    if level == logging.EXP:
        logging.basicConfig(filename="logging/" + id + '.log', level=level)
    else:
        logging.basicConfig(level=level)
    logging.info("ID: " + id)
    logging.info("Date: " + date.strftime('%Y-%m-%d %H:%M:%S') + "\n")
    return id


def log_params(parameters):
    paramstr = "Parameters:\n"
    for name, value in parameters.items():
        paramstr += name + " " + str(value) + "\n"
    logging.info(paramstr)
    

def log_finish(id, parameters, remark=""):
    """
    Save the parameters and 
    """
    parameters = copy.deepcopy(parameters)
    parameters["ID"] = id
    date = datetime.now()
    parameters["finishing date"] = str(date.date())
    parameters["finishing time"] = str(date.time())
    parameters["remark"] = remark
    if logging.getLogger().getEffectiveLevel() == logging.EXP:
        logging.info("Saving parameters into record...")
        with open("logging/logging_record.json", "a") as write_file:
            try:
                json.dump(parameters, write_file)
                write_file.write("\n")
            except Exception as err:
                logging.exception(
                    "Failed to save heading data into the record.\n" +
                    str(parameters))

    logging.info("Simulation finished.")


def save_data(id, data):
    if logging.getLogger().getEffectiveLevel() != logging.EXP:
        return
    logging.info("Saving data...")
    try:
        data_address = "data/" + id + ".pickle"
        outfile = open(data_address, 'wb')
        pickle.dump(data, outfile)
        outfile.close()
        logging.info("Data saved at " + '"' + data_address + '"')
    except Exception as err:
        logging.exception("Failed to save data")


def load_data(id):
    infile = open("data//" + id + ".pickle", "rb")
    data = pickle.load(infile)
    infile.close()
    return data


def create_iter_kwargs(parameters):

    from itertools import product
    names = parameters.keys()
    values = [value if isinstance(value, list) else [value]
                for value in list(parameters.values())]

    kwarg_list = []
    for param_list in product(*values):
        param_dict = {}
        for i, key in enumerate(names):
            value = param_list[i]
            param_dict[key] = value
        kwarg_list.append(param_dict)
    return kwarg_list


def find_record_id(ID):
    patterns = {"ID":ID}
    result = find_record_patterns(patterns)
    if not result:
        return None
    else:
        return result


def find_record_patterns(patterns):
    with open("logging/logging_record.json") as log:
        record_list = log.readlines()
    matched_records = []
    if patterns is not None:
        for record in record_list:
            record = json.loads(record)
            match = True
            for key in patterns.keys():
                try:
                    if record[key] != patterns[key]:
                        match = False
                        continue
                except KeyError:
                    match = False
                    continue
            if match:
                matched_records.append(record)
    for parameters in matched_records:
        try:
            parameters["protocol"] = tuple(parameters["protocol"])
        except KeyError:
            pass
    if len(matched_records) == 1:
        return matched_records[0]
    else:
        return matched_records


def mytimeit(func):
    """
    Decorator for automatically calculate the elapsed time.
    """
    def inner(*arg, **kwargs):
        start_time = time.time()
        result = func(*arg, **kwargs)
        end_time = time.time()
        print("Elapse time: {}\n".format(end_time - start_time))
        return result
    return inner


def clear_junk_data():
    for filename in os.listdir("./logging"):
        if filename.endswith(".log"):
            id_end = filename.rfind('.')
            ID = filename[:id_end]
            result = find_record_id(ID)
            if result is None:
                try:
                    os.remove("./logging/" + ID + ".log")
                except:
                    pass
    for filename in os.listdir("./data"):
        if filename.endswith(".pickle"):
            id_end = filename.rfind('.')
            ID = filename[:id_end]
            result = find_record_id(ID)
            if result is None:
                try:
                    os.remove("./data/" + ID + ".pickle")
                except:
                    pass