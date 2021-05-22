"""
Helper functions to retrieve signals from the .iqw files.
"""

import os
import sys
import glob
import time
import timeit
from glob import glob as ls
import pickle
import http.client
import json
import pandas as pd
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import closing

SIGNALS_DIR = "../Public_Data/"
SIGNALS2_DIR = "../Public Data Finals Signals/"
SAMPLES_DIR = '../Public Finals Samples/'


if (os.name == "nt"):
       filesep = '\\'
else:
       filesep = '/'


def vprint(mode, t):
    #Print to stdout, only if in verbose mode
    if(mode):
            print(t)
def mkdir(d):
    #Create a new directory
    if not os.path.exists(d):
        os.makedirs(d)

def zipdir(archivename, basedir):
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                if fn[-4:]!='.zip':
                    absfn = os.path.join(root, fn)
                    zfn = os.path.basename(absfn) #XXX: relative path
                    z.write(absfn, zfn)

def inventory_data(input_dir,verbose, *args):
    #Inventory the datasets in the input directory and return a signal data  
    vprint(verbose,"Reading "+ input_dir ) 
    signal_array = []
    data_characteristics = None


    if verbose:
        columns = [ "Symbol Rate (MHz)", "Modulation Type", "Modulation Order"]
        characteristics = []
        symbol_rates = [5,21.26, 25, 13.11, 30.73, None, None,None, None, 5, None,None, 12.05, None, None, None, 1.52, None, None]
        modulation_types = ["PSK","PSK", "QAM", "QAM", "QAM", "PSK", None,None,"PSK", "PSK", None,"PSK", "QAM", None, None, None,
                            "QAM", None, "PSK"]
        modulation_order = ["BPSK","QPSK", "QAM16", "QAM32", "QAM64", "BPSK", None, None, "QPSK", "BPSK", None,"QPSK","QAM64", None,
                            None, None, "QAM16", None, "QPSK"]

        for index in range(len(symbol_rates)):
            characteristics.extend([[symbol_rates[index],modulation_types[index], modulation_order[index]]])

    # Assume first that there is a hierarchy dataname/dataname_train.data
    input_names = ls(os.path.join(input_dir, '*.iqw'))
    input_names = sorted(input_names)
    if len(input_names) == 0:
        print('WARNING: Inventory data - No data file found')

    for i in range(0, len(input_names)):
        name = input_names[i]
        file = open(name, "rb")
        signal_array_ = np.fromfile(file, dtype="float32")
        signal_array.append(signal_array_)

    if verbose:
        data_characteristics = pd.DataFrame(data = np.asarray(characteristics), index = np.arange(len(signal_array)),columns = columns)

    vprint(verbose, "Number of signals " + str(len(signal_array)))
    vprint(verbose, "Length of a signal " + str(len(signal_array[0])))

    return signal_array,data_characteristics

def get_set_one(verbose=False):
       i,d = inventory_data(SIGNALS_DIR, verbose=False)
       return i
def get_set_two(verbose=False, cnstln=False):
       i,d = inventory_data(SIGNALS2_DIR, verbose=False)
       if cnstln:
            return i[:5]
       else:
            return i[5:]

# example call:
#         I_results, Q_results = load_samples('sig0')
def load_samples(SIG_INDEX, filename=None):
    if filename is None:
        filename = 'sig' + str(SIG_INDEX)
    path = SAMPLES_DIR + filename + '.npy'
    data = np.load(path)
    I_results = np.real(data)
    Q_results = np.imag(data)
    return I_results, Q_results

def get_symbol_rate(SIG_INDEX, set_two=False):
    if not set_two:
        dictionary = {0:5, 1:21.26, 2:25, 3:13.11, 4:30.73, 5:7.79, 6:15.66, 7:15.75, 8:8.23, 9:5, 10:20.0, 11:17.82, 12:12.05, 13:33.2, 14:5.0, 15:5.0, 16:1.52}
        symbol_rate = dictionary.get(SIG_INDEX, None)
    else:
        dictionary = {0:40, 1:40, 2:31.25, 3:12.5, 4:25}
        symbol_rate = dictionary.get(SIG_INDEX, None)
    if symbol_rate is not None:
        symbol_rate *= 1e6
    return symbol_rate


def write_result(output_path, verbose, symbol_rate_guess, modulation_type_guess, modulation_order_guess, EVM):
    with open(output_path, "w") as result_file:
        vprint(verbose, "(answer.txt) symbole_rate_guess: " + str(symbol_rate_guess))
        result_file.write(str(symbol_rate_guess) + "\n")
        vprint(verbose,"(answer.txt) modulation_type_guess: " + str(modulation_type_guess))
        result_file.write(str(modulation_type_guess) + "\n")
        vprint(verbose, "(answer.txt) modulation_order_guess: " + str(modulation_order_guess))
        result_file.write(str(modulation_order_guess) + "\n")
        vprint(verbose, "(answer.txt) EVM: " + str(EVM))
        result_file.write(str(EVM) + "\n")
    return

