"""Python code modified on 2022-12-27."""
import os
import shutil
import glob
import multiprocessing
import subprocess
import numpy as np
import pandas as pd
from itertools import cycle
from matplotlib import pylab as plt
from helper_functions import dict_builder, build_input_files, input_file_paths
from functools import partial


def build_pool(multiproc=True, pool_size=None):
    """
    build_pool(multiproc = True, pool_size = None).

    if multiproc == True creates a pool using multiprocessing.Pool
    False will eventually implement a way to use the IPython 3cluster module
    instead
    'pool_size' is the number of workers to use.
    values are from 0 to n
    if None, a default value of 2 is set.
    if -1, multiprocessing.cpu_count() is set.
    """
    if not pool_size:
        pool_size = 2
    elif pool_size == -1:
        pool_size = multiprocessing.cpu_count()-1
    elif not isinstance(pool_size, int):
        raise TypeError
    if multiproc:
        # pool_size = multiprocessing.cpu_count() * 2
        pool = multiprocessing.Pool(processes=pool_size)
        return pool
    else:
        raise TypeError


def fds_calculation(input_path, **kwargs):
    """
    fds_calculation(input_path, **kwargs).

    assumes a valid input path has been passed to it. The function then does
    a subprocess call (effectivly to the command line) to then run an mpirun
    job with the flag -np X set to 'proc_per_simulation'. 
    """
    cur_dir = os.getcwd()
    (input_path, input_file) = os.path.split(input_path)
    (input_head, input_ext) = os.path.splitext(input_file)

    explicit = kwargs['explicit']
    proc_per_simulation = kwargs['proc_per_simulation']
    open_mp_size = kwargs['open_mp_size']

    os.chdir(input_path + "\\")

    if explicit:
        # FDS call by directory with explicit fds version (explict = True)
        os.environ["OMP_NUM_THREADS"] = str(kwargs['open_mp_size'])
        retcode = subprocess.call(
            [cur_dir + "//" + 'fds6',
             input_path + "\\" + input_file, '&>',
             input_path + '\\' + input_head + '.err', '&'])

    if not explicit:
        # FDS call by default installation (explict = False)
        input = 'cmd /k fdsinit && fds_local -p ' + str(proc_per_simulation) \
            + ' -o ' + str(open_mp_size) + ' ' + input_path + '\\' + input_file
        retcode = subprocess.call(input)

    os.chdir(cur_dir)
    return retcode


def main(input_file, **kwargs):
    """
    main(input_file, **kwargs).

    For parametric studies on a multi-core machine it is from advantage
    to run FDS on several cores at the same time.
    In case
    a) of one mesh one can perform several jobs
    (corresponding to the number of existing parameter study runs)
    b) of multiple meshs one can perform one or just a few jobs
    on the available cores at the same time.

    The maximum number of possible processes are set with the variable
    "pool_size". This should be smaller than the physically existing cores.
    The number of available for openMP THREADS (meaning cores here)
    can be set with 'open_mp_size'.
    """
    IOoutput, param_dict = build_input_files(input_file,
                                             base_path=kwargs['base_path'])
    paths = input_file_paths(kwargs['base_path'])
    pool = build_pool(multiproc=kwargs['multiproc'],
                      pool_size=kwargs['pool_size'])
    """
    map(funct, iterable).

    apply 'funct' to each element in `paths`, collecting the results
    in a list that is returned.
    """
    try:
        pool_outputs = pool.map(partial(kwargs['funct'], **kwargs), paths)
    except ValueError:
        print("Oops! That was no valid input. Try again...")
        print(kwargs['funct'], " Input fault")
    finally:
        print(pool_outputs)

    pool.close()
    return pool_outputs, param_dict


def plotter(parameters, plotted_val='HRR', plotted_file='*_hrr.csv', **kwargs):
    """
    plotter(parameters, plotted_val='HRR', plotted_file='*_hrr.csv', **kwargs).

    takes in a parameter set and a plotted_val (for now a column label in the
    FDS output) reads the data in, and then plots all grouping variations of
    the plotted_val as a function of the parameter study variables.
    """
    dataLists = {}
    folderlist = []
    lines = ['-', '--', ':', '-.']
    # read data
    for folder in glob.glob(os.path.join(kwargs['base_path'] + "//", '*')):
        for datafile in glob.glob(os.path.join(folder + "//", plotted_file)):
            dataLists[
                folder.split("\\")[-1]] = pd.read_csv(datafile, skiprows=1)
            folderlist = np.append(folderlist, folder.split("\\")[-1])

    filename_map = pd.DataFrame(
        list(dict_builder(parameters, test_name=kwargs['test_name'] + " ")))
    for key in parameters.keys():
        # print ('key=', key)
        for results in filename_map.groupby(by=key):
            plt.figure()
            # print("result in plotter = ", results)
            linecycler = cycle(lines)
            for title in results[1]['TITLE'].values:
                i = results[1][results[1]['TITLE'] == title].index[0]
                fall = folderlist[i]
                # plt.plot(dataLists[fall].Time,
                # dataLists[fall][plotted_val], label = title)
                plt.plot(dataLists[fall].Time, dataLists[fall][plotted_val],
                         next(linecycler), label=title)
                # plt.title(plotted_val + ' : ' + key + ' ' + str(results[0]))
                plt.title(plotted_val + ' : ' + str(list(parameters.keys())))
                # plt.legend(loc=0, title=str(list(parameters.keys())))
                plt.legend(loc=0)
                plt.savefig(kwargs['base_path'] + "//" + plotted_val
                            + '_' + key + ' ' + str(results[0]) +
                            '.png', dpi=300)
                i = i + 1


if __name__ == '__main__':
    '''
    'test_name': string
        e.g. 'Mesh 10' a name that serves as an indicator in the legend
        for plot sof the study that was conducted
    'base_path': string
        e.g. 'output_files',  output directory. The results of the
        calculations and the figures created by the plotter routine
        will stored in subdirectories for each run
    'funct': fds_calculation
        # other functions are possible but have to coded
        in case of 'explicit' equal 'False' fds is called by standard
        (via "fds_local") procedure
    'version': string
        in case of 'explicit' equal True version has set to the executable
        fds version listed in the main directory (e.g. fds6.exe),
        works with older versions of fds
    'explicit': boolean
        True: 'fds.exe' is given explicit in the default directory
        False: 'fds.exe' is called by standard (via "fds_local") procedure
    'multiproc': boolean
        True: several processes possible at the same time
        False: only one process at the same time
    'pool_size': integer
        maximal number of cores/threads that are available
    'open_mp_size': integer
        number is equal to the cores/threads that are uesed for OpenMP
    'proc_per_simulation': integer
        number of cores/threads required for each job (cause of several meshs)
    '''

    input_file = 'example_input_file.fds'
    kwargs = {'test_name': 'Para 10:',
              'base_path': 'output_files',
              'funct': fds_calculation,
              'version': 'fds6',
              'explicit': False,
              'multiproc': True,
              'pool_size': 3,
              'open_mp_size': 2,
              'proc_per_simulation': 1}

    calling_dir = os.getcwd()
    if os.path.exists(os.path.join(calling_dir, kwargs['base_path'])):
        shutil.rmtree(os.path.join(calling_dir, kwargs['base_path']))

    pool_outputs, param_dict = main(input_file, **kwargs)

    plotter(param_dict, plotted_val='T_05',
            plotted_file='*_devc.csv', **kwargs)
    plotter(param_dict, plotted_val='T_35',
            plotted_file='*_devc.csv', **kwargs)
    plotter(param_dict, plotted_val='T_65',
            plotted_file='*_devc.csv', **kwargs)
