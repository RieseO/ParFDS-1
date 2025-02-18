The code is a number of files plus an example FDS input file. 

 - parFDS.py - the code which orchestrates the calls to FDS and wrapping
 - example_input_file.fds - an example modified FDS input template
 
Update history: Oriese@ 2022-12-27
 - includes plotter routines which plot the results for all parameters in comparison
 - code has been optimized to work with newer fds start routines "cmdfds" and "fds_local"

Most of the important stuff takes place after the if __name__ == '__main__' in ParFDS.py 

Here's a walkthrough: 

The first line which defines 'input_file' tells the code where to look for the template input file.

Parameters are defined via an augmented FDS syntax which is parsed in the code base. Variables in this syntax for these parametric studies are {Variable_Name SWEEP First_Value, Second_Value, Nubmer_Steps}, where SWEEP is a protected word which lets ParFDS know that we will be attempting a linear parametric study in that variable.

The 'kwargs', dictionary sets things like: 

 - 'test_name': string - e.g. 'Para 10' a name that serves as an indicator in the legend for the plots of the study that was conducted
 - 'base_path': string - e.g. 'output_files',  the name of the base folder which will contain the folders and input files generated by the code. The results of the calculations and the figures created by the plotter routine will stored in subdirectories for each run (assumed to be under the folder containing parFDS.py)
 - 'funct': fds_calculation - the actual subprocess call for fds_mpi (please see this function if you want to turn mpi on or off or set the -np flag for mpirun) if you are not using mpi it is a trivial change on the subprocess.call line. Other functions are possible but have to coded in case of 'explicit' equal 'False' fds is called by standard (via "fds_local") procedure
 - 'version': string - in case of 'explicit' equal True version has set to the executable fds version listed in the main directory (e.g. fds6.exe), works with older versions of fds
 - 'explicit': boolean -
    True: 'fds.exe' is given explicit in the default directory,
    False: 'fds.exe' is called by standard (via "fds_local") procedure
 - 'multiproc': boolean -
    True: several processes possible at the same time (is currently set to True, creating the multiprocessing pool used throughout),
    False: only one process at the same time (not tested in detail) 
 - 'pool_size': integer -
    maximal number of cores/threads that are available in the pool
 - 'open_mp_size': integer - number is equal to the cores/threads that are uesed for OpenMP
 - 'proc_per_simulation': integer - number of cores/threads required for each job (cause of several meshs)

Note, that if you use mpirun -np X and set pool_size to Y, the number of processes you will have running is X*Y at any one time as the code will spin up Y workers, which will each make an mpirun call, spinning up X processes. 

The code is then run using the main function, with the syntax:

main(input_file, parameters, **kwargs)

One more thing: in the example_input_file.fds example file, certain of the variables are in curley braces eg. {COLD_TEMP SWEEP 20, 20, 1}. That is actually used for the variable replacement in Python, which occurs in build_input_files() in helper_functions.py. The point here is that if the key 'STEP_WMAX' exists in the parameters dictionary, and also exists in the FDS input file in the form {COLD_TEMP SWEEP 20, 20, 1}, the code will know to replace that string with the appropriate numerical value when it generates the input files. 

Functional Note: the tests of this code have only been run under OS X and Ubuntu, so I am not sure at all if this code will balk on Windows. If you are going to use this on Windows, please ensure (at a minimum) that the command 'nosetests ./tests/*.py' run at the command prompt passes first. 

Code works with Windows 10, tested not with OS X and Ubuntu.
