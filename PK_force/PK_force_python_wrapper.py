'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.

! Fortran90 subroutine to be used in Python
! calculate Peach-Koehler force on dislocation configuration
! will be embedded via the fmodpy wrapper
'''

import os
import ctypes
import platform
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "PK_force." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['PK_force.f90', 'PK_force_c_wrapper.f90']
_symbol_files = []# 
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the prerequisite symbols for the compiled code.
for _ in _symbol_files:
    _ = ctypes.CDLL(os.path.join(_this_directory, _), mode=ctypes.RTLD_GLOBAL)
# Try to import the existing object. If that fails, recompile and then try.
try:
    # Check to see if the source files have been modified and a recompilation is needed.
    if (max(max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _symbol_files]),
            max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _ordered_dependencies]))
        > os.path.getmtime(_path_to_lib)):
        print()
        print("WARNING: Recompiling because the modification time of a source file is newer than the library.", flush=True)
        print()
        if os.path.exists(_path_to_lib):
            os.remove(_path_to_lib)
        raise NotImplementedError(f"The newest library code has not been compiled.")
    # Import the library.
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = [_fort_compiler] + _ordered_dependencies + _compile_options + ["-o", _shared_object_name]
    if _verbose:
        print("Running system command with arguments")
        print("  ", " ".join(_command))
    # Run the compilation command.
    import subprocess
    subprocess.check_call(_command, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


# ----------------------------------------------
# Wrapper for the Fortran subroutine CALC_FPK_PBC

def calc_fpk_pbc(xpos, ypos, bx, by, tau0, len_x, len_y, nmob, n, fpk=None):
    '''! Solution based on Eqs (2.1.25a) and (2.1.25b) from Linyong Pang "A new O(N) method for
! modeling and simulating the behavior of a large number of dislocations in
! anisotropic linear elastic media", PhD thesis, Stanford University, USA. 2001'''
    
    # Setting up "xpos"
    if ((not issubclass(type(xpos), numpy.ndarray)) or
        (not numpy.asarray(xpos).flags.f_contiguous) or
        (not (xpos.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'xpos' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        xpos = numpy.asarray(xpos, dtype=ctypes.c_double, order='F')
    xpos_dim_1 = ctypes.c_long(xpos.shape[0])
    
    # Setting up "ypos"
    if ((not issubclass(type(ypos), numpy.ndarray)) or
        (not numpy.asarray(ypos).flags.f_contiguous) or
        (not (ypos.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ypos' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ypos = numpy.asarray(ypos, dtype=ctypes.c_double, order='F')
    ypos_dim_1 = ctypes.c_long(ypos.shape[0])
    
    # Setting up "bx"
    if ((not issubclass(type(bx), numpy.ndarray)) or
        (not numpy.asarray(bx).flags.f_contiguous) or
        (not (bx.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'bx' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        bx = numpy.asarray(bx, dtype=ctypes.c_double, order='F')
    bx_dim_1 = ctypes.c_long(bx.shape[0])
    
    # Setting up "by"
    if ((not issubclass(type(by), numpy.ndarray)) or
        (not numpy.asarray(by).flags.f_contiguous) or
        (not (by.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'by' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        by = numpy.asarray(by, dtype=ctypes.c_double, order='F')
    by_dim_1 = ctypes.c_long(by.shape[0])
    
    # Setting up "tau0"
    if (type(tau0) is not ctypes.c_double): tau0 = ctypes.c_double(tau0)
    
    # Setting up "len_x"
    if (type(len_x) is not ctypes.c_double): len_x = ctypes.c_double(len_x)
    
    # Setting up "len_y"
    if (type(len_y) is not ctypes.c_double): len_y = ctypes.c_double(len_y)
    
    # Setting up "fpk"
    if (fpk is None):
        fpk = numpy.zeros(shape=(2, nmob), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(fpk), numpy.ndarray)) or
          (not numpy.asarray(fpk).flags.f_contiguous) or
          (not (fpk.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'fpk' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        fpk = numpy.asarray(fpk, dtype=ctypes.c_double, order='F')
    fpk_dim_1 = ctypes.c_long(fpk.shape[0])
    fpk_dim_2 = ctypes.c_long(fpk.shape[1])
    
    # Setting up "nmob"
    if (type(nmob) is not ctypes.c_int): nmob = ctypes.c_int(nmob)
    
    # Setting up "n"
    if (type(n) is not ctypes.c_int): n = ctypes.c_int(n)

    # Call C-accessible Fortran wrapper.
    clib.c_calc_fpk_pbc(ctypes.byref(xpos_dim_1), ctypes.c_void_p(xpos.ctypes.data), ctypes.byref(ypos_dim_1), ctypes.c_void_p(ypos.ctypes.data), ctypes.byref(bx_dim_1), ctypes.c_void_p(bx.ctypes.data), ctypes.byref(by_dim_1), ctypes.c_void_p(by.ctypes.data), ctypes.byref(tau0), ctypes.byref(len_x), ctypes.byref(len_y), ctypes.byref(fpk_dim_1), ctypes.byref(fpk_dim_2), ctypes.c_void_p(fpk.ctypes.data), ctypes.byref(nmob), ctypes.byref(n))

    # Return final results, 'INTENT(OUT)' arguments only.
    return fpk


# ----------------------------------------------
# Wrapper for the Fortran subroutine CALC_FPK

def calc_fpk(xpos, ypos, bx, by, tau0, nmob, n, fpk=None):
    ''''''
    
    # Setting up "xpos"
    if ((not issubclass(type(xpos), numpy.ndarray)) or
        (not numpy.asarray(xpos).flags.f_contiguous) or
        (not (xpos.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'xpos' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        xpos = numpy.asarray(xpos, dtype=ctypes.c_double, order='F')
    xpos_dim_1 = ctypes.c_long(xpos.shape[0])
    
    # Setting up "ypos"
    if ((not issubclass(type(ypos), numpy.ndarray)) or
        (not numpy.asarray(ypos).flags.f_contiguous) or
        (not (ypos.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ypos' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ypos = numpy.asarray(ypos, dtype=ctypes.c_double, order='F')
    ypos_dim_1 = ctypes.c_long(ypos.shape[0])
    
    # Setting up "bx"
    if ((not issubclass(type(bx), numpy.ndarray)) or
        (not numpy.asarray(bx).flags.f_contiguous) or
        (not (bx.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'bx' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        bx = numpy.asarray(bx, dtype=ctypes.c_double, order='F')
    bx_dim_1 = ctypes.c_long(bx.shape[0])
    
    # Setting up "by"
    if ((not issubclass(type(by), numpy.ndarray)) or
        (not numpy.asarray(by).flags.f_contiguous) or
        (not (by.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'by' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        by = numpy.asarray(by, dtype=ctypes.c_double, order='F')
    by_dim_1 = ctypes.c_long(by.shape[0])
    
    # Setting up "tau0"
    if (type(tau0) is not ctypes.c_double): tau0 = ctypes.c_double(tau0)
    
    # Setting up "fpk"
    if (fpk is None):
        fpk = numpy.zeros(shape=(2, nmob), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(fpk), numpy.ndarray)) or
          (not numpy.asarray(fpk).flags.f_contiguous) or
          (not (fpk.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'fpk' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        fpk = numpy.asarray(fpk, dtype=ctypes.c_double, order='F')
    fpk_dim_1 = ctypes.c_long(fpk.shape[0])
    fpk_dim_2 = ctypes.c_long(fpk.shape[1])
    
    # Setting up "nmob"
    if (type(nmob) is not ctypes.c_int): nmob = ctypes.c_int(nmob)
    
    # Setting up "n"
    if (type(n) is not ctypes.c_int): n = ctypes.c_int(n)

    # Call C-accessible Fortran wrapper.
    clib.c_calc_fpk(ctypes.byref(xpos_dim_1), ctypes.c_void_p(xpos.ctypes.data), ctypes.byref(ypos_dim_1), ctypes.c_void_p(ypos.ctypes.data), ctypes.byref(bx_dim_1), ctypes.c_void_p(bx.ctypes.data), ctypes.byref(by_dim_1), ctypes.c_void_p(by.ctypes.data), ctypes.byref(tau0), ctypes.byref(fpk_dim_1), ctypes.byref(fpk_dim_2), ctypes.c_void_p(fpk.ctypes.data), ctypes.byref(nmob), ctypes.byref(n))

    # Return final results, 'INTENT(OUT)' arguments only.
    return fpk

