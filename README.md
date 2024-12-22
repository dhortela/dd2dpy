# dd2dpy

### A python-based 2D dislocation dynamics code

  - Author: Daniel Hortelano-Roig
  - Organization: University of Oxford, UK
  - Contact: <daniel.hortelanoroig@gmail.com>

Note that dd2dpy is built upon (and is an extension/modification of) [pyLabDD](https://github.com/AHartmaier/pyLabDD), which is a python-based code written by Alexander Hartmaier.

Dislocation dynamics (DD) is a continuum model of material plasticity on the order of micrometres which treats dislocation line defects explicitly as discretised segments embedded within an elastic medium. The purpose of these models is to study the evolution of dislocations in materials and establish connections between this microscopic behaviour to macroscopic properties, such as stress-strain constitutive relationships.

Compared to 3D DD codes, 2D DD is computationally faster and inherently flexible with respect to boundary conditions and the inclusion of microstructures, hence can be used to solve more complicated boundary value problems that may otherwise be inaccessible to 3D methods. In 2D DD, dislocations are modelled as infinitely long, straight, edge-type dislocations with line direction perpendicular to the plane of the simulation.

dd2dpy is a 2D DD code originally developed for educational purposes, specifically for use in a summer school program. Amongst other things, it is capable of modelling Frank-Read sources and their nucleation, finite domains wherein dislocations are allowed to exit (ignoring image forces), and has the capacity to couple to FEM via the package [pyLabFEA](https://github.com/AHartmaier/pyLabFEA), which is also written by Alexander Hartmaier.

## Installation

The dd2dpy package requires an [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment with a recent Python version.

The repository can be cloned and installed locally. It is recommended to create a conda environment before installation. This can be done by the following the command line instructions

```
$ git clone https://github.com/dhortela/dd2dpy.git ./dd2dpy
$ cd dd2dpy
$ conda env create -f envdd2dpy.yml  
$ conda activate dd2dpy
$ python -m pip install . [--user]
```

The correct implementation can be tested with

```
$ pytest tests
```

After this, the package can be used within python, e.g. be importing the entire package with

```python
import pylabdd as dd
```

## Speedup with Fortran subroutines

The subroutines to calculate the Peach-Koehler (PK) force on dislocations are rather time consuming. A Fortran implementation of these subroutines can bring a considerable seepdup of the simulation. To install these faster subroutines, a Fortran compiler is required, e.g. gfortran. On MacOS, this can be achived by installing the command line tools with `xcode-select --install`. The embedding of the Fortran subroutines into Python is accomplished with the lightweight Fortran wrapper [fmodpy](https://pypi.org/project/fmodpy/).

To activate the Fortran subroutines, issue the command

```
$ python setup_fortran.py
```

After that, the package can be used in the usual way. You can control the success of the Fortran implementation by

```
$ python
>>> import pylabdd
```

This will inform you if the standard Python or the faster Fortran subroutines to calculate the PK force will be used.

## Tutorials

In the subfolder `notebooks`, tutorials are available via Jupyter notebooks on how to use dd2dpy, the dislocation dynamics method, and the Taylor hardening model.

Fully self-contained Google Colab notebooks, based on the Jupyter notebooks, are also available. These notebooks require no downloads, offering a convenient and accessible way to explore the material. Links:
  - [DD session 1](https://colab.research.google.com/drive/1I1ORC8PAWCpM8HQzW3tEAf1A924DFuD6#scrollTo=f02d9130-3198-4900-a382-9056897901f7)
  - [DD session 2](https://colab.research.google.com/drive/1tMaFwEGd27zwYbs8XHqfPTCeOotDaipy#scrollTo=CCE3IMWKpmKG)
  - [DD session 3](https://colab.research.google.com/drive/1i0TPtc9gJwDNPm9BSH155uJEfJ6Mcp9J#scrollTo=CCE3IMWKpmKG)
  - [Full DD session](https://colab.research.google.com/drive/1w63SGFggTDcwUbPM6m8LtE3RuWWZ66RC#scrollTo=f02d9130-3198-4900-a382-9056897901f7)
  - [FEM session](https://colab.research.google.com/drive/1_Zb-W7mjuLjIyOC_xDnqaGa-0zTh-Zff#scrollTo=o4UXsXWGTejT) (tutorial of [pyLabFEA](https://github.com/AHartmaier/pyLabFEA))
  - [Coupled DD+FEM session](https://colab.research.google.com/drive/1G5okYitqq5SvVCaGI-zf1om0v25BCESr#scrollTo=LAzTcysNxsG4) (coupling with [pyLabFEA](https://github.com/AHartmaier/pyLabFEA))

## Dependencies

dd2dpy requires the following packages as imports:

 - [NumPy](http://numpy.scipy.org) for array handling
 - [MatPlotLib](https://matplotlib.org/) for graphical output
 - [fmodpy](https://pypi.org/project/fmodpy/) for embedding of faster Fortran subroutines for PK force calculation (optional)

## License

The dd2dpy package is based off of [pyLabDD](https://github.com/AHartmaier/pyLabDD), which comes with ABSOLUTELY NO WARRANTY. This is free
software, and you are welcome to redistribute it under the conditions of
the GNU General Public License
([GPLv3](http://www.fsf.org/licensing/licenses/gpl.html))

The contents of the examples and notebooks are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
([CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/))
