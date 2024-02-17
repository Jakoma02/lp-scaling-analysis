# Experimental Analysis of Scaling Methods for LP

This repository contains code and data for paper "Experimental Analysis of LP Scaling
Methods Based on Circuit Imbalance Minimization" by Jakub Komárek and Martin Koutecký
of the Computer Science Institute of the Charles University, Prague. For explanation
of the research objective, see the paper contents first.

The main part of the implementation code is the `circuit_ineq.py` library, containing
the whole rescaling algorithm. Individual scripts in the `scripts` directory then
utilize this library to compute rescalings for MIPLIB/Netlib instances and to measure
how long solvers run before and after applying the rescaling.

The `data` directory contains results that we obtained by running the algorithm.
In the `rescaling` subdirectory, there are the rescaling vectors found by our implementation.
These vectors are in the form of SageMath vectors serialized by `pickle`. For an example
of how to work with the rescalings, see the `apply_prescaling.py` script.

In the `instances` subdirectory, there are MPS files for all successfully rescaled
instances in all original, rescaled and rescaled by powers of two forms, all
after performing relaxation and converting to the semi-standard form.

In the `solver_timings` subdirectory reside the results of measurements of solver
run times for all the problems in Netlib and MIPLIB that were feasible to run
the rescaling algorithm for. Results are separated to several JSON files by the used
solver and every file contains measurement results for all the problems and
instance variants (original/rescaled/rescaled by powers of 2).

If you are interested, don't hesitate to contact us at {komarek,koutecky}@iuuk.mff.cuni.cz.
