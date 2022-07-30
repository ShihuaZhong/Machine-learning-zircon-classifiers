This repository contains data and source code used in the manuscript [“Earth’s earliest continental crust is not dominated by the tonalite–trondhjemite–granodiorite (TTG) series”] submitted to PNAS on 28th June, 2022, by Shihua Zhong, Sanzhong Li, Yang Liu, Peter A. Cawood and Reimar Seltmann. All code by the authors is released under an [MIT open-source license](LICENSE)

CONTENTS:

[Data]: This directory contains compiled zircons compositions from I-type, S-type and TTG rocks worldwide. For each zircon analyses, 17 variables—including 11 REEs (Ce, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, and Lu), Th, U and 4 derived trace element ratios (Th/U, U/Yb, Ce/Ce* and Eu/Eu*)—were provided in a csv format (Zircons.csv). The reference sources of all these data can be found in online SI Appendix of this paper.

[Code]: This directory contains a program in python format. The program includes three parts: parameter optimization, dataset training, and prediction of zircon types.



INSTALLATION:

To use the included python code, you can directly double-click the python file. Then modify the path of the corresponding data set and click Run to get the prediction result.