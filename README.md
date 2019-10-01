Certainpy - a library to deal with uncertainty in all your calculations!
========================================================================

What is this?
-------------

This library provides two basic classes: an "Uncertain_Value" class that represents a number with
uncertainty, for example something like 3 +- 0.1, and an associated "Vec" class that represents a 2D
vector where each co-ordinate has an associated, independent uncertainty.

These classes then have a lot of the standard operations you expect from floats (basic arithmetic
operations, exponentiation, and arbitrary function execution) and vectors (addition, subtraction,
conversion between Polar and Cartestian) implemented (the Vec class needs a lot of work - any help 
would be much appreciated). 

How do I use this?
------------------

Simply `from certainpy import *`.

And from there you can create your values and start doing whatever you like with them:
![Using the package](/readme/images/using-the-package.png)


Using this project
------------------

This project is under the GNU GPL 3, with no warranty or fitness for any particular purpose. See 
LICENSE.txt for details. Don't hesitate to get in touch for anything!
