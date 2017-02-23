# penguin_2_0
PEnGUIn with Orbital Advection

To Compile:

1. Make sure CUDA is installed.

2. Create the following new directories under penguin_2_0/

  a) obj/
  
  b) binary/
  
  c) files/
  
  d) images/
  
3. Type 'make' inside penguin_2_0/ to create the executable 'penguin'

To Use:

1. Grid dimensions are defined under include/variable_types.h

2. Simulation parameters are defined under include/global.h

3. Outputs are saved in files/

4. The entire grid is periodically stored in binary/

For post-production:

A separate program is in the folder read/, which reads grids stored in binary/ and performs various analyses. It uses the mathgl library for creating images that are stored in images/.
