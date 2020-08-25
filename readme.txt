These are the files used to read data from Lathys results after executing "diag".

On a distant supercomputer, we have files of the type:
	- Magw_*.nc
	- Hsw_*.nc
	- Thew_*.nc
	- Elew_*.nc

There is one of each for every time dump, which for big simulations can be very memory consuming.

The above Story_Writer and Module_Diagnostics strive to extract the most relevant information from these ncfiles. The results are stored in a .json file, and some .png files. The Story_Reader script uses the data in the .json file to generate ready-to-use plots.

