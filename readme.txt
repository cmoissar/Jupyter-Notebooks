These are the files used to read data from Lathys results after executing "diag".

On a distant supercomputer, we have files of the type:
	- Magw_*.nc
	- Hsw_*.nc
	- Thew_*.nc
	- Elew_*.nc

There is one of each for every time dump, which for big simulations can be very memory consuming.

The above Story_Writer and Module_Diagnostics strive to extract the most relevant information from these ncfiles. The results are stored in a .json file, and some .png files. The Story_Reader script uses the data in the .json file to generate ready-to-use plots.

/!\ Warning. 
Story_Writer.ipynb and Story_Writer.py (idem with Module_Diagnostics.*) might be different. 
To avoid losing any work, you should: 
"mv Story_Writer.py sav_Story_Writer.py; jupyter nbconvert --to python Story_Writer.ipynb; meld Story_Writer.py sav_Story_Writer.py"
Make sure to update manually Story_Writer.ipynb to include the best work.
Same thing with Module_Diagnostics.py and Module_Diagnostics.ipynb.
This is quite lousy versioning still, sorry.
