module unload python/3.5.3
module unload intel/17.0
module load intel/17.2
module load intelpython3/2019.1

a=300
Dt_diag=10

INPUT=$(pwd)
JOBNAME=$(echo $INPUT| cut -d'/' -f 6)

var1=s/LOOP/True/g
var2=s/RUN_NAME/$JOBNAME/g

mkdir json_files

for TIME in $(seq 150 $Dt_diag $a)
 do
  var3=s/TIME/$TIME/g
  sed -e ${var1} -e ${var2} -e ${var3} <Story_Writer.ipynb >story_writer_loop.ipynb
  jupyter nbconvert --to python story_writer_loop.ipynb
  ipython story_writer_loop.py
  rm -f story_writer_loop.ipynb
  rm -f story_writer_loop.py
 done

module unload intelpython3/2019.1
module unload intel/17.2
module load intel/17.0
module load python/3.5.3
