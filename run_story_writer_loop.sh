a=300
Dt_diag=10

var1=s/LOOP/True/g

for TIME in $(seq 0 $Dt_diag $a)
 do
  var2=s/TIME/$TIME/g
  sed -e ${var1} -e ${var2} <Story_Writer.ipynb >story_writer_loop.ipynb
  ipython story_writer_loop.ipynb
  rm -f story_writer_loop.ipynb
 done

