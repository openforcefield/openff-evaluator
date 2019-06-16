pmemd.cuda -O -p cb6-but-dum.prmtop -ref cb6-but-dum.rst7 -c cb6-but-dum.rst7 -i minimize.in -o minimize.out -r minimize.rst7 -inf minimize.info -e minimize.mden
pmemd.cuda -O -p cb6-but-dum.prmtop -ref minimize.rst7 -c minimize.rst7 -i equilibration.in -o equilibration.out -r equilibration.rst7 -inf equilibration.info -e equilibration.mden
pmemd.cuda -O -p cb6-but-dum.prmtop -ref equilibration.rst7 -c equilibration.rst7 -i production.in -o production.out -r production.rst7 -inf production.info -e production.mden
