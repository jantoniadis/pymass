#!/bin/csh


#Check all distributions using all data
#python -W ignore pymass.py -d1 msps3.txt -d2 onepk.data --threads 10 -t 5 -n 2000 -w 800 --init '1.35 0.09 1.89 0.2 0.4' -f bimodal -s 1e-2 -o bimodal.data -c True

python -W ignore pymass.py -d1 msps3.txt -d2 onepk.data --threads 10 -t 5 -n 1000 -w 800 --init '33.0 0.01' -f gamma -s 1e-2 -o gamma.data

python -W ignore pymass.py  -d1 msps3.txt -d2 onepk.data --threads 10 -t 5 -n 2000 -w 800 --init '1.48, 0.1, 8.0' -f normal_exp -s 1e-2 -o normal_exp.data

#python -W ignore pymass.py -d1 msps3.txt -d2 onepk.data --threads 10 -t 5 -n 1000 -w 800 --init '1.41 0.1 4.5' -f skewed_normal -s 1e-2 -o skewed.data

#python -W ignore pymass.py -d1 msps3.txt -d2 onepk.data --threads 10 -t 5 -n 2000 -w 800 --init '1.48 0.2' -f normal -s 1e-2 -o normal.data

#python -W ignore pymass.py -d1 msps3.txt -d2 onepk.data --threads 10 -t 5 -n 2000 -w 800 --init '1.35 0.09 1.89 0.2 0.4 2.2' -f bimodal_cut -s 1e-2 -o bimodal_cut.data


#check using precise MSP data

python -W ignore pymass.py -d1 msps3.txt --threads 10 -t 5 -n 2000 -w 800 --init '1.35 0.09 1.89 0.2 0.4' -f bimodal -s 1e-2 -o bimodal_p.data

python -W ignore pymass.py -d1 msps3.txt --threads 10 -t 5 -n 1000 -w 800 --init '33.0 0.01' -f gamma -s 1e-2 -o gamma_p.data

python -W ignore pymass.py -d1 msps3.txt --threads 10 -t 5 -n 1000 -w 800 --init '1.41 0.1 4.5' -f skewed_normal -s 1e-2 -o skewed_p.data

python -W ignore pymass.py -d1 msps3.txt --threads 10 -t 5 -n 2000 -w 800 --init '1.48 0.2' -f normal -s 1e-2 -o normal_p.data

python -W ignore pymass.py -d1 msps3.txt --threads 10 -t 5 -n 2000 -w 800 --init '1.35 0.09 1.89 0.2 0.4 2.2' -f bimodal_cut -s 1e-2 -o bimodal_cut.data



