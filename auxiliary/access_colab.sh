#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/drive/1Ss2V4AO_SakK01xmkj_Lvt23XSqbvbRf#scrollTo=2Nr5qmjLmI-R
  sleep 3600
done
