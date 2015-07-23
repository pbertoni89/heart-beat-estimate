#/bin/bash

for i in $(ls s*.jpg)
do
    convert -equalize $i $i
done
