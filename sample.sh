gpu=$1
#data=$2
scheme='als_adam'
lr=0.001

for data in ncfd_1 ncfd_9
do
for count in 1 2 
do
    for window in 3 5
    do
        for penalty in 0.1 1 
        do
        python3 main4.py --gpu $gpu --name ${data} --opt_scheme $scheme --lr $lr --penalty $penalty --window $window --count $count --sparse 0
        python3 main4.py --gpu $gpu --name ${data} --opt_scheme $scheme --lr $lr --penalty $penalty --window $window --count $count --sparse 1 
    done
    done
done
done
