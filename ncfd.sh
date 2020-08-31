gpu=$1
data=ncfd
scheme='als_adam'
count=1
#lr=0.0001
#lr=0.001
for lr in 0.001
do
for rank in 10
#for rank in 5 15 20 25
do
    for window in 3 5 
    do
        #for penalty in 1 0.1 0.001
        for penalty in 0.1 
        do
            python3 main2.py --gpu $gpu --name $data --sparse 1 --opt_scheme $scheme --lr $lr --penalty $penalty --window $window  --count $count --rank $rank
            python3 main2.py --gpu $gpu --name $data --sparse 0 --opt_scheme $scheme --lr $lr --penalty $penalty --window $window  --count $count --rank $rank
        done
    done
done
done
