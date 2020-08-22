gpu=$1
scheme='als_adam'
#for penalty in 0.001 0.01 0.1 1 10 100 1000 10000 
for penalty in 0.1
do
    rank=10
    for count in 4
    do
        lr=0.01
#        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty $penalty --window 7  --count $count --rank $rank --sparse 0
##        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty $penalty --window 9  --count $count --rank $rank --sparse 0
#        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty $penalty --window 5  --count $count --rank $rank --sparse 0
#        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty $penalty --window 3  --count $count --rank $rank --sparse 0
        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty $penalty --window 7  --count $count --rank $rank --sparse 1
#        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty $penalty --window 9  --count $count --rank $rank --sparse 1
#        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty $penalty --window 5  --count $count --rank $rank --sparse 1
#        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty $penalty --window 3  --count $count --rank $rank --sparse 1
    done
done
