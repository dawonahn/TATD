gpu=$1
scheme='als_adam'
for rank in 20 40  
do
    for count in 3
    do
        lr=0.01
        #python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count --rank $rank
#        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count --rank $rank
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count --rank $rank
#        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 1000 --window 3  --count $count --rank $rank
    done
done
