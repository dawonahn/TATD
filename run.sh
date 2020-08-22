gpu=$1
scheme=$2

for count in 3 
do
    if [ $scheme == 'sgd' ]
    then
        lr=0.00001
#        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 10 --window 7 --count $count 
#        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 10 --window 9  --count $count 
#        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count 
        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 100 --window 3  --count $count 
    elif [ $scheme == 'adam' ]
    then
        lr=0.01
        for lr in 0.01 
        do
            python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count 
            python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count 
            python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count 
            python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 1000 --window 3  --count $count 
        done
    elif [ $scheme == 'alternating_adam' ]
    then
        for lr in 0.01
        do
            python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count 
            python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count 
            python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count 
            python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 1000 --window 3  --count $count 
        done
    elif [ $scheme == 'alternating_sgd' ]
    then
        lr=0.0001
        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 10 --window 7  --count $count 
        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 10 --window 9  --count $count 
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count 
        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 100 --window 3  --count $count 
    elif [ $scheme == 'als_adam' ]
    then
        lr=0.01
#        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count 
#        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count 
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count --sparse 0 
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count --sparse 1 

#        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 1000 --window 3  --count $count 
    elif [ $scheme == 'als_sgd' ]
    then
        lr=0.01
        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 10 --window 7  --count $count 
        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 10 --window 9  --count $count 
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count 
        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 100 --window 3  --count $count 
    fi
done
