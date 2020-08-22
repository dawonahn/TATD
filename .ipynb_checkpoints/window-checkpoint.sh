gpu=$1
data=$2
scheme='als_adam'
lr=0.01
count=10
for window in 3 5 7 9 11 
do
    if [ $data == 'indoor' ]
    then
        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count --window $window --sparse 0 
        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count --window $window --sparse 1 
    elif [ $data == 'radar' ]
    then
        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count --window $window --sparse 0 
        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count --window $window --sparse 1
    elif [ $data == 'mad' ]
    then
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count --sparse 0 --window $window 
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count --sparse 1 --window $window
    elif [ $data == 'beijing' ]
    then
        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 1000 --window 3  --count $count --window $window --sparse 0 
        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 1000 --window 3  --count $count --window $window --sparse 1 
    fi
done
