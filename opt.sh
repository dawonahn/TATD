gpu=$1
scheme=$2
lr=0.01

for count in 1
do
        python3 main.py --gpu $gpu --name mair_std --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count --sparse 1
        #python3 main.py --gpu $gpu --name mair_std --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count --sparse 0 
done
