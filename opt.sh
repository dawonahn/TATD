gpu=$1
lr=0.01

for lr in 0.00001
do
for count in 31
do
#        python3 main.py --gpu $gpu --name ncfd --opt_scheme als_adam --lr $lr --penalty 0.1 --window 3  --count $count --sparse 1
#       python3 main.py --gpu $gpu --name ncfd --opt_scheme adam --lr $lr --penalty 0.1 --window 3  --count $count --sparse 1
#        python3 main.py --gpu $gpu --name ncfd --opt_scheme alternating_adam --lr $lr --penalty 0.1 --window 3  --count $count --sparse 1
        python3 main.py --gpu $gpu --name ncfd --opt_scheme sgd --lr $lr --penalty 0.1 --window 3  --count $count --sparse 1
        python3 main.py --gpu $gpu --name ncfd --opt_scheme als_agd --lr $lr --penalty 0.1 --window 3  --count $count --sparse 1
done
done
