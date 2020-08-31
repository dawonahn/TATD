gpu=$1
scheme=$2
name=$3
lr=0.00001

for count in 5000
do
    if [ $name == 'indoor' ]
    then
        python3 main.py --gpu $gpu --name indoor --opt_scheme $scheme --lr $lr --penalty 100 --window 7 --count $count 
    elif [ $name == 'mad' ]
    then
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count --sparse 1 
        python3 main.py --gpu $gpu --name mad --opt_scheme $scheme --lr $lr --penalty 100 --window 5  --count $count --sparse 0
    elif [ $name == 'radar' ]
    then
        python3 main.py --gpu $gpu --name radar --opt_scheme $scheme --lr $lr --penalty 100 --window 9  --count $count 
    elif [ $name == 'beijing' ]
    then
        python3 main.py --gpu $gpu --name beijing --opt_scheme $scheme --lr $lr --penalty 1000 --window 3  --count $count 
    elif [ $name == 'indoor_mm' ]
    then
        python3 main.py --gpu $gpu --name indoor_mm --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count --sparse 1
        python3 main.py --gpu $gpu --name indoor_mm --opt_scheme $scheme --lr $lr --penalty 100 --window 7  --count $count --sparse 0 
    elif [ $name == 'indoor_org' ]
    then
        python3 main.py --gpu $gpu --name indoor_org --opt_scheme $scheme --lr $lr --penalty 10 --window 3 --count $count --sparse 1
        python3 main.py --gpu $gpu --name indoor_org --opt_scheme $scheme --lr $lr --penalty 10 --window 3 --count $count --sparse 0 
    elif [ $name == 'bair_mm' ]
    then
        python3 main.py --gpu $gpu --name bair_mm --opt_scheme $scheme --lr $lr --penalty 10 --window 3  --count $count --sparse 1
        python3 main.py --gpu $gpu --name bair_mm --opt_scheme $scheme --lr $lr --penalty 10 --window 3  --count $count --sparse 0 
    elif [ $name == 'bair_mm_1' ]
    then
        python3 main.py --gpu $gpu --name bair_mm_1 --opt_scheme $scheme --lr $lr --penalty 10 --window 3  --count $count --sparse 1
        python3 main.py --gpu $gpu --name bair_mm_1 --opt_scheme $scheme --lr $lr --penalty 10 --window 3  --count $count --sparse 0 
    elif [ $name == 'mair_mm' ]
    then
        python3 main.py --gpu $gpu --name mair_mm --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count --sparse 1
        python3 main.py --gpu $gpu --name mair_mm --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count --sparse 0 
    elif [ $name == 'traffic' ]
    then
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count --sparse 1
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count --sparse 0 
    elif [ $name == 'mad_mm' ]
    then
        python3 main.py --gpu $gpu --name mad_mm --opt_scheme $scheme --lr $lr --penalty 1 --window 3  --count $count --sparse 1
        python3 main.py --gpu $gpu --name mad_mm --opt_scheme $scheme --lr $lr --penalty 1 --window 3  --count $count --sparse 0 
    elif [ $name == 'mad_mm_1' ]
    then
        python3 main.py --gpu $gpu --name mad_mm_1 --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count --sparse 1
        python3 main.py --gpu $gpu --name mad_mm_1 --opt_scheme $scheme --lr $lr --penalty 10 --window 5  --count $count --sparse 0 
    elif [ $name == 'bair_org' ]
    then
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 100 --window 3 --count $count --sparse 1
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 100 --window 3  --count $count --sparse 0 
    elif [ $name == 'indoor_org' ]
    then
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 1 --window 3  --count $count --sparse 1
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 1 --window 3  --count $count --sparse 0 
     elif [ $name == 'mair_org' ]
    then
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 1 --window 5  --count $count --sparse 1
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 1 --window 5  --count $count --sparse 0 
     elif [ $name == 'traffic_org' ]
    then
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 10 --window 3  --count $count --sparse 1
        python3 main.py --gpu $gpu --name $name --opt_scheme $scheme --lr $lr --penalty 10 --window 3  --count $count --sparse 0 
    fi
done
