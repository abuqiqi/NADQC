# python run_main.py -cname QV -nq 120 -core 4 -cap 3
# python run_main.py -cname QV -nq 120 -core 4 -cap 30 -net mesh_grid
# python run_main.py -cname QV -nq 90 -core 3 -cap 30
# python run_main.py -cname QV -nq 60 -core 2 -cap 30

python run_main.py -cname random_n70 -nq 70 -core 3 -cap 30
python run_main.py -cname random_n130 -nq 130 -core 3 -cap 50

# python ablation.py -cname QV -nq 90 -core 3 -cap 30 -net mesh_grid
# python run_main.py -cname DraperQFTAdder -nq 45 -core 3 -cap 30
python run_main.py -cname ising_n420 -nq 420 -core 9 -cap 50
# python run_main.py -cname QAOA -nq 90 -core 3 -cap 30
# python run_main.py -cname QFT -nq 90 -core 3 -cap 30
# QFTAdder
# python run_main.py -cname QFTAdder -nq 90 -core 3 -cap 30