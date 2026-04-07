# ##########
# 2,3,4,5x10
# ##########
python main.py -cname QAOA -nq 20 -core 2 -cap 10
python main.py -cname QAOA -nq 30 -core 3 -cap 10
python main.py -cname QAOA -nq 40 -core 4 -cap 10
python main.py -cname QAOA -nq 50 -core 5 -cap 10

python main.py -cname QFT -nq 20 -core 2 -cap 10
python main.py -cname QFT -nq 30 -core 3 -cap 10
python main.py -cname QFT -nq 40 -core 4 -cap 10
python main.py -cname QFT -nq 50 -core 5 -cap 10

python main.py -cname QV -nq 20 -core 2 -cap 10
python main.py -cname QV -nq 30 -core 3 -cap 10
python main.py -cname QV -nq 40 -core 4 -cap 10
python main.py -cname QV -nq 50 -core 5 -cap 10

# ##########
# 2,3,4,5x20
# ##########
python main.py -cname QAOA -nq 40 -core 2 -cap 20
python main.py -cname QAOA -nq 60 -core 3 -cap 20
python main.py -cname QAOA -nq 80 -core 4 -cap 20
python main.py -cname QAOA -nq 100 -core 5 -cap 20

python main.py -cname QFT -nq 40 -core 2 -cap 20
python main.py -cname QFT -nq 60 -core 3 -cap 20
python main.py -cname QFT -nq 80 -core 4 -cap 20
python main.py -cname QFT -nq 100 -core 5 -cap 20

python main.py -cname QV -nq 40 -core 2 -cap 20
python main.py -cname QV -nq 60 -core 3 -cap 20
python main.py -cname QV -nq 80 -core 4 -cap 20
python main.py -cname QV -nq 100 -core 5 -cap 20

# ##########
# 2,3,4,5x30
# ##########


# ##########
# 4x20 mesh
# ##########
# python main.py -cname QAOA -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname QFT -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname QV -nq 80 -core 4 -cap 20 -net mesh_grid

# python main.py -cname QV -nq 30 -core 3 -cap 10
# python main.py -cname QV -nq 60 -core 3 -cap 20
# python main.py -cname QV -nq 90 -core 3 -cap 30
# python main.py -cname QV -nq 120 -core 3 -cap 40

# python main.py -cname QFT -nq 30 -core 3 -cap 10
# python main.py -cname QFT -nq 60 -core 3 -cap 20
# python main.py -cname QFT -nq 90 -core 3 -cap 30
# python main.py -cname QFT -nq 120 -core 3 -cap 40

# python main.py -cname QAOA -nq 30 -core 3 -cap 10
# python main.py -cname QAOA -nq 60 -core 3 -cap 20
# python main.py -cname QAOA -nq 90 -core 3 -cap 30
# python main.py -cname QAOA -nq 120 -core 3 -cap 40

# python main.py -cname QV -nq 40 -core 4 -cap 10 -net mesh_grid
# python main.py -cname QFT -nq 40 -core 4 -cap 10 -net mesh_grid
# python main.py -cname QAOA -nq 40 -core 4 -cap 10 -net mesh_grid

# python main.py -cname QV -nq 60 -core 2 -cap 35
# python main.py -cname QV -nq 90 -core 3 -cap 35
# python main.py -cname QV -nq 120 -core 4 -cap 35
# python run_main.py -cname QV -nq 120 -core 4 -cap 30 -net mesh_grid
# python main.py -cname QV -nq 90 -core 3 -cap 31
# python main.py -cname QV -nq 120 -core 4 -cap 31
# python run_main.py -cname random_n70 -nq 70 -core 3 -cap 30
# python run_main.py -cname random_n130 -nq 130 -core 3 -cap 50

# python ablation.py -cname QV -nq 90 -core 3 -cap 30 -net mesh_grid
# python run_main.py -cname DraperQFTAdder -nq 45 -core 3 -cap 30
# python run_main.py -cname ising_n420 -nq 420 -core 9 -cap 50
# python run_main.py -cname QAOA -nq 90 -core 3 -cap 30
# python run_main.py -cname QFT -nq 90 -core 3 -cap 30
# QFTAdder
# python run_main.py -cname QFTAdder -nq 90 -core 3 -cap 30