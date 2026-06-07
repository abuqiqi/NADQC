# cnames=(BV SwapTest Permutation MCMT  QFTAdder CuccaroAdder IQP Random QV VQC  QAOA

# cnames=(Comparator) # QFT VQC QAOA QiskitVQE QKNN
# # 
# nets=(all_to_all) #  mesh_grid star chain
# configs=(
#   # "100 2 50"
#   # "150 3 50"
#   "200 4 50"
#   # "250 5 50"
#   # "80 4 20"
# )

# for cname in "${cnames[@]}"; do
#   for net in "${nets[@]}"; do
#     for config in "${configs[@]}"; do
#       read -r nq core cap <<< "$config"
#       python main.py -cname "$cname" -nq "$nq" -core "$core" -cap "$cap" -net "$net"
#     done
#   done
# done

# ##########
# 2,3,4,5x20
# ##########
# python main.py -cname Random -nq 40 -core 2 -cap 20
# python main.py -cname Random -nq 60 -core 3 -cap 20
# python main.py -cname Random -nq 80 -core 4 -cap 20
# python main.py -cname Random -nq 100 -core 5 -cap 20

# ##########
# 3x20,30,40,50
# ##########
# python main.py -cname Random -nq 60 -core 3 -cap 20 -net all_to_all
# python main.py -cname Random -nq 90 -core 3 -cap 30 -net all_to_all
# python main.py -cname Random -nq 120 -core 3 -cap 40 -net all_to_all
# python main.py -cname Random -nq 150 -core 3 -cap 50 -net all_to_all
python main.py -cname Random -nq 30 -core 3 -cap 10 -net all_to_all
python main.py -cname Random -nq 60 -core 3 -cap 20 -net all_to_all
python main.py -cname Random -nq 90 -core 3 -cap 30 -net all_to_all
python main.py -cname Random -nq 120 -core 3 -cap 40 -net all_to_all

# python main.py -cname Random -nq 60 -core 3 -cap 20 -net chain
# python main.py -cname Random -nq 90 -core 3 -cap 30 -net chain
# python main.py -cname Random -nq 120 -core 3 -cap 40 -net chain
# python main.py -cname Random -nq 150 -core 3 -cap 50 -net chain
