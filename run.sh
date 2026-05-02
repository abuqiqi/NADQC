# ##########
# 2,3,4,5x10
# ##########
# python main.py -cname Random -nq 20 -core 2 -cap 10
# python main.py -cname Random -nq 30 -core 3 -cap 10
# python main.py -cname Random -nq 40 -core 4 -cap 10
# python main.py -cname Random -nq 50 -core 5 -cap 10

# python main.py -cname QAOA -nq 20 -core 2 -cap 10
# python main.py -cname QAOA -nq 30 -core 3 -cap 10
# python main.py -cname QAOA -nq 40 -core 4 -cap 10
# python main.py -cname QAOA -nq 50 -core 5 -cap 10

# python main.py -cname QFT -nq 20 -core 2 -cap 10
# python main.py -cname QFT -nq 30 -core 3 -cap 10
# python main.py -cname QFT -nq 40 -core 4 -cap 10
# python main.py -cname QFT -nq 50 -core 5 -cap 10

# python main.py -cname QV -nq 20 -core 2 -cap 10
# python main.py -cname QV -nq 30 -core 3 -cap 10
# python main.py -cname QV -nq 40 -core 4 -cap 10
# python main.py -cname QV -nq 50 -core 5 -cap 10

# ##########
# 2,3,4,5x20
# ##########

cnames=(BV) # Permutation DraperQFTAdder QFT QAOA QV
nets=(all_to_all)
configs=(
  "100 2 50"
  "150 3 50"
  "200 4 50"
  "250 5 50"
)

for cname in "${cnames[@]}"; do
  for net in "${nets[@]}"; do
    for config in "${configs[@]}"; do
      read -r nq core cap <<< "$config"
      python main.py -cname "$cname" -nq "$nq" -core "$core" -cap "$cap" -net "$net"
    done
  done
done

# python main.py -cname Random -nq 40 -core 2 -cap 20
# python main.py -cname Random -nq 60 -core 3 -cap 20
# python main.py -cname Random -nq 80 -core 4 -cap 20
# python main.py -cname Random -nq 100 -core 5 -cap 20

# python main.py -cname QAOA -nq 40 -core 2 -cap 20
# python main.py -cname QAOA -nq 60 -core 3 -cap 20
# python main.py -cname QAOA -nq 80 -core 4 -cap 20
# python main.py -cname QAOA -nq 100 -core 5 -cap 20

# python main.py -cname QFT -nq 40 -core 2 -cap 20
# python main.py -cname QFT -nq 60 -core 3 -cap 20
# python main.py -cname QFT -nq 80 -core 4 -cap 20
# python main.py -cname QFT -nq 100 -core 5 -cap 20

# python main.py -cname QV -nq 40 -core 2 -cap 20
# python main.py -cname QV -nq 60 -core 3 -cap 20
# python main.py -cname QV -nq 80 -core 4 -cap 20
# python main.py -cname QV -nq 100 -core 5 -cap 20

# ##########
# 3x10,20,30,40,50
# ##########
# python main.py -cname Random -nq 60 -core 3 -cap 20 -net all_to_all
# python main.py -cname Random -nq 90 -core 3 -cap 30 -net all_to_all
# python main.py -cname Random -nq 120 -core 3 -cap 40 -net all_to_all
# python main.py -cname Random -nq 150 -core 3 -cap 50 -net all_to_all

# python main.py -cname Random -nq 60 -core 3 -cap 20 -net chain
# python main.py -cname Random -nq 90 -core 3 -cap 30 -net chain
# python main.py -cname Random -nq 120 -core 3 -cap 40 -net chain
# python main.py -cname Random -nq 150 -core 3 -cap 50 -net chain

# ##########
# 4x20
# ##########
# python main.py -cname BV -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname CuccaroAdder -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname DraperQFTAdder -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname Permutation -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname MCMT -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname IQP -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname QAOA -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname QFT -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname Random -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname QV -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname VQC_AA -nq 80 -core 4 -cap 20 -net mesh_grid
# python main.py -cname Pauli -nq 80 -core 4 -cap 20 -net mesh_grid

# python main.py -cname BV -nq 80 -core 4 -cap 20 -net star
# python main.py -cname CuccaroAdder -nq 80 -core 4 -cap 20 -net star
# python main.py -cname DraperQFTAdder -nq 80 -core 4 -cap 20 -net star
# python main.py -cname Permutation -nq 80 -core 4 -cap 20 -net star
# python main.py -cname MCMT -nq 80 -core 4 -cap 20 -net star
# python main.py -cname IQP -nq 80 -core 4 -cap 20 -net star
# python main.py -cname QAOA -nq 80 -core 4 -cap 20 -net star
# python main.py -cname QFT -nq 80 -core 4 -cap 20 -net star
# python main.py -cname Random -nq 80 -core 4 -cap 20 -net star
# python main.py -cname QV -nq 80 -core 4 -cap 20 -net star
# python main.py -cname VQC_AA -nq 80 -core 4 -cap 20 -net star
# python main.py -cname Pauli -nq 80 -core 4 -cap 20 -net star

# python main.py -cname BV -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname CuccaroAdder -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname DraperQFTAdder -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname Permutation -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname MCMT -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname IQP -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname QAOA -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname QFT -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname Random -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname QV -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname VQC_AA -nq 80 -core 4 -cap 20 -net chain
# python main.py -cname Pauli -nq 80 -core 4 -cap 20 -net chain

# python main.py -cname BV -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname CuccaroAdder -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname DraperQFTAdder -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname Permutation -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname MCMT -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname IQP -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname QAOA -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname QFT -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname Random -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname QV -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname VQC_AA -nq 80 -core 4 -cap 20 -net all_to_all
# python main.py -cname Pauli -nq 80 -core 4 -cap 20 -net all_to_all
