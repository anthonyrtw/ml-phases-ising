#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <torch/torch.h>

// Constants
const int L = 10;
const int EquilibrationSteps = 50000;
const float J = 1.0;
const int numT = 100;  
const float minT = 0.01, maxT = 5; 
const int numRuns = 1;


// Initialize lattice with +1 spins
torch::Tensor initializeLattice(int L) {
    return 2 * torch::randint(0, 2, {L, L}) - 1;
}

// Calculates change in energy if a spin flip occurs at lattice site (x, y)
float deltaEnergy(torch::Tensor& lattice, int x, int y) {
    int spin = lattice.index({x, y}).item<int>();
    int spin_right = lattice.index({(x + 1 + L) % L, y}).item<int>();
    int spin_left = lattice.index({(x - 1 + L) % L, y}).item<int>();
    int spin_up = lattice.index({x, (y + 1 + L) % L}).item<int>();
    int spin_down = lattice.index({x,(y - 1 + L) % L}).item<int>();
    return 2 * J * spin * (spin_right + spin_left + spin_up + spin_down);
}

// Calculates the normalised magnetisation of the lattice
float normMagnetization(torch::Tensor& lattice) {
    return torch::sum(lattice).item<int>() / static_cast<float>(lattice.numel());
}

// Runs metropolis algorithm for simulating Ising model for a specified value of T
void equilibration(torch::Tensor& lattice, float T, int steps, std::mt19937& rng) {
    std::uniform_int_distribution<> coord(0, L - 1);
    std::uniform_real_distribution<> prob(0.0, 1.0);

    for (int step = 0; step < steps; ++step) {
        int x = coord(rng);
        int y = coord(rng);

        // Calculate change in energy associated with spin flip at (x, y)
        float dE = deltaEnergy(lattice, x, y);

        // If energy decreases, flip spin. Else run probability of spin flip
        if (dE < 0) {
            lattice.index_put_({x, y}, -lattice.index({x, y}).item<int>());
        } else {
            float flip_prob = exp(-dE / T);
            float P = prob(rng);

            if (P < flip_prob) {
                lattice.index_put_({x, y}, -lattice.index({x, y}).item<int>());
            }
        }
    }
}

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    torch::Tensor lattices = torch::empty({0});
    torch::Tensor magnetizations = torch::empty({0});
    torch::Tensor temperatures = torch::linspace(minT, maxT, numT);

    for (int run = 0; run < numRuns; ++run){ 
        torch::Tensor lattice = initializeLattice(L);

        // Cycle through each temperature and perform thermalisation at each temperature
        for (int i = 0; i < temperatures.size(0); ++i) {

            float T = temperatures.index({i}).item<float>();

            // Update the lattice using thermalization
            equilibration(lattice, T, EquilibrationSteps, rng);
    
            // Store magnetization and lattice to tensor
            torch::Tensor magnetization = torch::tensor({normMagnetization(lattice)});

            if (lattices.numel() == 0) {
                lattices = lattice.unsqueeze(0);
                magnetizations = magnetization;
            } else { 
                lattices = torch::cat({lattices, lattice.unsqueeze(0)}, 0);
                magnetizations = torch::cat({magnetizations, magnetization});
            }
        }
    }

    // Shuffle the data
    torch::Tensor temperatures_data = temperatures.repeat({numRuns * numT});

    std::vector<int64_t> shuffle_indices(numRuns * numT);
    for (int i = 0; i < numRuns * numT; ++i) {
        shuffle_indices[i] = i;
    }
    std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), rng);

    torch::Tensor shuffled_lattices = lattices.index_select(0, torch::tensor(shuffle_indices));
    torch::Tensor shuffled_temperatures = temperatures_data.index_select(0, torch::tensor(shuffle_indices));
    torch::Tensor shuffled_magnetizations = magnetizations.index_select(0, torch::tensor(shuffle_indices));
    
    // Export the data
    torch::save(shuffled_lattices, "../../data/lattices.pt");
    torch::save(shuffled_temperatures, "../../data/temperatures.pt");
    torch::save(shuffled_magnetizations, "../../data/magnetizations.pt");

    return 0;
}
