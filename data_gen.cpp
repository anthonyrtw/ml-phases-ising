#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <torch/torch.h>

// Constants
const std::string shape = "square"; // or "triangular";
const int L = 10;
const int EquilibrationSteps = 50000;
const float J = 1.0;
const int numT = 42;  
const float minT = 1.0, maxT = 3.5; 
const int numRuns = 100;

// Initialize lattice with +1 spins
torch::Tensor initializeLattice(int L) {
    return 2 * torch::randint(0, 2, {L, L}) - 1;
}

// Calculates change in energy if a spin flip occurs at lattice site (x, y)
float deltaEnergySquare(torch::Tensor& lattice, float J, int x, int y) {
    int spin = lattice.index({x, y}).item<int>();
    int spin_right = lattice.index({(x + 1 + L) % L, y}).item<int>();
    int spin_left = lattice.index({(x - 1 + L) % L, y}).item<int>();
    int spin_up = lattice.index({x, (y + 1 + L) % L}).item<int>();
    int spin_down = lattice.index({x,(y - 1 + L) % L}).item<int>();
    return 2 * J * spin * (spin_right + spin_left + spin_up + spin_down);
}

// Calculates change in energy if a spin flip occurs at lattice site (x, y)
float deltaEnergyTriangular(torch::Tensor& lattice, float J, int x, int y) {
    int spin = lattice.index({x, y}).item<int>();
    int spin1 = lattice.index({(x - 1 + L) % L, y}).item<int>();
    int spin2 = lattice.index({(x + 1 + L) % L, y}).item<int>();
    int spin3 = lattice.index({x, (y - 1 + L) % L}).item<int>();
    int spin4 = lattice.index({x, (y + 1 + L) % L}).item<int>();
    int spin5 = lattice.index({(x - 1 + L) % L, (y + 1 + L) % L}).item<int>();
    int spin6 = lattice.index({(x + 1 + L) % L, (y - 1 + L) % L}).item<int>();

    return 2 * J * spin * (spin1 + spin2 + spin3 + spin4 + spin5 + spin6);
}

// Calculates the normalised magnetisation of the lattice
float normMagnetization(torch::Tensor& lattice) {
    return torch::sum(lattice).item<float>() / static_cast<float>(lattice.numel());
}

// Runs metropolis algorithm for simulating Ising model for a specified value of T
void equilibration(std::string shape, torch::Tensor& lattice, float T, float J, int steps, std::mt19937& rng) {
    std::uniform_int_distribution<> coord(0, L - 1);
    std::uniform_real_distribution<> prob(0.0, 1.0);

    // Specify energy calculation based on lattice
    float (*deltaEnergy)(torch::Tensor&, float, int, int);
    if (shape == "triangular") {
        deltaEnergy = deltaEnergyTriangular;
    } else {
        deltaEnergy = deltaEnergySquare;
    } 

    for (int step = 0; step < steps; ++step) {
        // Select random coordinate in lattice
        int x = coord(rng);
        int y = coord(rng);

        // Calculate change in energy associated with spin flip at (x, y)
        float dE = deltaEnergy(lattice, J, x, y);

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
            equilibration(shape, lattice, T, J, EquilibrationSteps, rng);
    
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
    torch::Tensor temperatures_data = temperatures.repeat({numRuns});

    std::vector<int64_t> shuffle_indices(numRuns * numT);
    for (int i = 0; i < numRuns * numT; ++i) {
        shuffle_indices[i] = i;
    }
    std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), rng);

    torch::Tensor shuffled_lattices = lattices.index_select(0, torch::tensor(shuffle_indices, torch::kInt64));
    torch::Tensor shuffled_temperatures = temperatures_data.index_select(0, torch::tensor(shuffle_indices, torch::kInt64));
    torch::Tensor shuffled_magnetizations = magnetizations.index_select(0, torch::tensor(shuffle_indices, torch::kInt64));
    
    // Export the data
    auto pickled_lattices = torch::pickle_save(shuffled_lattices);
    auto pickled_temperatures = torch::pickle_save(shuffled_temperatures);
    auto pickled_magnetizations = torch::pickle_save(shuffled_magnetizations);

    std::string dir = "../../data/" + shape + "/L" + std::to_string(L);
    std::filesystem::create_directories(dir);

    std::ofstream fout1(dir + "/lattices.pt", std::ios::out | std::ios::binary);
    fout1.write(pickled_lattices.data(), pickled_lattices.size());
    fout1.close();

    std::ofstream fout2(dir + "/temperatures.pt", std::ios::out | std::ios::binary);
    fout2.write(pickled_temperatures.data(), pickled_temperatures.size());
    fout2.close();

    std::ofstream fout3(dir + "/magnetizations.pt", std::ios::out | std::ios::binary);
    fout3.write(pickled_magnetizations.data(), pickled_magnetizations.size());
    fout3.close();

    return 0;
}
