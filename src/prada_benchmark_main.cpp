#include "prada.h"
#include "utils.h"

#include <mockturtle/networks/mig.hpp>

#include <chrono>
#include <iostream>

using namespace mockturtle;
using namespace eggmock;
using namespace std::chrono;

// usage: exec [network]
int main( int const argc, char** argv )
{
  if ( argc != 2 )
  {
    std::cerr << "usage: " << argv[0] << std::endl;
    return 1;
  }

  std::optional<mig_network> mig = get_ntk<mig_network>( argv[1] );
  if ( !mig )
  {
    return 1;
  }

  auto const pre_opt_size = mig->size();

  auto const opt_begin = system_clock::now();
  preoptimize_mig( *mig );
  auto const t_opt = duration_cast<milliseconds>( system_clock::now() - opt_begin ).count();

  auto constexpr settings = prada_compiler_settings{
      .print_program = false,
      .verbose = false,
      // .verbose = true, // to look at the generated programs
  };

  const auto [egraph_classes, egraph_nodes, egraph_size,
              instruction_count, runtime_estimate, energy_consumption_estimate,
              t_runner, t_extractor, t_compiler] = prada_compile( settings, *mig );

  std::cout << t_opt << "," << t_runner << "," << t_extractor << "," << t_compiler << ","
            << pre_opt_size << "," << mig->size() << "," << mig->num_cis() << "," << mig->num_cos() << ","
            << instruction_count << "," << runtime_estimate << "," << energy_consumption_estimate << ","
            << egraph_classes << "," << egraph_nodes << "," << egraph_size << std::endl;
  return 0;
}
