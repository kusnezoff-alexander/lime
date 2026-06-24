#include "ambit.h"
#include "utils.h"

#include <mockturtle/networks/mig.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

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

  // AMBIT_REWRITE=0 disables e-graph rewriting (no saturation) -> much smaller
  // graph to extract, at the cost of an unoptimized (longer) program.
  const char* rw_env = std::getenv( "AMBIT_REWRITE" );
  bool const do_rewrite = !( rw_env && std::string( rw_env ) == "0" );

  auto const settings = ambit_compiler_settings{
      .print_program = false,
      // .verbose = false,
      .verbose = true, // to look at the generated programs
      .rewrite = do_rewrite,
  };

  const auto [egraph_classes, egraph_nodes, egraph_size,
              instruction_count,
              t_runner, t_extractor, t_compiler] = ambit_compile( settings, *mig );

  std::cout << t_opt << "\t" << t_runner << "\t" << t_extractor << "\t" << t_compiler << "\t"
            << pre_opt_size << "\t" << mig->size() << "\t" << mig->num_cis() << "\t" << mig->num_cos() << "\t"
            << instruction_count << "\t"
            << egraph_classes << "\t" << egraph_nodes << "\t" << egraph_size;
  return 0;
}
