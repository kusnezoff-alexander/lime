#include "ambit.h"
#include "eggmock.h"
#include "utils.h"

#include <chrono>
#include <iostream>
#include <lorina/aiger.hpp>
#include <lorina/bench.hpp>
#include <lorina/blif.hpp>
#include <lorina/common.hpp>
#include <lorina/genlib.hpp>
#include <lorina/pla.hpp>
#include <lorina/verilog.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>

using namespace mockturtle;
using namespace eggmock;
using namespace std::chrono;

// usage: exec [network]
int main( int argc, char** argv )
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

  auto opt_begin = system_clock::now();
  preoptimize_mig( *mig );
  auto t_opt = duration_cast<milliseconds>( system_clock::now() - opt_begin ).count();

  auto settings = ambit_compiler_settings{
      .print_program = false,
      .verbose = false,
  };
  ambit_compile_result result = send_mig( *mig, ambit_compile( settings ) );

  std::cout
      << t_opt << "\t"
      << result.t_runner << "\t"
      << result.t_extractor << "\t"
      << result.t_compiler << "\t"
      << result.instruction_count
      << std::endl;
  return 0;
}
