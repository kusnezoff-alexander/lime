#include "eggmock.h"

#include "ambit.h"

#include <mockturtle/io/write_dot.hpp>
#include <mockturtle/networks/mig.hpp>
#include <sys/types.h>

using namespace mockturtle;
using namespace eggmock;

int main()
{
  mig_network in;
  const auto b_i = in.create_pi();
  const auto b_i_next = in.create_pi();
  const auto m = in.create_pi();

  const auto O1 = in.create_and( m, b_i_next );
  const auto O2 = in.create_and( in.create_not( m ), b_i );
  const auto bi = in.create_or( O1, O2 );
  in.create_po( bi );

  write_dot( in, "in.dot" );

  ambit_compile_result result = eggmock::send_mig( in, ambit_compile( ambit_compiler_settings{
                                                           .print_program = true,
                                                           .verbose = true,
                                                       } ) );
  std::cout << "IC:" << result.instruction_count << std::endl;
  std::cout << "t1:" << result.t_runner << std::endl;
  std::cout << "t2:" << result.t_extractor << std::endl;
  std::cout << "t3:" << result.t_compiler << std::endl;

  // mig_network rewritten = rewrite_mig( in, ambit_rewriter() );
  // write_dot( rewritten, "out.dot" );
}