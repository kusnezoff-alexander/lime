#include "eggmock.h"
#include <mockturtle/networks/mig.hpp>

using namespace mockturtle;
using namespace eggmock;

extern "C"
{
  eggmock::mig_rewrite lime_rewrite();
}

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

  mig_network out = rewrite_mig( in, lime_rewrite() );

  write_dot( out, "out.dot" );
}