#include "utils.h"
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_rewriting.hpp>
#include <mockturtle/algorithms/mig_resub.hpp>
#include <mockturtle/algorithms/node_resynthesis/mig_npn.hpp>

using namespace mockturtle;

void preoptimize_mig( mockturtle::mig_network& ntk )
{
  // TODO: sensible defaults here or so
  for ( int i = 0; i < 5; i++ )
  {
    mig_npn_resynthesis resyn;
    cut_rewriting_params ps;
    ps.cut_enumeration_ps.cut_size = 4;
    ntk = cut_rewriting( ntk, resyn, ps );

    // mig_resubstitution( ntk );
  }

  ntk = cleanup_dangling( ntk );
}
