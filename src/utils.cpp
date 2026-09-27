#include "utils.h"

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/functional_reduction.hpp>
#include <mockturtle/algorithms/mig_algebraic_rewriting.hpp>
#include <mockturtle/algorithms/mig_inv_optimization.hpp>
#include <mockturtle/algorithms/mig_inv_propagation.hpp>
#include <mockturtle/algorithms/mig_resub.hpp>
#include <mockturtle/algorithms/resubstitution.hpp>

using namespace mockturtle;

void preoptimize_mig( mockturtle::mig_network& ntk )
{
  // Every pass gets freshly constructed views on a cleaned-up network. Sharing one depth_view /
  // fanout_view across all passes is unsound: mig_inv_optimization changes the network in ways
  // the views do not track, and mig_resubstitution2 then substitutes based on stale fanout and
  // level information, producing non-equivalent networks (e.g. for ge32 and div8).
  {
    fanout_view fanout_mig{ ntk };
    functional_reduction( fanout_mig );
  }
  ntk = cleanup_dangling( ntk );
  {
    fanout_view fanout_mig{ ntk };
    mig_inv_optimization( fanout_mig );
  }
  ntk = cleanup_dangling( ntk );
  {
    depth_view depth_mig{ ntk };
    fanout_view fanout_mig{ depth_mig };
    resubstitution_params ps;
    resubstitution_stats st;
    mig_resubstitution2( fanout_mig, ps, &st );
    // mig_resubstitution( fanout_mig,  ps, &st  );
  }
  ntk = cleanup_dangling( ntk );
  {
    depth_view depth_mig{ ntk };
    mig_algebraic_depth_rewriting( depth_mig );
  }
  ntk = cleanup_dangling( ntk );
}
