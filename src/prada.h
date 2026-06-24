#pragma once

#include "eggmock.h"
#include "utils.h"

#include <mockturtle/networks/mig.hpp>

#include <cstdint>
#include <utility>

extern "C"
{
  struct prada_compiler_statistics
  {
    uint64_t egraph_classes;
    uint64_t egraph_nodes;
    uint64_t egraph_size;

    uint64_t instruction_count;
    uint64_t runtime_estimate;
    uint64_t energy_consumption_estimate;

    uint64_t t_runner;
    uint64_t t_extractor;
    uint64_t t_compiler;
  };

  struct prada_compiler_settings
  {
    bool print_program;
    bool verbose;
    bool preoptimize = true;
    bool rewrite = true;
  };

  struct prada_compiler_settings_ffi
  {
    bool print_program;
    bool verbose;
    bool rewrite = true;

    prada_compiler_settings_ffi( prada_compiler_settings s )
        : print_program( s.print_program ), verbose( s.verbose ), rewrite( s.rewrite ) {}
  };

  eggmock::mig_receiver<prada_compiler_statistics> prada_compile_ffi(
      prada_compiler_settings_ffi settings );
  eggmock::mig_receiver<prada_compiler_statistics> prada_rewrite_ffi(
      prada_compiler_settings_ffi settings,
      eggmock::mig_receiver<void> receiver );
}

inline std::pair<mockturtle::mig_network, prada_compiler_statistics> prada_rewrite(
    prada_compiler_settings settings,
    mockturtle::mig_network& ntk )
{
  if ( settings.preoptimize )
  {
    preoptimize_mig( ntk );
  }
  mockturtle::mig_network out;
  const auto stat = eggmock::send_mig(
      ntk, prada_rewrite_ffi( settings, eggmock::receive_mig( out ) ) );
  return { out, stat };
}

inline prada_compiler_statistics prada_compile(
    prada_compiler_settings settings,
    mockturtle::mig_network& ntk )
{
  if ( settings.preoptimize )
  {
    preoptimize_mig( ntk );
  }
  mockturtle::mig_network out;
  const auto stat = eggmock::send_mig( ntk, prada_compile_ffi( settings ) );
  return stat;
}
