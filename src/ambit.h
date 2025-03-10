#pragma once

#include "eggmock.h"

#include <cstdint>

extern "C"
{
  struct ambit_compiler_statistics
  {
    uint64_t egraph_classes;
    uint64_t egraph_nodes;
    uint64_t egraph_size;

    uint64_t instruction_count;

    uint64_t t_runner;
    uint64_t t_extractor;
    uint64_t t_compiler;
  };

  struct ambit_compiler_settings
  {
    bool print_program;
    bool verbose;
  };

  eggmock::mig_receiver<eggmock::mig_rewrite> ambit_rewriter( ambit_compiler_settings settings );
  eggmock::mig_receiver<ambit_compiler_statistics> ambit_compile( ambit_compiler_settings settings );
}
