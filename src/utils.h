#pragma once

#include <iostream>
#include <lorina/aiger.hpp>
#include <lorina/common.hpp>
#include <lorina/pla.hpp>
#include <lorina/verilog.hpp>
#include <mockturtle/generators/arithmetic.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/io/pla_reader.hpp>
#include <mockturtle/io/verilog_reader.hpp>

#include <optional>
#include <string>
#include <string_view>

using namespace mockturtle;
// using namespace std; // DON'T !!!

template<class ntk>
std::optional<ntk> get_ntk( std::string const& key );

template<class ntk>
std::optional<ntk> read_ntk( std::string const& path );

void preoptimize_mig( mockturtle::mig_network& ntk );

/**
 * Collection of logic networks (eg for benchmarking)
 * - included logic networks: Full Adder ("fa"), Multiplexer ("mux"), Greater than ("gt"), "kogge_stone"
 * - carry_ripple_adder_inplace ("add"), carry_ripple_multiplier ("mul"), sum-adder ("pop")
 */
template<class ntk_t>
signal<ntk_t> ripple_gt( ntk_t& ntk,
                         const std::vector<signal<ntk_t>>& a_bits,
                         const std::vector<signal<ntk_t>>& b_bits )
{
  size_t bw = a_bits.size();
  signal<ntk_t> gt = ntk.get_constant( false ); // greater-than so far
  signal<ntk_t> eq = ntk.get_constant( true );  // equality so far

  for ( int i = bw - 1; i >= 0; --i ) // MSB -> LSB
  {
    auto ai = a_bits[i];
    auto bi = b_bits[i];

    auto ai_and_not_bi = ntk.create_and( ai, ntk.create_not( bi ) );
    gt = ntk.create_or( gt, ntk.create_and( eq, ai_and_not_bi ) );

    eq = ntk.create_and( eq, ntk.create_not( ntk.create_xor( ai, bi ) ) );
  }

  return gt;
}

template<class ntk_t>
signal<ntk_t> ripple_eq( ntk_t& ntk,
                         const std::vector<signal<ntk_t>>& a,
                         const std::vector<signal<ntk_t>>& b )
{
  signal<ntk_t> neq = ntk.create_xor( a[0], b[0] );
  for ( size_t i = 1; i < a.size(); ++i )
    neq = ntk.create_or( neq, ntk.create_xor( a[i], b[i] ) );
  return ntk.create_not( neq ); // eq = NOT any difference
}

template<class ntk_t>
std::vector<signal<ntk_t>>
twos_complement( ntk_t& ntk,
                 const std::vector<signal<ntk_t>>& a_bits )
{
  const size_t bw = a_bits.size();

  std::vector<signal<ntk_t>> out( bw );

  // +1 for two's complement
  signal<ntk_t> carry = ntk.get_constant( true );

  for ( size_t i = 0; i < bw; ++i )
  {
    auto na = ntk.create_not( a_bits[i] );
    out[i] = ntk.create_xor( na, carry );
    carry = ntk.create_and( na, carry );
  }

  return out;
}

template<class ntk_t>
std::vector<signal<ntk_t>>
ripple_add( ntk_t& ntk,
            const std::vector<signal<ntk_t>>& a,
            const std::vector<signal<ntk_t>>& b )
{
  const size_t bw = a.size();
  std::vector<signal<ntk_t>> sum( bw );

  signal<ntk_t> carry = ntk.get_constant( false );

  for ( size_t i = 0; i < bw; ++i )
  {
    // sum = a ⊕ b ⊕ carry
    auto axb = ntk.create_xor( a[i], b[i] );
    sum[i] = ntk.create_xor( axb, carry );

    // carry = majority(a, b, carry)
    auto ab = ntk.create_and( a[i], b[i] );
    auto ac = ntk.create_and( a[i], carry );
    auto bc = ntk.create_and( b[i], carry );
    carry = ntk.create_or( ab, ntk.create_or( ac, bc ) );
  }

  return sum;
}

template<class ntk_t>
std::vector<signal<ntk_t>>
ripple_sub( ntk_t& ntk,
            const std::vector<signal<ntk_t>>& a,
            const std::vector<signal<ntk_t>>& b )
{
  // a - b = a + two's complement of b
  auto b_2c = twos_complement( ntk, b );
  return ripple_add( ntk, a, b_2c );
}

template<class ntk_t>
std::optional<ntk_t> get_ntk( std::string const& key )
{
  // Read from file
  if ( key.find( '.' ) != std::string::npos )
  {
    return read_ntk<ntk_t>( key );
  }

  // Full adder
  if ( key == "fa" )
  {
    ntk_t ntk;
    const auto in1 = ntk.create_pi();
    const auto in2 = ntk.create_pi();
    const auto in3 = ntk.create_pi();
    const auto [s, c] = mockturtle::full_adder( ntk, in1, in2, in3 );
    ntk.create_po( s );
    ntk.create_po( c );
    return ntk;
  }

  if ( key == "mux" )
  {
    ntk_t ntk;

    const auto b_i = ntk.create_pi();
    const auto b_i_next = ntk.create_pi();
    const auto m = ntk.create_pi();

    const auto O1 = ntk.create_and( m, b_i_next );
    const auto O2 = ntk.create_and( ntk.create_not( m ), b_i );
    const auto bi = ntk.create_or( O1, O2 );
    ntk.create_po( bi );

    return ntk;
  }

  if ( key == "gt" )
  {
    ntk_t ntk;

    const auto a = ntk.create_pi();
    const auto b = ntk.create_pi();
    const auto c = ntk.create_pi();
    const auto f = ntk.create_pi();

    const auto flag = ntk.create_xor( a, b );
    const auto gt = ntk.create_and( ntk.create_not( f ), ntk.create_and( a, ntk.create_not( b ) ) );
    const auto found = ntk.create_or( flag, f );
    const auto cmp = ntk.create_or( c, gt );

    ntk.create_po( cmp );
    ntk.create_po( found );

    return ntk;
  }

  if ( key == "kogge_stone" )
  {
    ntk_t ntk;

    // Create primary inputs
    const auto A = ntk.create_pi();
    const auto B = ntk.create_pi();

    // Compute intermediate signals
    const auto X = ntk.create_xor( A, B );
    const auto P0 = ntk.create_or( A, B );
    const auto G0 = ntk.create_and( A, B );

    ntk.create_po( P0 );
    const auto shifted_P0 = ntk.create_pi();
    const auto P1 = ntk.create_and( P0, shifted_P0 );

    ntk.create_po( G0 );
    const auto shifted_G0 = ntk.create_pi();
    const auto G1 = ntk.create_or( G0, ntk.create_and( P0, shifted_G0 ) );

    ntk.create_po( G1 );
    const auto shifted_G1 = ntk.create_pi();
    ntk.create_po( P1 );
    const auto shifted_P1 = ntk.create_pi();
    const auto P2 = ntk.create_and( P1, shifted_P1 );
    const auto G2 = ntk.create_or( G1, ntk.create_and( P1, shifted_G1 ) );

    // const auto P3 = ntk.create_and(P2, ntk.create_shift(P2));
    ntk.create_po( G2 );
    const auto shifted_G2 = ntk.create_pi();
    const auto G3 = ntk.create_or( G2, ntk.create_and( P2, shifted_G2 ) );

    const auto Carry = G3;
    const auto shifted_G3 = ntk.create_pi();
    const auto S = ntk.create_xor( X, shifted_G3 );

    // Create primary outputs
    ntk.create_po( S );
    ntk.create_po( Carry );

    return ntk;
  }

  bool const is_add = key.starts_with( "add" );
  bool const is_mul = key.starts_with( "mul" );
  bool const is_pop = key.starts_with( "pop" );
  if ( is_add || is_mul || is_pop )
  {
    unsigned long n;
    try
    {
      auto const rest = std::string_view( key ).substr( 3 );
      n = std::stoul( rest.data() );
    }
    catch ( std::exception& )
    {
      std::cerr << "invalid number given in '" << key << "'" << std::endl;
      return {};
    }

    ntk_t ntk;
    auto const gen_inputs = [&] {
      std::vector<typename ntk_t::signal> signals;
      signals.reserve( n );
      for ( unsigned long i = 0; i < n; i++ )
      {
        signals.emplace_back( ntk.create_pi() );
      }
      return signals;
    };

    std::vector<typename ntk_t::signal> outputs;
    if ( is_add )
    {
      auto carry = ntk.get_constant( false );
      outputs = gen_inputs();
      mockturtle::carry_ripple_adder_inplace( ntk, outputs, gen_inputs(), carry );
    }
    else if ( is_mul )
    {
      outputs = mockturtle::carry_ripple_multiplier( ntk, gen_inputs(), gen_inputs() );
    }
    else if ( is_pop )
    {
      outputs = mockturtle::sideways_sum_adder( ntk, gen_inputs() );
    }
    else
    {
      throw;
    }

    for ( auto const& sig : outputs )
    {
      ntk.create_po( sig );
    }
    return ntk;
  }

  if ( key == "gt4" || key == "gt8" || key == "gt16" ||
       key == "gt32" || key == "gt64" )
  {
    ntk_t ntk;
    size_t bw = 0;
    if ( key == "gt4" )
      bw = 4;
    else if ( key == "gt8" )
      bw = 8;
    else if ( key == "gt16" )
      bw = 16;
    else if ( key == "gt32" )
      bw = 32;
    else if ( key == "gt64" )
      bw = 64;

    // Create PIs for each bit
    std::vector<signal<ntk_t>> a_bits( bw );
    std::vector<signal<ntk_t>> b_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
    {
      a_bits[i] = ntk.create_pi();
    }
    for ( size_t i = 0; i < bw; ++i )
    {
      b_bits[i] = ntk.create_pi();
    }

    // Ripple GT network
    signal<ntk_t> gt = ripple_gt( ntk, a_bits, b_bits );

    // Output
    ntk.create_po( gt );

    return ntk;
  }

  if ( key == "ge4" || key == "ge8" || key == "ge16" ||
       key == "ge32" || key == "ge64" )
  {
    ntk_t ntk;
    size_t bw = 0;
    if ( key == "ge4" )
      bw = 4;
    else if ( key == "ge8" )
      bw = 8;
    else if ( key == "ge16" )
      bw = 16;
    else if ( key == "ge32" )
      bw = 32;
    else if ( key == "ge64" )
      bw = 64;

    std::vector<signal<ntk_t>> a_bits( bw );
    std::vector<signal<ntk_t>> b_bits( bw );

    // create primary inputs
    for ( size_t i = 0; i < bw; ++i )
    {
      a_bits[i] = ntk.create_pi();
    }
    for ( size_t i = 0; i < bw; ++i )
    {
      b_bits[i] = ntk.create_pi();
    }

    // ge = NOT(a < b) = a > b OR a == b
    signal<ntk_t> gt = ripple_gt( ntk, a_bits, b_bits );
    signal<ntk_t> eq = ripple_eq( ntk, a_bits, b_bits );
    signal<ntk_t> ge = ntk.create_or( gt, eq );

    ntk.create_po( ge );
    return ntk;
  }

  if ( key == "eq4" || key == "eq8" || key == "eq16" ||
       key == "eq32" || key == "eq64" )
  {
    ntk_t ntk;
    size_t bw = 0;
    if ( key == "eq4" )
      bw = 4;
    else if ( key == "eq8" )
      bw = 8;
    else if ( key == "eq16" )
      bw = 16;
    else if ( key == "eq32" )
      bw = 32;
    else if ( key == "eq64" )
      bw = 64;

    std::vector<signal<ntk_t>> a_bits( bw );
    std::vector<signal<ntk_t>> b_bits( bw );

    // create primary inputs
    for ( size_t i = 0; i < bw; ++i )
    {
      a_bits[i] = ntk.create_pi();
    }
    for ( size_t i = 0; i < bw; ++i )
    {
      b_bits[i] = ntk.create_pi();
    }

    signal<ntk_t> eq = ripple_eq( ntk, a_bits, b_bits );
    ntk.create_po( eq );
    return ntk;
  }

  if ( key.find( "min" ) != std::string::npos || key.find( "max" ) != std::string::npos )
  {

    ntk_t ntk;

    size_t bw = 0;
    bool is_min = false, is_max = false;

    if ( key == "min4" )
    {
      bw = 4;
      is_min = true;
    }
    else if ( key == "min8" )
    {
      bw = 8;
      is_min = true;
    }
    else if ( key == "min16" )
    {
      bw = 16;
      is_min = true;
    }
    else if ( key == "min32" )
    {
      bw = 32;
      is_min = true;
    }
    else if ( key == "min64" )
    {
      bw = 64;
      is_min = true;
    }
    else if ( key == "max4" )
    {
      bw = 4;
      is_max = true;
    }
    else if ( key == "max8" )
    {
      bw = 8;
      is_max = true;
    }
    else if ( key == "max16" )
    {
      bw = 16;
      is_max = true;
    }
    else if ( key == "max32" )
    {
      bw = 32;
      is_max = true;
    }
    else if ( key == "max64" )
    {
      bw = 64;
      is_max = true;
    }
    else
      return std::nullopt;

    // Create PIs for each bit
    std::vector<signal<ntk_t>> a_bits( bw );
    std::vector<signal<ntk_t>> b_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
    {
      a_bits[i] = ntk.create_pi();
    }
    for ( size_t i = 0; i < bw; ++i )
    {
      b_bits[i] = ntk.create_pi();
    }

    // Compute a > b
    auto gt = ripple_gt( ntk, a_bits, b_bits );

    // Select output: min = (gt ? b : a), max = (gt ? a : b)
    std::vector<signal<ntk_t>> out_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
    {
      auto ai = a_bits[i];
      auto bi = b_bits[i];

      if ( is_min )
      {
        // min = if a > b then b else a
        out_bits[i] = ntk.create_or(
            ntk.create_and( gt, bi ),
            ntk.create_and( ntk.create_not( gt ), ai ) );
      }
      else if ( is_max )
      {
        // max = if a > b then a else b
        out_bits[i] = ntk.create_or(
            ntk.create_and( gt, ai ),
            ntk.create_and( ntk.create_not( gt ), bi ) );
      }
    }

    // Create POs
    for ( auto& bit : out_bits )
      ntk.create_po( bit );

    return ntk;
  }

  if ( key == "sub4" || key == "sub8" || key == "sub16" ||
       key == "sub32" || key == "sub64" )
  {
    ntk_t ntk;
    size_t bw = 0;

    if ( key == "sub4" )
      bw = 4;
    else if ( key == "sub8" )
      bw = 8;
    else if ( key == "sub16" )
      bw = 16;
    else if ( key == "sub32" )
      bw = 32;
    else if ( key == "sub64" )
      bw = 64;

    // Inputs
    std::vector<signal<ntk_t>> a_bits( bw );
    std::vector<signal<ntk_t>> b_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
    {
      a_bits[i] = ntk.create_pi();
    }
    for ( size_t i = 0; i < bw; ++i )
    {
      b_bits[i] = ntk.create_pi();
    }

    // a - b
    auto out_bits = ripple_sub( ntk, a_bits, b_bits );

    // Outputs
    for ( auto& bit : out_bits )
      ntk.create_po( bit );

    return ntk;
  }

  // DIV: div4, div8, div16, div32, div64
  // Unsigned restoring array divider. Inputs: a (dividend) then b (divisor),
  // each `bw` bits. Output: quotient q = a / b (`bw` bits).
  if ( key == "div4" || key == "div8" || key == "div16" ||
       key == "div32" || key == "div64" )
  {
    ntk_t ntk;
    size_t bw = 0;

    if ( key == "div4" )
      bw = 4;
    else if ( key == "div8" )
      bw = 8;
    else if ( key == "div16" )
      bw = 16;
    else if ( key == "div32" )
      bw = 32;
    else if ( key == "div64" )
      bw = 64;

    // Inputs: a (dividend) first, then b (divisor)
    std::vector<signal<ntk_t>> a_bits( bw );
    std::vector<signal<ntk_t>> b_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
      a_bits[i] = ntk.create_pi();
    for ( size_t i = 0; i < bw; ++i )
      b_bits[i] = ntk.create_pi();

    // Working remainder is one bit wider than the operands so the shifted-in
    // dividend bit never overflows.
    const size_t rw = bw + 1;

    // Divisor, zero-extended to rw bits.
    std::vector<signal<ntk_t>> b_ext( rw );
    for ( size_t i = 0; i < bw; ++i )
      b_ext[i] = b_bits[i];
    b_ext[bw] = ntk.get_constant( false );

    // Remainder, initialised to 0.
    std::vector<signal<ntk_t>> R( rw, ntk.get_constant( false ) );
    std::vector<signal<ntk_t>> q( bw );

    for ( int i = bw - 1; i >= 0; --i )
    {
      // R = (R << 1) | a[i]
      for ( int k = rw - 1; k >= 1; --k )
        R[k] = R[k - 1];
      R[0] = a_bits[i];

      // diff = R - b ; borrow (R < b) means this quotient bit is 0.
      auto diff = ripple_sub( ntk, R, b_ext );
      auto lt = ripple_gt( ntk, b_ext, R ); // b > R  <=>  R < b
      q[i] = ntk.create_not( lt );

      // Restore: keep R if it underflowed, otherwise take the subtracted value.
      for ( size_t k = 0; k < rw; ++k )
        R[k] = ntk.create_or(
            ntk.create_and( lt, R[k] ),
            ntk.create_and( ntk.create_not( lt ), diff[k] ) );
    }

    // Output the quotient.
    for ( auto& bit : q )
      ntk.create_po( bit );

    return ntk;
  }

  // ABS: abs4, abs8, abs16, abs32, abs64
  // Computes absolute value of signed integer in two's complement.
  // If sign bit (MSB) is 1 (negative), compute two's complement (~a + 1), else return a.
  if ( key == "abs4" || key == "abs8" || key == "abs16" ||
       key == "abs32" || key == "abs64" )
  {
    ntk_t ntk;
    size_t bw = 0;

    if ( key == "abs4" )
      bw = 4;
    else if ( key == "abs8" )
      bw = 8;
    else if ( key == "abs16" )
      bw = 16;
    else if ( key == "abs32" )
      bw = 32;
    else if ( key == "abs64" )
      bw = 64;

    // Input: a_bits (signed number in two's complement)
    std::vector<signal<ntk_t>> a_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
      a_bits[i] = ntk.create_pi();

    // Check if negative (MSB = sign bit)
    auto is_negative = a_bits[bw - 1];

    // Compute two's complement: ~a + 1
    auto negated = twos_complement( ntk, a_bits );

    // Select: if negative then negated else a
    std::vector<signal<ntk_t>> out_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
    {
      out_bits[i] = ntk.create_or(
          ntk.create_and( is_negative, negated[i] ),
          ntk.create_and( ntk.create_not( is_negative ), a_bits[i] ) );
    }

    // Outputs
    for ( auto& bit : out_bits )
      ntk.create_po( bit );

    return ntk;
  }

  // BITCOUNT: bitcount4, bitcount8, bitcount16, bitcount32, bitcount64
  // Counts the number of 1-bits in the input (popcount).
  // Uses sideways_sum_adder which builds a tree of full adders for parallel counting.
  if ( key == "bitcount4" || key == "bitcount8" || key == "bitcount16" ||
       key == "bitcount32" || key == "bitcount64" )
  {
    ntk_t ntk;
    size_t bw = 0;

    if ( key == "bitcount4" )
      bw = 4;
    else if ( key == "bitcount8" )
      bw = 8;
    else if ( key == "bitcount16" )
      bw = 16;
    else if ( key == "bitcount32" )
      bw = 32;
    else if ( key == "bitcount64" )
      bw = 64;

    std::vector<signal<ntk_t>> a_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
      a_bits[i] = ntk.create_pi();

    auto out_bits = mockturtle::sideways_sum_adder( ntk, a_bits );

    for ( auto& bit : out_bits )
      ntk.create_po( bit );

    return ntk;
  }

  // IF_ELSE (MUX): ifelse4, ifelse8, ifelse16, ifelse32, ifelse64
  // dst[i] = (mask[i] != 0) ? src1[i] : src2[i]  (per-bit ternary select)
  if ( key == "ifelse4" || key == "ifelse8" || key == "ifelse16" ||
       key == "ifelse32" || key == "ifelse64" )
  {
    ntk_t ntk;
    size_t bw = 0;

    if ( key == "ifelse4" )
      bw = 4;
    else if ( key == "ifelse8" )
      bw = 8;
    else if ( key == "ifelse16" )
      bw = 16;
    else if ( key == "ifelse32" )
      bw = 32;
    else if ( key == "ifelse64" )
      bw = 64;

    // Inputs: mask_bits, src1_bits (true branch), src2_bits (false branch)
    std::vector<signal<ntk_t>> mask_bits( bw );
    std::vector<signal<ntk_t>> src1_bits( bw );
    std::vector<signal<ntk_t>> src2_bits( bw );

    for ( size_t i = 0; i < bw; ++i )
    {
      mask_bits[i] = ntk.create_pi();
    }
    for ( size_t i = 0; i < bw; ++i )
    {
      src1_bits[i] = ntk.create_pi();
    }
    for ( size_t i = 0; i < bw; ++i )
    {
      src2_bits[i] = ntk.create_pi();
    }

    // Per-bit mux: out[i] = mask[i] ? src1[i] : src2[i]
    std::vector<signal<ntk_t>> out_bits( bw );
    for ( size_t i = 0; i < bw; ++i )
    {
      // out[i] = (mask[i] AND src1[i]) OR (NOT mask[i] AND src2[i])
      out_bits[i] = ntk.create_or(
          ntk.create_and( mask_bits[i], src1_bits[i] ),
          ntk.create_and( ntk.create_not( mask_bits[i] ), src2_bits[i] ) );
    }

    // Outputs
    for ( auto& bit : out_bits )
      ntk.create_po( bit );

    return ntk;
  }

  std::cerr << "invalid network '" << key << "'" << std::endl;
  return {};
}

/**
 * Read network from .aig/.pla/.verilog file
 */
template<class ntk_t>
std::optional<ntk_t> read_ntk( const std::string& path )
{
  ntk_t ntk;
  lorina::return_code res;
  if ( path.ends_with( ".aig" ) )
  {
    res = lorina::read_aiger( path, mockturtle::aiger_reader( ntk ) );
  }
  else if ( path.ends_with( ".pla" ) )
  {
    res = lorina::read_pla( path, mockturtle::pla_reader( ntk ) );
  }
  else if ( path.ends_with( ".verilog" ) )
  {
    res = lorina::read_verilog( path, mockturtle::verilog_reader( ntk ) );
  }
  else
  {
    std::cerr << "unknown file ending (file " << path << ")" << std::endl;
    return {};
  }
  if ( res != lorina::return_code::success )
  {
    std::cerr << "could not parse file " << path << std::endl;
    return {};
  }
  return ntk;
}
