#pragma once

#include <iostream>
#include <lorina/aiger.hpp>
#include <lorina/bench.hpp>
#include <lorina/common.hpp>
#include <lorina/genlib.hpp>
#include <lorina/pla.hpp>
#include <lorina/verilog.hpp>
#include <mockturtle/generators/arithmetic.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/io/pla_reader.hpp>
#include <mockturtle/io/verilog_reader.hpp>
#include <mockturtle/networks/mig.hpp>

#include <optional>
#include <string>
#include <string_view>

template<class ntk>
std::optional<ntk> get_ntk( std::string const& key );

template<class ntk>
std::optional<ntk> read_ntk( std::string const& path );

void preoptimize_mig( mockturtle::mig_network& ntk );

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
    auto in1 = ntk.create_pi();
    auto in2 = ntk.create_pi();
    auto in3 = ntk.create_pi();
    auto [s, c] = mockturtle::full_adder( ntk, in1, in2, in3 );
    ntk.create_po( s );
    ntk.create_po( c );
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

  std::cerr << "invalid network '" << key << "'" << std::endl;
  return {};
}

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
