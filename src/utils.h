#pragma once

#include <cstdlib>
#include <iostream>
#include <lorina/aiger.hpp>
#include <lorina/bench.hpp>
#include <lorina/blif.hpp>
#include <lorina/common.hpp>
#include <lorina/genlib.hpp>
#include <lorina/pla.hpp>
#include <lorina/verilog.hpp>
#include <mockturtle/generators/arithmetic.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/io/bench_reader.hpp>
#include <mockturtle/io/blif_reader.hpp>
#include <mockturtle/io/genlib_reader.hpp>
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
  else if ( key.starts_with( "add" ) )
  {
    try
    {
      auto rest = std::string_view( key ).substr( 3 );
      unsigned long n = std::stoul( rest.data() );

      ntk_t ntk;
      std::vector<typename ntk_t::signal> as;
      std::vector<typename ntk_t::signal> bs;
      for ( unsigned long i = 0; i < n; i++ )
      {
        as.emplace_back( ntk.create_pi() );
        bs.emplace_back( ntk.create_pi() );
      }
      auto carry = ntk.get_constant( false );
      mockturtle::carry_ripple_adder_inplace( ntk, as, bs, carry );
      for ( auto const& sig : as )
      {
        ntk.create_po( sig );
      }
      return ntk;
    }
    catch ( std::exception& e )
    {
      // ignore
    }
  }
  return read_ntk<ntk_t>( key );
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
