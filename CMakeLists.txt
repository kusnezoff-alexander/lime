cmake_minimum_required(VERSION 3.19)

project(lime)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(mockturtle)

include(FetchContent)
FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH rs/Cargo.toml)

add_executable(lime src/main.cpp)
target_link_libraries(lime PRIVATE mockturtle lime_rs)