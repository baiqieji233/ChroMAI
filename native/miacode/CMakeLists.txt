cmake_minimum_required(VERSION 3.21)
project(maichart_muri LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTORCC ON)
find_package(Qt6 6.8 REQUIRED COMPONENTS Core Gui)
set(src "${CMAKE_CURRENT_SOURCE_DIR}/upstream/src")
if(NOT EXISTS "${MIACODE_SLIDE_DATA}")
  message(FATAL_ERROR "MIACODE_SLIDE_DATA must point to the verified reference dataset")
endif()
configure_file(slide_data.qrc.in slide_data.qrc @ONLY)
file(GLOB muri_sources "${src}/tools/muri/*.cpp")
list(FILTER muri_sources EXCLUDE REGEX "MuriDump.cpp$")
add_library(maichart_muri SHARED
  bridge.cpp ${muri_sources}
  ${src}/common/MuriTypes.cpp
  ${src}/common/OperationLog.cpp
  ${src}/common/DebugLog.cpp
  ${src}/core/chart/document/SimaiDocument.cpp
  ${src}/core/chart/document/SimaiTimingMetadata.cpp
  ${src}/core/chart/parser/SimaiNativeParser.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/slide_data.qrc")
target_link_libraries(maichart_muri PRIVATE Qt6::Core Qt6::Gui)
target_include_directories(maichart_muri PRIVATE ${src} ${src}/common
  ${src}/core/chart/document ${src}/core/chart/parser ${src}/timeline ${src}/tools)
if(MSVC)
  target_compile_options(maichart_muri PRIVATE /utf-8)
endif()
