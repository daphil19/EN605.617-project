include(ExternalProject)
# use externalProject instead of fetchContent because fftw's cmake configuration is trash
# (externalProject allows us to customize configure/make/make install whereas fetchcontent assumes cmake targets only)
ExternalProject_Add(
    fftw
    URL http://fftw.org/fftw-3.3.10.tar.gz
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-shared --enable-threads --enable-sse2 --enable-avx --enable-avx2
)

add_library(fftw_benchmark fftw_driver.cpp FFTWPerformer.cpp)

# manually identify the fftw installation
ExternalProject_Get_Property(fftw INSTALL_DIR)
# the executable needs to know where to find libs to link
target_link_directories(fftw_benchmark PUBLIC ${INSTALL_DIR}/lib)
target_include_directories(fftw_benchmark PRIVATE ${INSTALL_DIR}/include)
target_link_libraries(fftw_benchmark fftw3_threads fftw3 m io)
add_dependencies(fftw_benchmark fftw)
