cmake_minimum_required(VERSION 2.8)
project( qr_code )

find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS program_options REQUIRED )
add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} ${Boost_LIBRARIES})
