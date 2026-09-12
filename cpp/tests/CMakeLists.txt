set(FIXTURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

function(bardecode_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bardecode_lib GTest::gtest_main)
  target_compile_definitions(${name} PRIVATE FIXTURES_DIR="${FIXTURES_DIR}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "PATH=$<TARGET_FILE_DIR:bardecode>;$<TARGET_FILE_DIR:ZXing>")
endfunction()

bardecode_test(test_detector)
bardecode_test(test_decoder)
bardecode_test(test_image_io)
bardecode_test(test_pipeline)

# default_model_path() resolves <exe_dir>/models/yolov8s_barcode.onnx
set(MODEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../bardecode/models/yolov8s_barcode.onnx")
if(EXISTS "${MODEL_SRC}")
  add_custom_command(TARGET test_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_SRC}"
            "$<TARGET_FILE_DIR:test_pipeline>/models/yolov8s_barcode.onnx")
endif()
