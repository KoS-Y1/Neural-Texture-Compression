add_library(Shader INTERFACE
        global_io.slang
)
target_include_directories(Shader INTERFACE ./)