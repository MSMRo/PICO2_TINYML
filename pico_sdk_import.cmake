# ==============================================================================
# Raspberry Pi Pico SDK Import Script
# ==============================================================================

if (DEFINED ENV{PICO_SDK_PATH} AND (NOT PICO_SDK_PATH))
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
    message("Using PICO_SDK_PATH from environment ('${PICO_SDK_PATH}')")
endif ()

# Check default VS Code / Pico SDK user installation directory if not set
if (NOT PICO_SDK_PATH)
    if (EXISTS "$ENV{USERPROFILE}/.pico-sdk/sdk/2.3.1")
        set(PICO_SDK_PATH "$ENV{USERPROFILE}/.pico-sdk/sdk/2.3.1")
        message("Using PICO_SDK_PATH auto-detected at '${PICO_SDK_PATH}'")
    elseif (EXISTS "C:/Users/User/.pico-sdk/sdk/2.3.1")
        set(PICO_SDK_PATH "C:/Users/User/.pico-sdk/sdk/2.3.1")
        message("Using PICO_SDK_PATH auto-detected at '${PICO_SDK_PATH}'")
    endif ()
endif ()

# Point to pre-built picotool and pioasm to avoid compiling host tools
if (NOT picotool_DIR)
    if (EXISTS "$ENV{USERPROFILE}/.pico-sdk/picotool/2.3.1/picotool")
        set(picotool_DIR "$ENV{USERPROFILE}/.pico-sdk/picotool/2.3.1/picotool" CACHE PATH "Path to picotool")
    elseif (EXISTS "C:/Users/User/.pico-sdk/picotool/2.3.1/picotool")
        set(picotool_DIR "C:/Users/User/.pico-sdk/picotool/2.3.1/picotool" CACHE PATH "Path to picotool")
    endif ()
endif ()

if (NOT pioasm_DIR)
    if (EXISTS "$ENV{USERPROFILE}/.pico-sdk/tools/2.3.1/pioasm")
        set(pioasm_DIR "$ENV{USERPROFILE}/.pico-sdk/tools/2.3.1/pioasm" CACHE PATH "Path to pioasm")
    elseif (EXISTS "C:/Users/User/.pico-sdk/tools/2.3.1/pioasm")
        set(pioasm_DIR "C:/Users/User/.pico-sdk/tools/2.3.1/pioasm" CACHE PATH "Path to pioasm")
    endif ()
endif ()

if (DEFINED ENV{PICO_SDK_FETCH_FROM_GIT} AND (NOT PICO_SDK_FETCH_FROM_GIT))
    set(PICO_SDK_FETCH_FROM_GIT $ENV{PICO_SDK_FETCH_FROM_GIT})
    message("Using PICO_SDK_FETCH_FROM_GIT from environment ('${PICO_SDK_FETCH_FROM_GIT}')")
endif ()

if (DEFINED ENV{PICO_SDK_FETCH_FROM_GIT_PATH} AND (NOT PICO_SDK_FETCH_FROM_GIT_PATH))
    set(PICO_SDK_FETCH_FROM_GIT_PATH $ENV{PICO_SDK_FETCH_FROM_GIT_PATH})
    message("Using PICO_SDK_FETCH_FROM_GIT_PATH from environment ('${PICO_SDK_FETCH_FROM_GIT_PATH}')")
endif ()

set(PICO_SDK_PATH "${PICO_SDK_PATH}" CACHE PATH "Path to the Raspberry Pi Pico SDK")
set(PICO_SDK_FETCH_FROM_GIT "${PICO_SDK_FETCH_FROM_GIT}" CACHE BOOL "Set to ON to fetch copy of SDK from git if not otherwise locatable")
set(PICO_SDK_FETCH_FROM_GIT_PATH "${PICO_SDK_FETCH_FROM_GIT_PATH}" CACHE FILEPATH "location to download SDK")

if (NOT PICO_SDK_PATH)
    if (PICO_SDK_FETCH_FROM_GIT)
        include(FetchContent)
        set(FETCHCONTENT_BASE_DIR_SAVE ${FETCHCONTENT_BASE_DIR})
        if (PICO_SDK_FETCH_FROM_GIT_PATH)
            get_filename_component(FETCHCONTENT_BASE_DIR "${PICO_SDK_FETCH_FROM_GIT_PATH}" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
        endif ()
        if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
            FetchContent_Declare(
                    pico_sdk
                    GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
                    GIT_TAG master
                    GIT_SUBMODULES_RECURSE FALSE
            )
        else ()
            FetchContent_Declare(
                    pico_sdk
                    GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
                    GIT_TAG master
            )
        endif ()

        if (NOT pico_sdk)
            message("Downloading Raspberry Pi Pico SDK")
            FetchContent_Populate(pico_sdk)
            set(PICO_SDK_PATH ${pico_sdk_SOURCE_DIR})
        endif ()
        set(FETCHCONTENT_BASE_DIR ${FETCHCONTENT_BASE_DIR_SAVE})
    else ()
        message(FATAL_ERROR
                "SDK location was not specified. Please set PICO_SDK_PATH or set PICO_SDK_FETCH_FROM_GIT to on to fetch from git."
                )
    endif ()
endif ()

get_filename_component(PICO_SDK_PATH "${PICO_SDK_PATH}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
if (NOT EXISTS "${PICO_SDK_PATH}")
    message(FATAL_ERROR "Directory '${PICO_SDK_PATH}' for PICO_SDK_PATH does not exist")
endif ()

set(PICO_SDK_INIT_CMAKE_FILE ${PICO_SDK_PATH}/pico_sdk_init.cmake)
if (NOT EXISTS ${PICO_SDK_INIT_CMAKE_FILE})
    message(FATAL_ERROR "Directory '${PICO_SDK_PATH}' does not appear to contain the Raspberry Pi Pico SDK")
endif ()

include(${PICO_SDK_INIT_CMAKE_FILE})
