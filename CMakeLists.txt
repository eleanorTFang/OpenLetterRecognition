CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_C_FLAGS_RELEASE "-Wall -O3")
SET(CMAKE_C_FLAGS_DEBUG "-Wall -g")

SET(Boost_USE_STATIC_LIBS OFF) 
SET(Boost_USE_MULTITHREADED ON)  
SET(Boost_USE_STATIC_RUNTIME OFF) 

FIND_PACKAGE(Boost COMPONENTS system filesystem) 
IF(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS}) 
ENDIF()

FIND_PACKAGE(Qt4 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(CMAKE_CXX_FLAGS "-D_CREATE_MIDDLE_IMAGE -D_USING_QT4 -D_USING_WINDOW -D_ENABLE_INVOCATION")

SET(SOURCES
LetterImageCreator.cpp
LetterFeatureExtractor.cpp
LetterVoronoiSpecificer.cpp
LetterLabelCreator.cpp
LetterTrashCleaner.cpp
LetterIdentifier.cpp
LetterRecognizer.cpp
)

SET(HEADERS
LetterImageCreator.h
LetterFeatureExtractor.h
LetterVoronoiSpecificer.h
LetterLabelCreator.h
LetterTrashCleaner.h
LetterIdentifier.h
LetterRecognizer.h
)

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
LINK_DIRECTORIES(${PROJECT_LIBRARY_DIR})

ADD_EXECUTABLE(LetterRecognition ${SOURCES})
TARGET_LINK_LIBRARIES(LetterRecognition ${OpenCV_LIBS} ${QT_LIBRARIES} ${Boost_LIBRARIES})



