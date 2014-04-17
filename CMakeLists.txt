CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_C_FLAGS_RELEASE "-Wall -O3")
SET(CMAKE_C_FLAGS_DEBUG "-Wall -g")

FIND_PACKAGE(Qt4 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(CMAKE_CXX_FLAGS "-D_CREATE_MIDDLE_IMAGE")

SET(SOURCES
LetterImageCreator.cpp
LetterFeatureExtractor.cpp
LetterVoronoiSpecificer.cpp
LetterLabelCreator.cpp
LetterFineExclusioner.cpp
LetterIdentificationer.cpp
LetterFacade.cpp
)

SET(HEADERS
LetterImageCreator.h
LetterFeatureExtractor.h
LetterVoronoiSpecificer.h
LetterLabelCreator.h
LetterFineExclusioner.h
LetterIdentificationer.h
)

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
LINK_DIRECTORIES(${PROJECT_LIBRARY_DIR})

ADD_EXECUTABLE(LetterRecognition ${SOURCES})
TARGET_LINK_LIBRARIES(LetterRecognition ${OpenCV_LIBS} ${QT_LIBRARIES})



