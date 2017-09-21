# Absolute path for V-REP programming folder:
# Examples:
#   PATH = /Applications/V-REP_PRO_EDU_V3_0_4_Mac/programming    # Typical MacOS config
#   PATH = C:\Program Files\V-REP3\V-REP_PRO_EDU\programming    # Typical MacOS config


win32{
    PATH = "C:\Program Files (x86)\V-REP3\V-REP_PRO_EDU\programming"
}
macx{
    PATH = /Applications/V-REP_PRO_EDU_V3_0_4_Mac/programming
}
unix{
    PATH = /Applications/V-REP_PRO_EDU_V3_0_4_Mac/programming
}


#--------------------------------
#General purpose definitions:
#--------------------------------


win32{
    INCLUDEPATH += $${PATH}\remoteApi
    INCLUDEPATH += $${PATH}\include
    INCLUDEPATH += $${PATH}\common
}
macx{
    INCLUDEPATH += $${PATH}/remoteApi
    INCLUDEPATH += $${PATH}/include
    INCLUDEPATH += $${PATH}/common
}
unix{
    INCLUDEPATH += $${PATH}/remoteApi
    INCLUDEPATH += $${PATH}/include
    INCLUDEPATH += $${PATH}/common
}
DEFINES += "MAX_EXT_API_CONNECTIONS=255"
DEFINES += "NON_MATLAB_PARSING"
QMAKE_LFLAGS += -pthread


QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets


#--------------------------------
# Libs:
#--------------------------------


win32{
    LIBS += -lwinmm
    LIBS += -lwsock32
}
unix{
    LIBS=-ldl
}
#--------------------------------
# Targets:
#--------------------------------


TARGET = main
TEMPLATE = app


SOURCES += Main.cpp \
    Robot.cpp \
    Simulator.cpp

HEADERS += $${PATH}/include/v_repLib.h \
    Robot.h \
    Simulator.h

win32{
    SOURCES += $${PATH}\remoteApi\extApi.c
    SOURCES += $${PATH}\remoteApi\extApiPlatform.c
    #SOURCES += $${PATH}\common\v_repLib.cpp
}
macx{
    SOURCES += $${PATH}/remoteApi/extApi.c
    SOURCES += $${PATH}/remoteApi/extApiPlatform.c
    SOURCES += $${PATH}/common/v_repLib.cpp
}
unix{
    SOURCES += $${PATH}/remoteApi/extApi.c
    SOURCES += $${PATH}/remoteApi/extApiPlatform.c
    SOURCES += $${PATH}/common/v_repLib.cpp
}


win32{
    #HEADERS += $${PATH}\include\v_repLib.h
}
macx{
    HEADERS += $${PATH}/include/v_repLib.h
}
unix{
    HEADERS += $${PATH}/include/v_repLib.h
}
unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}
#OTHER_FILES +=
