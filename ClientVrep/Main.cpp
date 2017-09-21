#include <QCoreApplication>
#include "Robot.h"
#include "Simulator.h"
#include <iostream>
#include <unistd.h>

extern "C" {
   #include "extApi.h"
    #include "v_repLib.h"
}

int main(int argc, char *argv[])
{
    Robot *robot;
    Simulator *vrep = new Simulator("127.0.0.1", 19999);
    if (vrep->connect() ==-1){
        std::cout << "Failed to Connect" << std::endl;
        return 0;
    }

    robot = new Robot(vrep, "Pioneer_p3dx");

    for (int i=0; i<3000; ++i)
    {
        std::cout << "Here we go... " << i << std::endl;
        robot->update();
        robot->writeGT();
        robot->writeSonars();
        extApi_sleepMs(50);
    }
    vrep->disconnect();
    exit(0);
}
