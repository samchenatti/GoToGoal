#include "Simulator.h"

Simulator::Simulator(std::string ip, int portNumber)
{
    this->id = -1;
    this->ip = ip;
    this->portNumber = portNumber;
}

int Simulator::connect()
{
    id = 1;
    // Connection to server
    id = simxStart(ip.c_str(), portNumber, true, true, 2000, 5);
    if (id == -1)
        throw std::string("Unable to connect to V-REP Server");
    return id;
}

void Simulator::disconnect()
{
    if (id != -1)
        simxFinish(id);
}

void Simulator::pause()
{
    if (id != -1)
        simxPauseCommunication(id,0);
}

void Simulator::resume()
{
    if (id != -1)
        simxPauseCommunication(id,1);
}

int Simulator::getHandle(std::string name)
{
    int handle=-1;
    if (id != -1)
        if (simxGetObjectHandle(id, name.c_str(), &handle, simx_opmode_oneshot_wait) != simx_return_ok)
            throw std::string("Unable to receive handle");
    return handle;

}

int Simulator::readProximitySensor(simxInt sensorHandle, simxUChar *state, float *coord)
{
    if (id != -1)
        simxReadProximitySensor(id,sensorHandle,state,coord,NULL,NULL,simx_opmode_buffer);
    return 1;
}

int Simulator::getObjectPosition(simxInt sensorHandle, float *coord)
{
    if (id != -1)
        simxGetObjectPosition(id,sensorHandle,-1,coord,simx_opmode_streaming);
    return 1;
}

int Simulator::getObjectOrientation(simxInt sensorHandle, float *coord)
{
    if (id != -1)
        simxGetObjectOrientation(id,sensorHandle,-1,coord,simx_opmode_streaming);
    return 1;
}

int Simulator::getJointPosition(simxInt jointHandle, float *coord)
{
    if (id != -1)
        simxGetJointPosition(id,jointHandle,coord,simx_opmode_streaming);
    return 1;
}

int Simulator::setJointTargetVelocity(simxInt jointHandle, float velocity) {
    if (id != -1)
        simxSetJointTargetVelocity(id, jointHandle, velocity, simx_opmode_streaming);
    return 1;
}

