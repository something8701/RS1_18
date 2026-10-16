# What the mission coordinator is doing, for dashboards and debugging.
std_msgs/Header header
string state                   # WAITING_FOR_DRONE, IDLE, DISPATCHING, ENROUTE, SEARCHING
bool drone_ready               # the drone's tree baseline is frozen
MissionInfo current            # empty label when idle
MissionInfo[] queue
MissionInfo[] done             # finished missions, oldest first
uint32 inspected
uint32 total
