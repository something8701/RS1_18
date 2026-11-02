# MissionQueue.msg: What the mission coordinator is doing, for dashboards and debugging
std_msgs/Header header
string state
bool drone_ready
MissionInfo current
MissionInfo[] queue
MissionInfo[] done
uint32 inspected
uint32 total
