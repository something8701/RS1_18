# MissionInfo.msg: One Husky inspection mission (a suspected area the drone flagged)
uint8 QUEUED=0
uint8 ACTIVE=1
uint8 INSPECTED=2
uint8 UNREACHABLE=3
uint8 CANCELLED=4

string label
string type
geometry_msgs/Point position
geometry_msgs/Point area_min
geometry_msgs/Point area_max
float64 priority
uint8 attempts
uint8 status
geometry_msgs/Point[] path
uint16 reached
string[] photos
string result
