# WasteSite.msg: A piece of land waste (e.g
uint8 DETECTED=0
uint8 CONFIRMED=1
uint8 NOT_FOUND=2

uint32 id
geometry_msgs/Point position
float32 area_m2
float32 confidence
uint8 status
string photo
