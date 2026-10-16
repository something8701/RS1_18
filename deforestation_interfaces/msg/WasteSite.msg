# A piece of land waste (e.g. a garbage bag) found in the forest.

uint8 DETECTED=0     # seen from the air by the drone camera
uint8 CONFIRMED=1    # the Husky went there and saw it
uint8 NOT_FOUND=2    # the Husky went there and did not see it

uint32 id
# Position in the frame of the enclosing header (parrot1_odom)
geometry_msgs/Point position
# Red area seen from the air (m2)
float32 area_m2
# Detection confidence (0-100)
float32 confidence
uint8 status
# Photo the Husky took of it ('' if none yet)
string photo
