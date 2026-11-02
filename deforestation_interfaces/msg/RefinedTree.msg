# RefinedTree.msg: A drone tree checked on the ground by the Husky (or a trunk only the Husky found)
uint8 UNCHECKED=0
uint8 STANDING=1
uint8 NO_TRUNK=2
uint8 CUT_CONFIRMED=3
uint8 FALSE_LOST=4
uint8 HIDDEN=5

uint32 id
geometry_msgs/Point drone_position
geometry_msgs/Point ground_position
uint8 status
uint16 trunk_hits
uint16 views
