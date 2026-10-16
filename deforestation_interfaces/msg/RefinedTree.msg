# A drone tree checked on the ground by the Husky (or a trunk only the Husky found).

uint8 UNCHECKED=0      # the Husky has not looked at it yet
uint8 STANDING=1       # its trunk is there
uint8 NO_TRUNK=2       # looked, no trunk (a crown with no trunk under it)
uint8 CUT_CONFIRMED=3  # the drone reported it LOST and there is no trunk
uint8 FALSE_LOST=4     # the drone reported it LOST but the trunk is still there
uint8 HIDDEN=5         # a trunk the drone never had (e.g. under another crown)

uint32 id                           # drone tree id, 0 for HIDDEN
geometry_msgs/Point drone_position
geometry_msgs/Point ground_position # the trunk, when the Husky saw one
uint8 status
uint16 trunk_hits                   # LiDAR sightings of the trunk
uint16 views                        # LiDAR beams that passed clear through the spot
