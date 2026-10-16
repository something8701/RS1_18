# One Husky inspection mission (a suspected area the drone flagged).
uint8 QUEUED=0
uint8 ACTIVE=1
uint8 INSPECTED=2
uint8 UNREACHABLE=3
uint8 CANCELLED=4

string label
string type                    # CLEARING, WASTE, ...
geometry_msgs/Point position   # the flag
geometry_msgs/Point area_min   # the suspected area's box: the drone's canopy loss
geometry_msgs/Point area_max   # around the flag (map frame)
float64 priority               # lower = sooner
uint8 attempts
uint8 status
geometry_msgs/Point[] path     # the search path once dispatched (z = heading)
uint16 reached                 # waypoints of it reached so far
string[] photos                # photos taken at this site
string result                  # the inspection report, when done
