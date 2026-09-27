std_msgs/Header header

# LOST or GAINED
string event_type

# Stable tree identifier assigned from the baseline, or newly assigned for growth
int32 tree_id

# Tree location in header.frame_id
float64 x
float64 y

# Affected area in square metres
float64 area_m2
