std_msgs/Header header

# Position of the suspicious area in the frame given by header.frame_id
geometry_msgs/Point position

# Detection type: GAP, ANOMALY, or LINE
string type

# Detection confidence (0-100)
float64 confidence

# Affected area in square metres
float64 area_m2

# Composite severity score (0-100), computed from confidence × area
float64 severity

# Human-readable description
string description
