# TreeStatus.msg: Tree counts of a tree tracker (/tree_status from the Husky's tree_mapper, /parrot_tree_status from …
std_msgs/Header header

int32 baseline_count
int32 current_count
int32 lost_count
int32 gained_count
