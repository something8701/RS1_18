# test_mission_control.py: Operator control of the Husky (mission_coordinator /mission_command): the operator approves, …
import json

import pytest
import rclpy
from std_msgs.msg import String

from deforestation_interfaces.msg import MissionInfo
from deforestation_monitoring.mission_coordinator import Mission, MissionCoordinator


@pytest.fixture(scope='module')
def ros_ctx():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros_ctx):
    n = MissionCoordinator()
    n.drone_ready = True
    n.nav_client.server_is_ready = lambda: True
    n.started = []
    n._send_goal = lambda m: n.started.append(m.label)
    for i, label in enumerate(['Area 1', 'Area 2', 'Area 3']):
        n.mission_queue.append(Mission(float(i), float(i), 0.0, 'CLEARING', 90.0, 4.0, 50.0, label, 'map', seq=i))
    n.robot_x = None
    yield n
    n.destroy_node()


def cmd(n, **c):
    acks = []
    n.ack_pub.publish = lambda m: acks.append(json.loads(m.data))
    n._command_cb(String(data=json.dumps(c)))
    return acks[-1]


def test_without_asking_the_husky_starts_the_best_area_itself(node):
    node.process_queue()
    assert node.started == ['Area 1']


def test_asking_first_waits_for_proceed_and_says_what_is_next(node):
    assert cmd(node, cmd='ask', on=True)['ok']
    node.process_queue()
    assert node.started == []
    control = []
    node.control_pub.publish = lambda m: control.append(json.loads(m.data))
    node._publish_status()
    assert control[-1]['next'] == 'Area 1' and control[-1]['ask_first']
    a = cmd(node, cmd='proceed')
    assert a['ok'] and 'Area 1' in a['msg']
    node.process_queue()
    assert node.started == ['Area 1']


def test_choosing_another_area_makes_it_next(node):
    cmd(node, cmd='ask', on=True)
    assert cmd(node, cmd='choose', label='Area 3')['ok']
    node.process_queue()
    assert node.started == ['Area 3']


def test_choose_without_asking_reorders_and_unpin_undoes_it(node):
    cmd(node, cmd='choose', label='Area 2')
    assert node._next_label() == 'Area 2'
    assert cmd(node, cmd='unpin', label='Area 2')['ok']
    assert node._next_label() == 'Area 1'


def test_skip_and_unskip_are_reversible(node):
    a = cmd(node, cmd='skip', label='Area 1')
    assert a['ok'] and [m.label for m in node.mission_queue].count('Area 1') == 0
    skipped = [m for m in node.done if m.label == 'Area 1'][0]
    assert skipped.status == MissionInfo.CANCELLED
    assert cmd(node, cmd='unskip', label='Area 1')['ok']
    assert 'Area 1' in [m.label for m in node.mission_queue]
    assert not [m for m in node.done if m.label == 'Area 1']


def test_pause_holds_the_next_start_and_resume_carries_on(node):
    cmd(node, cmd='pause')
    node.process_queue()
    assert node.started == []
    assert cmd(node, cmd='resume')['ok']
    node.process_queue()
    assert node.started == ['Area 1']


def test_commands_that_cannot_be_carried_out_say_why(node):
    a = cmd(node, cmd='choose', label='Area 9')
    assert not a['ok'] and 'not waiting' in a['msg']
    a = cmd(node, cmd='unskip', label='Area 2')
    assert not a['ok'] and 'was not skipped' in a['msg']
    a = cmd(node, cmd='fly', label='x')
    assert not a['ok'] and 'Unknown command' in a['msg']
    a = cmd(node, id='42', cmd='skip', label='Area 2')
    assert a['id'] == '42'
