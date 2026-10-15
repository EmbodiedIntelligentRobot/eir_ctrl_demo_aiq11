#!/usr/bin/env python3
"""Prepare a local RViz view of measured AIQ11 body joints; never starts a controller."""
import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="32-joint AIQ11 PRO control URDF")
    parser.add_argument("--visual-urdf", type=Path, help="Original visual URDF to restore body joints fixed in the control model")
    parser.add_argument("--meshes", type=Path, required=True, help="Local model mesh directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ET.parse(args.urdf).getroot()
    config = Path(__file__).resolve().parents[1] / "config/robots/pro.yaml"
    expected = set(yaml.safe_load(config.read_text())["joint_names"])
    if args.visual_urdf:
        visual = ET.parse(args.visual_urdf).getroot()
        for joint in list(root.findall("joint")):
            if joint.get("name") not in expected or joint.get("type") != "fixed":
                continue
            source = visual.find(f"./joint[@name='{joint.get('name')}']")
            if source is None or source.get("type") not in ("revolute", "continuous"):
                parser.error(f"Missing moving visual joint: {joint.get('name')}")
            for field in ("parent", "child", "origin"):
                if joint.find(field).attrib != source.find(field).attrib:
                    parser.error(f"Control and visual frames differ for {joint.get('name')}")
            root.remove(joint)
            root.append(source)
    moving = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    if moving != expected:
        parser.error(f"URDF moving joints differ from body feedback: missing={expected-moving}, extra={moving-expected}")
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = {link.get("name") for link in root.findall("link")} - children
    if roots != {"base_link"}:
        parser.error("Expected a control URDF rooted at base_link")
    for mesh in root.findall(".//mesh"):
        path = args.meshes.resolve() / Path(mesh.attrib["filename"]).name
        if not path.is_file():
            parser.error(f"Missing mesh: {path}")
        mesh.set("filename", path.as_uri())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("base_orientation_tf.py"), output / "base_orientation_tf.py")
    description = ET.tostring(root, encoding="unicode")
    (output / "body.urdf").write_text(description + "\n")
    params = {"/**": {"ros__parameters": {
        "robot_description": description, "use_sim_time": False, "publish_frequency": 50.0,
    }}}
    (output / "robot_state_publisher.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    rviz = {
        "Panels": [{"Class": "rviz_common/Displays", "Name": "Displays"}],
        "Visualization Manager": {
            "Global Options": {"Fixed Frame": "imu_world", "Frame Rate": 20},
            # An empty tool list leaves RViz without a current tool: no camera drag,
            # and Humble's ToolManager can dereference null on a key event.
            "Tools": [
                {"Class": "rviz_default_plugins/MoveCamera"},
                {"Class": "rviz_default_plugins/FocusCamera"},
            ],
            "Displays": [
                {"Class": "rviz_default_plugins/Grid", "Name": "Grid", "Enabled": True,
                 "Cell Size": 0.2, "Plane Cell Count": 20, "Reference Frame": "imu_world"},
                {"Class": "rviz_default_plugins/RobotModel", "Name": "Measured body joints",
                 "Enabled": True, "Visual Enabled": True, "Collision Enabled": False,
                 "Description Source": "Topic", "Description Topic": {
                     "Value": "/aiq11_joint_view/robot_description", "Reliability Policy": "Reliable",
                     "Durability Policy": "Transient Local", "History Policy": "Keep Last", "Depth": 1,
                 }},
            ],
            "Views": {"Current": {
                "Class": "rviz_default_plugins/Orbit", "Distance": 3.0, "Pitch": 0.2, "Yaw": 0.7,
                "Focal Point": {"X": 0.0, "Y": 0.0, "Z": -0.3},
            }},
        },
    }
    (output / "joints.rviz").write_text(yaml.safe_dump(rviz, sort_keys=False))
    script = output / "view.sh"
    script.write_text('''#!/usr/bin/env bash
set -eo pipefail
view_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher \\
    --ros-args -r __node:=aiq11_joint_direction_viewer \\
    -r robot_description:=/aiq11_joint_view/robot_description \\
    -r /tf:=/aiq11_joint_view/tf -r /tf_static:=/aiq11_joint_view/tf_static \\
    --params-file "$view_dir/robot_state_publisher.yaml" &
publisher_pid=$!
/usr/bin/python3 "$view_dir/base_orientation_tf.py" --ros-args \\
    -r /tf:=/aiq11_joint_view/tf &
orientation_pid=$!
cleanup() {
    kill "$publisher_pid" "$orientation_pid" 2>/dev/null || true
    wait "$publisher_pid" "$orientation_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
/opt/ros/humble/lib/rviz2/rviz2 -d "$view_dir/joints.rviz" --ros-args \\
    -r /tf:=/aiq11_joint_view/tf -r /tf_static:=/aiq11_joint_view/tf_static
''')
    script.chmod(0o755)
    print(f"Local viewer prepared: {script}\nInput: /joint_states and /base_orientation")


if __name__ == "__main__":
    main()
