#!/usr/bin/env python3
import rospy

if __name__ == "__main__":
    rospy.init_node("set_update_paramter_p")

    # 设置各种类型参数
    rospy.set_param("algorithms",{"yolov8":"start", "yolov8_pose":"start"})

    # pdict = rospy.get_param("p_dict")
    # print(pdict)
    # pdict["yolov8"] = "stop"
    # rospy.set_param("p_dict", pdict)
    # pdict = rospy.get_param("p_dict")
    # print(pdict)


