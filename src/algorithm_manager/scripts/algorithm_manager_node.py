#!/usr/bin/env python3
import signal
import subprocess
import os
import time
import rospy
import threading
import psutil


class AlgorithmManager:
    def __init__(self):
        print("开始执行 AlgorithmManager")
        # 算法启动超时时间变量
        self.algorithm_start_timeout = 60
        # 存储已经运行的算法进程信息
        self.algorithm_processes = {}
        self.python_interpreter = "/home/nvidia/miniforge3/envs/yolov8/bin/python"

        # 存储算法名称和路径的字典列表
        self.algorithms = [
            {"name": "yolov8", "status":"stop", "conda_env": "yolov8", "command": "rosrun yolov8 detect.py"},
            {"name": "yolov8_trt", "status":"stop", "conda_env": "", "command": "rosrun infer detect_trt"},
            {"name": "pose", "status":"stop", "conda_env": "yolov8", "command": "rosrun yolov8 pose.py"},
            # {"name": "track", "path": "/home/image514/xjb/ultralytics-main/", "file": "track.py"},
            # {"name": "classify", "path": "/home/image514/xjb/ultralytics-main/", "file": "classify.py"},
            # {"name": "segment", "path": "/home/image514/xjb/ultralytics-main/", "file": "segment.py"},
            # {"name": "YOLO-Facev2", "path": "/home/image514/xjb/YOLO-FaceV2/", "file": "detect.py"},
            # {"name": "detect-yolov7-tensorRT", "path": "/home/image514/xjb/infer/workspace/", "command": "./infer cam_detect yolov7-640-fp16 v7 view-img"},
            # {"name": "detect-yolov7-tiny-tensorRT", "path": "/home/image514/xjb/infer/workspace/", "command": "./infer cam_detect yolov7-tiny-720-fp16 v7 view-img"},
            # {"name": "detect-yolov8n-tensorRT", "path": "/home/image514/xjb/infer/workspace/", "command": "./infer cam_detect yolov8n-fp16 v8 view-img"},
            # {"name": "detect-yolov8m-tensorRT", "path": "/home/image514/xjb/infer/workspace/", "command": "./infer cam_detect yolov8m-640-fp16 v8 view-img"},
        ]

        # 初始化rosparam参数
        self.init_ros_param()

        # 启动两个线程
        threading.Thread(target=self.loop_get_ros_param, daemon=True).start()
        threading.Thread(target=self.loop_check_algorithm, daemon=True).start()

    # 线程函数，循环获取ros参数服务器中的算法参数，更新算法状态
    def loop_get_ros_param(self):
        while True:
            if not rospy.has_param("algorithms"):
                time.sleep(1)
                continue
            # 获取算法参数
            algorithms = rospy.get_param("algorithms")
            # 更新算法状态
            for name, status in algorithms.items():
                index = self.get_algorithm_index(name)
                if index < 0:
                    print("算法名称不存在")
                    continue
                self.algorithms[index]["status"] = status
            time.sleep(1)

    # 根据算法名称获取算法在列表中的索引
    def get_algorithm_index(self, name):
        for index, algorithm in enumerate(self.algorithms):
            if algorithm["name"] == name:
                return index
        return -1

    # 线程函数，管理算法函数，循环检查算法状态，根据状态启动或结束该算法
    def loop_check_algorithm(self):
        while True:
            for index, algorithm in enumerate(self.algorithms):
                name = algorithm["name"]
                status = algorithm["status"]
                if status == "start":
                    if name not in self.algorithm_processes:
                        self.run_algorithm(index)
                    else:
                        # 如果60秒内算法没有启动，则重新启动
                        if self.check_processes(name):
                            algorithm["status"] = "keep"
                            rospy.set_param("algorithms/" + name, "keep")
                            self.algorithm_start_timeout = 60
                        else:
                            self.algorithm_start_timeout -= 1
                            if self.algorithm_start_timeout <= 0:
                                self.kill_algorithm(index)
                                self.run_algorithm(index)
                                self.algorithm_start_timeout = 60
                elif status == "keep":
                    # 如果算法异常结束，则重新启动
                    if not self.check_processes(name):
                        self.kill_algorithm(index)
                        algorithm["status"] = "start"
                        rospy.set_param("algorithms/" + name, "start")
                elif status == "stop":
                    if name in self.algorithm_processes:
                        self.kill_algorithm(index)
                else:
                    print(f"无效的算法状态: {status}")
            time.sleep(1)

    # 检查算法进程是否存在，不存在则删除
    def check_processes(self, name):
        pidList = self.algorithm_processes.get(name)
        if pidList is None:
            return False
        for process_id in pidList:
            if not psutil.pid_exists(process_id):
                print(f"进程 {name} 异常结束, 进程ID: {process_id},将重新启动。")
                return False
        return True

    # 初始化rosparam参数
    def init_ros_param(self):
        for algorithm in self.algorithms:
            name = algorithm["name"]
            status = algorithm["status"]
            rospy.set_param("algorithms/" + name, status)

    # 列出已有的算法
    def list_algorithms(self):
        print("\n--------------------------------------------------")
        print("已有算法信息：")
        print("--------------------------------------------------")
        for index, algorithm in enumerate(self.algorithms):
            status = "运行中" if algorithm["name"] in self.algorithm_processes else "未运行"
            print(f"{index}: {algorithm['name']} - {status}")
        print("--------------------------------------------------")

    # 运行算法
    def run_algorithm(self, index):
        if index < 0 or index >= len(self.algorithms):
            print("无效的算法序号")
            return

        algorithm = self.algorithms[index]
        name = algorithm["name"]
        if name in self.algorithm_processes:
            print(name, " 算法已经在运行中")
            return

        conda_env = algorithm['conda_env']
        print(f"conda_env: {conda_env}")
        if conda_env == "":
            command = algorithm["command"]
        else:
            print("conda_env is not None")
            # command = f"exec bash && source activate {conda_env} && " + algorithm["command"]
            command = f"conda run -n {conda_env} {algorithm['command']}"
        # if "file" in algorithm:
        #     algorithm_file = algorithm["file"]
        #     # command = f"cd {algorithm_path} && {python_interpreter} {algorithm_file}"
        #     command = f"cd {algorithm_path} && gnome-terminal -- /bin/bash -c '{python_interpreter} {algorithm_file}'"
        # else:
        #     command = f"cd {algorithm_path} && " + algorithm["command"]
        #     # command = f"bash {algorithm_path}"f"{name}"
        process = subprocess.Popen(command, shell=True)
        # 等待3秒，等算法启动后再获取进程id
        time.sleep(3)
        pidName = algorithm['command'].split(" ")[2] # 算法别名
        flag, pidlist = self.getPid(f"{pidName}")
        if flag:
            self.algorithm_processes[name] = pidlist
            print(f"已启动算法 '{name}'，进程ID: {self.algorithm_processes[name]}")
        else:
            print(f"获取算法 '{name}' ID失败")

    # 杀死进程命令
    def kill_algorithm(self, index):
        if index < 0 or index >= len(self.algorithms):
            print("无效的算法序号")
            return

        algorithm = self.algorithms[index]
        name = algorithm["name"]
        pidlist = self.algorithm_processes.get(name)
        if pidlist:
            for pid in pidlist:
                try:
                    os.kill(pid, signal.SIGINT)
                except OSError:
                    pass
            del self.algorithm_processes[name]
            print(f"已杀死算法进程，名称 {name}")
        else:
            print("该算法未在运行中")

    # 根据name得到对应进程pid
    def getPid(self, cmd: str) -> (bool, list):
        # getPidCmd = 'ps -aux|grep {keywords}|grep -v grep|cut -c 9-15|xargs'.format(
        #     keywords=cmd.replace(' ', '\ '))
        getPidCmd = f'ps -ef|grep {cmd}|grep -v grep|cut -c 9-15'
        out = []
        for retry in range(20):
            if len(out) == 2:
                break
            fp = os.popen(getPidCmd)
            out = fp.read().replace('\n', '').split(' ')
            time.sleep(0.1)
        pidlist = []
        for item in out:
            if (item != ''):
                pidlist.append(item)
        if len(pidlist) == 0:
            return False, []
        else:
            return True, list(map(int, pidlist))

    # 结束函数，结束所有子线程
    def __del__(self):
        print("等待已有进程退出")
        for process in self.algorithm_processes.values():
            for pid in process:
                try:
                    os.kill(pid, signal.SIGINT)
                except OSError:
                    pass
        print("所有算法已结束")


if __name__ == "__main__":
    rospy.init_node("algorithm_manager_node")

    algorithm_manager = AlgorithmManager()
    algorithm_manager.list_algorithms()

    rospy.spin()


