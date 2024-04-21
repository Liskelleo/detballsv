###      encoding: utf-8        ###
###     auther: Liskelleo.      ###
###    software-licence: APL    ###
### e-mail: liskell@outlook.com ###
### home-page: GitHub@Liskelleo ###

import cv2
import random
import os.path
import numpy as np
from datetime import datetime


class detballsv:
    def __init__(self, filepath, frames, fps, scale, ball_diameter, times=None,
    error=3, calibrating_mode=False, frame_check=True):
        self.frames = int(frames)
        self.filepath = str(filepath)
        self.fps = int(fps)
        self.scale = float(scale)
        self.error = int(error)
        self.calibrating_mode = bool(calibrating_mode)
        if self.calibrating_mode:
            self.ball_diameter = [int(ball_diameter)]
            self.times = times
        else:
            self.ball_diameter = tuple(ball_diameter)
            self.times = list(times)
            if len(self.times) != 8:
                raise TypeError("关键帧总数为8!")
            for t in self.times:
                if type(t) != int:
                    raise TypeError("输入的帧数必为整数!")
        self.frame_check = bool(frame_check)

    @staticmethod
    def dict_filter(**kwargs):
        dic1, dic2 = {}, {}
        for key, values in kwargs.items():
            for entry in values:
                if 'ball1' in entry and entry['ball1'] is not None:
                    dic1[key] = entry['ball1']
                if 'ball2' in entry and entry['ball2'] is not None:
                    dic2[key] = entry['ball2']
        return dic1, dic2

    @staticmethod
    def data_check(raw_data, threshold=2):
        if len(raw_data) >= 3:
            mean_value = np.mean(raw_data)
            std_dev = np.std(raw_data)
            filtered_data = [x for x in raw_data if (mean_value - threshold * std_dev) <= x <= (mean_value + threshold * std_dev)]
            filtered_mean = np.mean(filtered_data)
        else:
            filtered_mean = np.mean(raw_data)
        return filtered_mean
    
    def calculate_speed(self, data, calcmode=0, factor=0.6, max_iterations=10000):
        datanum = factor * len(data)
        frame_interval = factor * int(max(map(int, data.keys())) - min(map(int, data.keys())))
        if calcmode:
            keys, values = list(data.keys()), list(data.values())
            times = int((len(keys) // 2) / 2) + 1
            data_points = [(values[i] - values[len(keys)-1-i])[0][0] / (int(keys[i]) - int(keys[len(keys)-1-i])) * self.fps * self.scale for i in range(int(times))]
            result = self.data_check(data_points)
            return result
        else:
            keys = list(data.keys())
            selected_pairs, data_points = [], []
            iteration, spill = 0, 0
            while len(selected_pairs) < datanum:
                pair = random.sample(keys, 2)
                key1, key2 = pair
                if abs(int(key1) - int(key2)) > frame_interval:
                    selected_pairs.append((key1, key2))
                iteration += 1
                if iteration > max_iterations:
                    spill = True
                    print("计算此速度时迭代次数已达到最大值, 请调整固定参数再试!")
                    break
            if not spill:
                for pair in selected_pairs:
                    key1, key2 = pair
                    time_interval = (int(key1) - int(key2)) / self.fps
                    pos_interval = (data[key1] - data[key2]).tolist()[0][0]
                    data_points.append(pos_interval * self.scale / time_interval)
                result = self.data_check(data_points)
                return result
            
    def detect_ball(self):
        vdict = {}
        filename =  os.path.basename(self.filepath)
        tempjudge, collision_moment = [np.infty], 0
        if self.calibrating_mode:
            flag = True
            box = []
        for image_num in range(self.frames):
            if self.times is not None:
                if image_num not in self.times:
                    continue
            tempstack = []
            image_path = "{}\\{} {:0>3d}.jpg".format(self.filepath, filename, image_num)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            if image is None:
                print("无法读取图像: {image_path}")
                return None
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            for d in self.ball_diameter:
                ball_info = {}
                circles = cv2.HoughCircles(
                    blurred, 
                    cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                    param1=250, param2=30,
                    minRadius=int(d/2 - self.error),
                    maxRadius=int(d/2 + self.error)
                )
                if self.calibrating_mode:
                    if circles is not None:
                        if self.frame_check:
                            for i in circles[0, :]:
                                center = (int(i[0]), int(i[1]))
                                radius = int(i[2])
                                cv2.circle(image, center, radius, (0, 255, 0), 2)
                                cv2.circle(image, center, 2, (0, 0, 255), 3)
                        if len(circles[0]) == 1:
                            circles = np.round(circles[0, :]).astype("int")
                            ball_info["ball"] = circles
                        else:
                            ball_info["ball"] = None
                    else:
                        ball_info["ball"] = None
                    tempstack.append(ball_info)
                else:
                    if circles is not None:
                        if self.frame_check:
                            for i in circles[0, :]:
                                center = (int(i[0]), int(i[1]))
                                radius = int(i[2])
                                cv2.circle(image, center, radius, (0, 255, 0), 2)
                                cv2.circle(image, center, 2, (0, 0, 255), 3)
                        if len(circles[0]) == 1:
                            circles = np.round(circles[0, :]).astype("int")
                            if d == self.ball_diameter[0]:
                                ball_info["ball1"] = circles
                            else:
                                ball_info["ball2"] = circles
                        else:
                            if d == self.ball_diameter[0]:
                                ball_info["ball1"] = None
                            else:
                                ball_info["ball2"] = None
                    else:
                        if d == self.ball_diameter[0]:
                            ball_info["ball1"] = None
                        else:
                            ball_info["ball2"] = None
                    tempstack.append(ball_info)
            if self.frame_check:
                if self.calibrating_mode:
                    colorjudge = True
                    height, width, channels = image.shape
                    ball_info = tempstack[0]["ball"]
                    if ball_info is None: 
                        ball_info = "Not Detected."
                        colorjudge = False
                    else:
                        ball_info = tempstack[0]["ball"].tolist()[0]
                    text_lines = [f"Balls Info (X, Y, Radius)",
                                f"BALL: {ball_info}",
                                f"Frame Rate: {self.fps} fps", 
                                f"Frame Scale: 1 / {int(1/self.scale)}",
                                f"Frame Size: {width} * {height}",
                                f"Detect Sensitivity: {self.error} px",
                                f"FRAME NUM: {image_num} / {self.frames}"]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 1
                    font_color = [(0, 0, 255), (0, 255, 0)]
                    y_position = 30
                    for line in text_lines:
                        if y_position == 30:
                            cv2.putText(image, line, (20, y_position), font, font_scale, font_color[1], font_thickness)
                        elif y_position == 55:
                            if not colorjudge:
                                cv2.putText(image, line, (20, y_position), font, font_scale, font_color[0], font_thickness)
                            else:
                                cv2.putText(image, line, (20, y_position), font, font_scale, font_color[1], font_thickness)
                        else:
                            cv2.putText(image, line, (width-250, height-y_position+65), font, font_scale, font_color[1], font_thickness)
                        y_position += 25
                else:
                    colorjudge1, colorjudge2 = True, True
                    height, width = image.shape
                    ball1_info, ball2_info = tempstack[0]["ball1"], tempstack[1]["ball2"]
                    if ball1_info is None: 
                        ball1_info = "Not Detected."
                        colorjudge1 = False
                    else:
                        ball1_info = tempstack[0]["ball1"].tolist()[0]
                    if ball2_info is None: 
                        ball2_info = "Not Detected."
                        colorjudge2 = False
                    else:
                        ball2_info = tempstack[1]["ball2"].tolist()[0]
                    text_lines = [f"Balls Info (X, Y, Radius)",
                                f"BALL-1: {ball1_info}",
                                f"BALL-2: {ball2_info}",
                                f"Frame Rate: {self.fps} fps", 
                                f"Frame Scale: 1 / {int(1/self.scale)}",
                                f"Frame Size: {width} * {height}",
                                f"Detect Sensitivity: {self.error} px",
                                f"FRAME NUM: {image_num+1} / {self.frames}"]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 1
                    font_color = [(0, 0, 255), (0, 255, 0)]
                    y_position = 30
                    for line in text_lines:
                        if y_position == 30:
                            cv2.putText(image, line, (20, y_position), font, font_scale, font_color[1], font_thickness)
                        elif y_position == 55:
                            if not colorjudge1:
                                cv2.putText(image, line, (20, y_position), font, font_scale, font_color[0], font_thickness)
                            else:
                                cv2.putText(image, line, (20, y_position), font, font_scale, font_color[1], font_thickness)
                        elif y_position == 80:
                            if not colorjudge2:
                                cv2.putText(image, line, (20, y_position), font, font_scale, font_color[0], font_thickness)
                            else:
                                cv2.putText(image, line, (20, y_position), font, font_scale, font_color[1], font_thickness)
                        else:
                            cv2.putText(image, line, (width-250, height-y_position+80), font, font_scale, font_color[1], font_thickness)
                        y_position += 25
                if self.calibrating_mode:
                    if colorjudge:
                        cv2.imshow('Ball(s) Detection', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    cv2.imshow('Ball(s) Detection', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            vdict[str(image_num)] = tempstack
            if self.calibrating_mode:
                if tempstack[0]["ball"] is not None:
                    distance = tempstack[0]["ball"].tolist()[0][0]
                    tempjudge.append(distance)
                    if distance <= tempjudge[0]:
                        if flag:
                            collision_moment = image_num
                        else:
                            if not len(box):
                                box.append(image_num)
                    else:
                        if flag:
                            flag = False
                    del tempjudge[0]
        if self.calibrating_mode:
            dic0 = {}
            if len(box): collision_moment = int(box[0] - 1)
            vdict0 = {key: vdict[key] for key in vdict if int(key) < collision_moment - 3}
            for key, values in vdict0.items():
                for entry in values:
                    if 'ball' in entry and entry['ball'] is not None:
                        dic0[key] = entry['ball']
            velocity = self.calculate_speed(dic0)
            print("注意: 速度方向向右为正!")
            print("标定小球碰撞前的速度为{}m/s.".format(velocity))
            with open("实验数据 - {}.txt".format(filename), "a", encoding="utf-8") as file:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"实验数据 - {filename} (当次运行: {current_time})\n")
                file.write("标定小球碰撞前的速度为{}m/s.\n".format(velocity))
        else:
            velocity = []
            calclist = [(self.times[i], self.times[i + 1]) for i in range(0, len(self.times), 2)]
            count = 0
            for c in calclist:
                if count % 2 == 0:
                    num, index = 0, 'ball1'
                else:
                    num, index = 1, 'ball2'
                k, v = c
                if k == -1 or v == -1:
                    velocity.append('--')
                else:
                    temp = []
                    for i in c:
                        try:
                            temp.append((i,vdict[str(i)][num][index].tolist()[0][0]))
                        except AttributeError:
                            raise Exception(f"关键帧{i}中球{num+1}识别失败!")
                    velocity.append((temp[0][1] - temp[1][1]) / (temp[0][0] - temp[1][0]) * self.fps * self.scale)
                count += 1
            if self.calibrating_mode:
                print("注意: 速度方向向右为正!")
                print("标定小球碰撞前的速度为{}m/s.".format(velocity[0]))
                with open("实验数据 - {}.txt".format(filename), "a", encoding="utf-8") as file:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"实验数据 - {filename} (当次运行: {current_time})\n")
                    file.write("标定小球碰撞前的速度为{}m/s.\n".format(velocity))
            else:
                print("注意: 速度方向向右为正!")
                print("球1碰撞前速度为{}m/s.".format(velocity[0]))
                print("球1碰撞后速度为{}m/s.".format(velocity[2]))
                print("球2碰撞前速度为{}m/s.".format(velocity[1]))
                print("球2碰撞后速度为{}m/s.".format(velocity[3]))
                with open("实验数据 - {}.txt".format(filename), "a", encoding="utf-8") as file:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"实验数据 - {filename} (当次运行: {current_time})\n")
                    file.write("球1碰撞前速度为{}m/s,".format(velocity[0]))
                    file.write("球1碰撞后速度为{}m/s.\n".format(velocity[2]))
                    file.write("球2碰撞前速度为{}m/s,".format(velocity[1]))
                    file.write("球2碰撞后速度为{}m/s.\n".format(velocity[3]))
