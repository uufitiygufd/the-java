import numpy as np
import csv
import sys
import os
import time
import glob
import pyqtgraph as pg
from numpy import pi
from scipy.signal import butter, sosfilt
import serial  # 导入模块
from PyQt5 import QtCore
from multiprocessing import Value

from acconeer.exptool import configs, utils
from acconeer.exptool.pg_process import PGProccessDiedException
import acconeer.exptool as et

import pss2
start_state = Value('b', False)
reset_state = Value('b',False,lock=False)
lungcapacity = 0

def main():
    args = utils.ExampleArgumentParser(num_sens=1).parse_args()
    utils.config_logging(args)

    port = args.serial_port or utils.autodetect_serial_port()

    ser = serial.Serial(port, 1000000, timeout=200)

    ser.flushInput()
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
    else:
        print("打开串口失败。")

    ser.flushInput()

    sensor_config = get_sensor_config()
    processing_config = get_processing_config()
    session_info = {'range_start_m': 0.3, 'range_length_m': 0.5, 'data_length': 1034, 'stitch_count': 0, 'step_length_m': 0.000484, 'depth_lowpass_cutoff_ratio': 0.004035}

    pg_updater = PGUpdater(sensor_config, processing_config, session_info)
    pg_process = pss2.PGProcess(pg_updater, start_state, reset_state, lungcapacity)
    pg_process.start()
    while True:
        if start_state.value:
            break
        else:
            time.sleep(0.1)

    interrupt_handler = utils.ExampleInterruptHandler()
    processing_config.hist_plot_len = 30

    processor = BreathingProcessor(sensor_config, processing_config, session_info)

    while not interrupt_handler.got_signal:
        buf_1 = ser.read(2)
        start_marker = int.from_bytes(buf_1[0:2], 'little')
        while start_marker != 0x55AA:  #
            ser.flushInput()#避免出现收不到55AA重新连接串口的情况
            buf_1 = ser.read(2)
            start_marker = int.from_bytes(buf_1[0:2], 'little')

        buf_1 = ser.read(4)
        packet_len = int.from_bytes(buf_1[2:4], 'little')
        packet = ser.read(packet_len)

        data = np.frombuffer(packet, dtype="<i2").astype("float")
        comdat=data.reshape((-1, 2)).view(dtype="complex").flatten()
        plot_data = processor.process(comdat)  # data 1034 complex

        if reset_state.value:
            break

        if plot_data is not None:
            try:
                pg_process.put_data(plot_data)
            except PGProccessDiedException:
                break

    print("失去连接...")
    pg_process.close()
    ser.close()
    print("重置")
    sys.stdout.flush()
    print(sys.argv)
    python = sys.executable
    os.execl(python, 'python', *sys.argv)

def get_sensor_config():
    config = configs.IQServiceConfig()
    config.range_interval = [0.3, 0.8]
    config.update_rate = 20  #80/(66/16.2)=19.6
    config.gain = 0.5
    config.repetition_mode = configs.IQServiceConfig.RepetitionMode.SENSOR_DRIVEN
    return config

class ProcessingConfiguration(et.configbase.ProcessingConfig):
    VERSION = 1

    hist_plot_len = et.configbase.FloatParameter(
        label="Plot length",
        unit="s",
        default_value=30,
        limits=(1, 30),
        decimals=0,
    )

get_processing_config = ProcessingConfiguration

class BreathingProcessor:
    peak_hist_len = 1800
    phase_weights_alpha = 0.23
    peak_loc_alpha = 0.24
    sweep_alpha = 0.17
    env_alpha = 0.24

    k = 0
    k1 = 0
    lungcapacity_text = "正在检测中..."
    lung_VT_text = None
    lung_VT = 500
    lung_ERV_text = None
    lung_ERV = 1700
    lung_IRV_text = None
    lung_IRV = 2200
    lung_VE_text = None
    lung_VE = 6


    def __init__(self, sensor_config, processing_config, session_info):
        assert sensor_config.update_rate is not None

        self.f = sensor_config.update_rate
        self.hist_plot_len = int(round(processing_config.hist_plot_len * self.f))

        self.peak_history = np.zeros(self.peak_hist_len, dtype="complex")
        self.movement_history = np.zeros(self.peak_hist_len, dtype="float")
        self.breath_history = np.zeros(self.hist_plot_len, dtype="float")

        self.breath_sos = butter(2, (2 * 0.05 / self.f, 2 * 0.5 / self.f), btype='bandpass', analog=False, output='sos', fs=None)
        self.breath_zi = np.zeros((2, 2))
        # self.breath_sos = butter(2, 2 * 0.5 / self.f, btype='lowpass', analog=False, output='sos',fs=None)
        # self.breath_zi = np.zeros((1, 2))

        self.last_lp_sweep = None
        self.lp_phase_weights = None
        self.lp_sweep = None
        self.lp_peak_loc = 0

        self.sweep_index = 0

        self.depths1 = et.utils.get_range_depths(sensor_config, session_info)

        self.dir = os.path.dirname(sys.path[0])
        self.dir_record = os.path.join(self.dir, 'record')
        self.dir_recordfile = os.path.join(self.dir_record, '*')
        self.record_list = glob.glob(self.dir_recordfile)
        self.dir_name = os.path.join(self.dir, 'name')
        self.dir_namefile = os.path.join(self.dir_name, 'name.txt')
        self.filename = open(self.dir_namefile, 'r')
        self.name = self.filename.read()
        self.filename.close()
        self.dir_genderfile = os.path.join(self.dir_name, 'gender.txt')
        self.filegender = open(self.dir_genderfile, 'r')
        self.gender = self.filegender.read()
        self.filegender.close()
        self.dir_heightfile = os.path.join(self.dir_name, 'height.txt')
        self.fileheight = open(self.dir_heightfile, 'r')
        self.height = self.fileheight.read()
        self.fileheight.close()
        self.dir_weightfile = os.path.join(self.dir_name, 'weight.txt')
        self.fileweight = open(self.dir_weightfile, 'r')
        self.weight = self.fileweight.read()
        self.fileweight.close()
        self.height = float(self.height)
        self.weight = float(self.weight)
    def process(self, sweep):
        # a = np.array([0])
        # name = str(len(self.record_list) + 1) + str(self.name) + '.csv'
        # with open(os.path.join(self.dir_record, name), 'ab') as f:
        #     np.savetxt(f, sweep, delimiter=',', newline=',')
        #     np.savetxt(f, a, delimiter=',')

        if self.sweep_index == 0:
            self.lp_sweep = np.array(sweep)
            self.lp_env = np.abs(sweep)
            self.lp_peak_loc = np.argmax(self.lp_env)

            out_data = None
        else:
            self.lp_sweep = self.lp(sweep, self.lp_sweep, self.sweep_alpha)    #线性滤波，旧信号占0.17，新信号占0.83
            env = np.abs(self.lp_sweep)        #信号幅值（包络）
            self.lp_env = self.lp(env, self.lp_env, self.env_alpha)  #信号幅值加权

            peak_loc = np.argmax(self.lp_env)    #信号的最大值索引
            self.lp_peak_loc = self.lp(peak_loc, self.lp_peak_loc, self.peak_loc_alpha)  #信号的最大值索引加权

            peak_idx = int(round(self.lp_peak_loc))   #信号的最大值索引
            peak = np.mean(self.lp_sweep[peak_idx - 50 : peak_idx + 50])  #信号的最大值索引前后50值取平均获得峰值
            self.push(peak, self.peak_history)  #存储峰值

            delta = self.lp_sweep * np.conj(self.last_lp_sweep)  #信号和旧信号的共轭相乘

            phase_weights = np.imag(delta)
            if self.lp_phase_weights is None:
                self.lp_phase_weights = phase_weights
            else:
                self.lp_phase_weights = self.lp(
                    phase_weights,
                    self.lp_phase_weights,
                    self.phase_weights_alpha,
                )   #线性滤波

            weights = np.abs(self.lp_phase_weights) * env  #虚部分量的绝对值与幅值的乘积（放大权重）

            delta_dist = np.dot(weights, np.angle(delta))  #放大权重weights与乘积delta的辐角（相位）的点积
            delta_dist *= 2.5 / (2.0 * pi * sum(weights + 0.00001))
            #通过权重的方式放大了所需的波的相位，减小了杂波的相位，没有完全过滤

            y = self.movement_history[0] + delta_dist
            self.push(y, self.movement_history)

            y_breath, self.breath_zi = sosfilt(self.breath_sos, np.array([y]), zi=self.breath_zi)  #呼吸滤波
            self.push(y_breath, self.breath_history)

            #肺活量
            depthmax = np.max(self.breath_history)
            depthmin = np.min(self.breath_history)
            depthmaxidx = np.argmax(self.breath_history)
            depthminidx = np.argmin(self.breath_history)
            BreathingProcessor.k += 1
            if depthmaxidx != 0 and depthminidx != 0 and BreathingProcessor.k > 100 and depthmax > 7 and depthmin < -7:
                breathdepthmax = depthmax - depthmin
                global lungcapacity
                lungcapacity = round(80.596*breathdepthmax + 34.455*self.height + 20.64*self.weight-4856.98)
                # lungcapacity = breathdepthmax
                BreathingProcessor.lungcapacity_text = "肺活量为 {} ml".format(lungcapacity)

            #潮气容积
            breath_history2 = self.breath_history[: self.hist_plot_len // 3]
            breath_history2_max = np.max(breath_history2)
            maxs2 = self.find_breathpeaks(breath_history2)
            mins2 = self.find_breathpeaks(-breath_history2)
            max_idx2 = 0
            min_idx2 = 0
            breath_dist = np.array([])
            while True:
                if mins2.shape[0] < 2 or maxs2.shape[0] < 2:
                    break
                if not (min_idx2 < mins2.shape[0] and max_idx2 < maxs2.shape[0]):
                    break
                if breath_history2_max > 5:
                    break

                if maxs2[max_idx2, 0] < mins2[min_idx2, 0]:   #如果波峰的索引小于波谷的索引
                    inhale_dist = mins2[min_idx2, 1] + maxs2[max_idx2, 1]    #则判断为吸气，胸廓舒张，舒张深度为波峰与波谷的绝对值之和
                    breath_dist = np.append(breath_dist, inhale_dist)
                    max_idx2 += 1
                else:
                    exhale_dist = mins2[min_idx2, 1] + maxs2[max_idx2, 1]    #如果波峰的索引大于波谷的索引则判断为呼气，胸廓收缩，收缩深度为波峰与波谷的绝对值之和
                    breath_dist = np.append(breath_dist, exhale_dist)
                    min_idx2 += 1

            if breath_dist.size:
                lung_VT_dist = np.mean(breath_dist)
                BreathingProcessor.lung_VT = round(80.897*lung_VT_dist - 13.801*self.height + 18.808*self.weight + 1508.305)
                BreathingProcessor.lung_VT_text = "潮气容积为 {} ml".format(BreathingProcessor.lung_VT)

            #补呼气容积
            breath_history3 = self.breath_history[: self.hist_plot_len // 3]
            breath_history3_max = np.max(breath_history3)
            maxs3 = self.find_breathpeaks(breath_history3)
            mins3 = self.find_breathpeaks(-breath_history3)
            max_idx3 = 0
            min_idx3 = 0
            while True:
                if mins3.shape[0] < 2 or maxs3.shape[0] < 2:
                    break
                if not (min_idx3 < mins3.shape[0] and max_idx3 < maxs3.shape[0]):
                    break
                # if breath_history2_max < 3:
                #     break

                if maxs3[max_idx3, 1] == breath_history3_max and (max_idx3+1) < maxs3.shape[0]:
                    lung_ERV_dist = breath_history3_max - maxs3[max_idx3+1, 1]
                    # BreathingProcessor.lung_ERV = round(-34.335*lung_ERV_dist + 46.684*self.height - 24.464*self.weight - 4439.24)
                    BreathingProcessor.lung_ERV = lung_ERV_dist
                    BreathingProcessor.lung_ERV_text = "补呼气深度为 {} mm".format(BreathingProcessor.lung_ERV)
                    break

                if maxs3[max_idx3, 0] < mins3[min_idx3, 0]:   #如果波峰的索引小于波谷的索引
                    max_idx3 += 1
                else:
                    min_idx3 += 1

            # 补吸气容积
            if BreathingProcessor.lung_ERV != 0 and BreathingProcessor.lung_VT != 0 and lungcapacity != 0:
                BreathingProcessor.lung_IRV = lungcapacity - BreathingProcessor.lung_ERV - BreathingProcessor.lung_VT
                BreathingProcessor.lung_IRV_text = "补吸气容积为 {} ml".format(BreathingProcessor.lung_IRV)

            # 静息通气量
            breath_history4 = self.breath_history[: self.hist_plot_len // 3]
            breath_history4_max = np.max(breath_history4)
            maxs4 = self.find_peaks(breath_history4, 8)
            mins4 = self.find_peaks(-breath_history4, 8)
            max_idx4 = 0
            min_idx4 = 0
            inhale_time = None
            exhale_time = None
            inhale_dist = 0
            exhale_dist = 0
            first_peak = None
            breathbpm1 = np.array([])
            while not (inhale_time and exhale_time):
                if mins3.shape[0] < 2 or maxs3.shape[0] < 2:
                    break
                if not (min_idx4 < mins4.shape[0] and max_idx4 < maxs4.shape[0]):
                    break
                if breath_history4_max > 5:
                    break

                if maxs4[max_idx4, 0] < mins4[min_idx4, 0]:  # 如果波峰的索引小于波谷的索引
                    exhale_dist = mins4[min_idx4, 1] + maxs4[max_idx4, 1]  # 则判断为呼气，呼气深度为波峰与波谷的绝对值之和
                    if exhale_dist > 1 and exhale_dist < 10:  # 限制条件：呼气深度在1~20之间
                        exhale_time = mins4[min_idx4, 0] - maxs4[max_idx4, 0]  # 呼气时间为波谷的索引减去波峰的索引
                        if first_peak is None:
                            first_peak = maxs4[max_idx4, 0]  # 记录第一个峰值为波峰
                    max_idx4 += 1
                else:
                    inhale_dist = mins4[min_idx4, 1] + maxs4[max_idx4, 1]  # 如果波峰的索引大于波谷的索引则判断为吸气，吸气深度为波峰与波谷的绝对值之和
                    if inhale_dist > 1 and inhale_dist < 10:  # 限制条件：吸气深度在1~20之间，
                        inhale_time = maxs4[max_idx4, 0] - mins4[min_idx4, 0]  # 吸气时间为波峰的索引减去波谷的索引
                        if first_peak is None:
                            first_peak = mins4[min_idx4, 0]  # 记录第一个峰值为波谷
                    min_idx4 += 1

                if inhale_time is not None and exhale_time is not None:
                    bpm = 60.0 / ((inhale_time + exhale_time) / self.f)  # 计算bpm
                    symmetry = (inhale_dist - exhale_dist) / (inhale_dist + exhale_dist)  # 计算呼吸对称性
                    first_peak_rel = first_peak / (inhale_time + exhale_time)  # 第一次峰值应该出现于整个呼吸过程的前0.7中
                    if 6 < bpm < 30 and abs(symmetry) < 0.6 and first_peak_rel < 0.7:
                        breathbpm1 = np.append(breathbpm1, bpm)

            if breathbpm1.size:
                breathbpm2 = np.mean(breathbpm1)
                BreathingProcessor.lung_VE = breathbpm2 * BreathingProcessor.lung_VT / 1000
                BreathingProcessor.lung_VE_text = "静息通气量为 {:0.1f} L/min".format(BreathingProcessor.lung_VE)

            # Make an explicit copy, otherwise flip will not return a new object
            breath_hist_plot = self.breath_history[: self.hist_plot_len]
            breath_hist_plot = np.array(np.flip(breath_hist_plot, axis=0))
            breath_hist_plot -= (np.max(breath_hist_plot) + np.min(breath_hist_plot)) * 0.5

            out_data = {
                "peak_hist": self.peak_history[:100],  #每一帧的峰值
                "env_ampl": abs(self.lp_sweep),   #信号幅值（包络）
                "env_delta": self.lp_phase_weights,   #信号相位
                "peak_idx": peak_idx,    #信号的最大值索引
                "breathing_history": breath_hist_plot,
                "lungcapacity":lungcapacity,
                "lungcapacity_text": BreathingProcessor.lungcapacity_text,
                "lung_VT_text": BreathingProcessor.lung_VT_text,
                "lung_ERV_text": BreathingProcessor.lung_ERV_text,
                "lung_IRV_text": BreathingProcessor.lung_IRV_text,
                "lung_VE_text": BreathingProcessor.lung_VE_text,
            }

        self.last_lp_sweep = self.lp_sweep
        self.sweep_index += 1
        return out_data

    def lp(self, new, state, alpha):
        return alpha * state + (1 - alpha) * new

    def push(self, val, arr):
        res = np.empty_like(arr)
        res[0] = val
        res[1:] = arr[:-1]
        arr[...] = res

    def find_peaks(self, env, width):
        n = len(env)
        peaks = np.zeros((0, 2))
        for idx in range(0, n, width):
            mi = np.argmax(env[idx : min(idx + width, n)]) + idx   #获得idx后面width帧的最大值的索引
            mi2 = np.argmax(env[max(mi - width, 0) : min(mi + width, n)])  #获得idx前后各width帧的最大值的索引
            mi2 += max(mi - width, 0)
            if mi == mi2 and (0 < mi < n - 1):  #如果这两个索引值相同，则其是一个峰值
                peaks = np.concatenate((peaks, np.array([[mi, env[mi]]])), axis=0)
        return peaks

    def find_breathpeaks(self, env):
        n = len(env)
        peaks = np.zeros((0, 2))
        mi = []
        for idx in range(1, n-1):
            if env[idx - 1] < env[idx] and env[idx] > env[idx + 1]:
                mi.append(idx)
        for i in range(len(mi) - 1, 0, -1):
            if mi[i] - mi[i - 1] < 40 and mi[i] - mi[i - 2] < 200:   #设定采样率为20hz,测量呼吸频率<30的情况下取40
                mi[i - 1] = mi[i]
                del mi[i - 1]
        for i in range(0, len(mi)):
                peaks = np.concatenate((peaks, np.array([[mi[i], env[mi[i]]]])), axis=0)
        return peaks

class PGUpdater:
    def __init__(self, sensor_config, processing_config, session_info):
        assert sensor_config.update_rate is not None

        f = sensor_config.update_rate
        self.depths = et.utils.get_range_depths(sensor_config, session_info)
        self.hist_plot_len_s = processing_config.hist_plot_len
        self.hist_plot_len = int(round(self.hist_plot_len_s * f))
        self.move_xs = (np.arange(-self.hist_plot_len, 0) + 1) / f
        self.smooth_max = et.utils.SmoothMax(f, hysteresis=0.4, tau_decay=1.5)

    def setup(self, win):
        self.env_plot = win.addPlot(title="目标对象距离")
        self.env_plot.setMenuEnabled(False)
        self.env_plot.setMouseEnabled(x=False, y=False)   #鼠标交互，变化xy轴
        self.env_plot.hideButtons()

        self.env_plot.addLegend()
        self.env_plot.showGrid(x=True, y=True)
        self.env_curve = self.env_plot.plot(
            pen=et.utils.pg_pen_cycler(0),
            name="幅值变化",
        )
        self.delta_curve = self.env_plot.plot(
            pen=et.utils.pg_pen_cycler(1),
            name="相位变化",
        )
        self.peak_vline = pg.InfiniteLine(pen=pg.mkPen("k", width=2.5, style=QtCore.Qt.DashLine))
        self.env_plot.addItem(self.peak_vline)

        win.nextRow()

        self.move_plot = win.addPlot(title="呼吸波形")
        self.move_plot.setMenuEnabled(False)
        self.move_plot.setMouseEnabled(x=False, y=False)
        self.move_plot.hideButtons()
        self.move_plot.showGrid(x=True, y=True)
        self.move_plot.setLabel("bottom", "时间 (s)")
        self.move_plot.setLabel("left", "呼吸深度")
        self.move_plot.setYRange(-1, 1)
        self.move_plot.setXRange(-self.hist_plot_len_s, 0)

        self.move_curve = self.move_plot.plot(pen=et.utils.pg_pen_cycler(0))
        self.move_text_item1 = pg.TextItem(color=pg.mkColor("k"), anchor=(0, 1))
        self.move_text_item1.setPos(self.move_xs[0], -1)
        self.move_plot.addItem(self.move_text_item1)
        self.move_text_item2 = pg.TextItem(color=pg.mkColor("k"), anchor=(0, 1))
        self.move_text_item2.setPos(self.move_xs[0], -1)
        self.move_plot.addItem(self.move_text_item2)
        self.move_text_item3 = pg.TextItem(color=pg.mkColor("k"), anchor=(0, 1))
        self.move_text_item3.setPos(self.move_xs[0], -1)
        self.move_plot.addItem(self.move_text_item3)
        self.move_text_item4 = pg.TextItem(color=pg.mkColor("k"), anchor=(0, 1))
        self.move_text_item4.setPos(self.move_xs[0], -1)
        self.move_plot.addItem(self.move_text_item4)
        self.move_text_item5 = pg.TextItem(color=pg.mkColor("k"), anchor=(0, 1))
        self.move_text_item5.setPos(self.move_xs[0], -1)
        self.move_plot.addItem(self.move_text_item5)

    def update(self, data):
        envelope = data["env_ampl"]  # 信号幅值（包络）
        m = max(200,np.max(envelope)+100) # 信号幅值的平滑最大值？？
        plot_delta = data["env_delta"] * m * 2e-5 + 0.5 * m
        peak_x = self.depths[data["peak_idx"]]   #第一张图的峰值数据，黑色竖线

        self.env_plot.setYRange(0, m)
        self.env_curve.setData(self.depths, envelope)  # 第一张图蓝线
        self.delta_curve.setData(self.depths, plot_delta)  # 第一张图橙线
        self.peak_vline.setValue(peak_x)    #第一张图的黑色竖线

        m = max(1, max(np.abs(data["breathing_history"])))
        self.move_curve.setData(self.move_xs, data["breathing_history"])
        self.move_plot.setYRange(-m, m)
        self.move_text_item1.setPos(self.move_xs[0], -m)
        self.move_text_item1.setText(data["lungcapacity_text"])
        self.move_text_item2.setPos(self.move_xs[0], -m+1.5)
        self.move_text_item2.setText(data["lung_VT_text"])
        self.move_text_item3.setPos(self.move_xs[0], -m+4.5)
        self.move_text_item3.setText(data["lung_ERV_text"])
        self.move_text_item4.setPos(self.move_xs[0], -m+6)
        self.move_text_item4.setText(data["lung_IRV_text"])
        self.move_text_item5.setPos(self.move_xs[0], -m+3)
        self.move_text_item5.setText(data["lung_VE_text"])

if __name__ == "__main__":
    main()

