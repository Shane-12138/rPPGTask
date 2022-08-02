import torch
import numpy as np
from scipy.signal import savgol_filter
from tsmoothie.smoother import LowessSmoother
import heartpy as hp
from torchvision.transforms import Compose, _transforms_video, ToTensor
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject


class rPPGThread(QThread):
    rppg_signal = pyqtSignal(np.ndarray)
    hr_signal = pyqtSignal(float)
    freqs_signal = pyqtSignal(np.ndarray)
    rr_signal = pyqtSignal(float)

    def __init__(self, model, video, target_time):
        super(rPPGThread, self).__init__()
        self.model = model
        self.video = video
        self.transform = Compose([
            _transforms_video.ToTensorVideo(),
            _transforms_video.NormalizeVideo([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = 'cuda'
        self.target_time = target_time

    def calc_hr_from_freqs(self, bvp, bvp_sampling_rate):
        N = len(bvp)
        bvp = np.fft.fft(bvp)
        freqs = np.linspace(0, bvp_sampling_rate / 2, N // 2)
        bvp = np.abs(bvp[:N // 2])
        max_freqs = freqs[np.argmax(bvp)]
        return 60. * max_freqs

    def calculate_hr(self, bvp, bvp_sampling_rate):
        try:
            _, mesures = hp.process(bvp, sample_rate=bvp_sampling_rate, windowsize=4.)
            return mesures['bpm'], mesures['breathingrate']

        except Exception as e:
            print('Calculate HR from the frequency domain of the ground truth BVP signal.')
            return self.calc_hr_from_freqs(bvp, bvp_sampling_rate), 0

    def run(self):
        self.video = torch.FloatTensor(self.video).unsqueeze(0).to(self.device)
        self.model.train()
        with torch.no_grad():
            # rPPG_signal, HR_value = self.model(self.video)
            rPPG_signal = self.model(self.video)

            # 归一化
            rPPG_signal = (rPPG_signal - torch.mean(rPPG_signal)) / torch.std(rPPG_signal)
            rPPG_signal = rPPG_signal.detach().cpu().numpy().squeeze(0)

            # import random
            # save_name = "../output/bvp/" + str(random.random()).replace(".", "-") + ".txt"
            # np.savetxt(save_name, rPPG_signal, fmt='%f', delimiter=" ")

            # torch显存释放
            self.video = None
            torch.cuda.empty_cache()

            # 滤波
            # rPPG_signal = savgol_filter(rPPG_signal, 25, 7)

            # 滤波
            smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
            smoother.smooth(rPPG_signal)
            rPPG_signal = smoother.smooth_data[0]

            sample_rate = len(rPPG_signal) / self.target_time
            # rPPG_signal = hp.filter_signal(rPPG_signal, cutoff=[0.75, 2.5], sample_rate=sample_rate,
            #                                order=5, filtertype='bandpass')

            # 计算心率和频谱
            hr, rr = self.calculate_hr(rPPG_signal, sample_rate)
            N = len(rPPG_signal)
            bvp_fft = np.fft.fft(rPPG_signal)
            freqs = np.linspace(0, N / 8 / 2, N // 2)
            bvp_fft = np.abs(bvp_fft[:N // 2])

            if np.isnan(hr) or hr < 30 or hr > 200:
                hr = 75.0
            if np.isnan(rr) or rr > 2:
                rr = 2

            self.rppg_signal.emit(rPPG_signal)
            self.hr_signal.emit(hr)
            self.freqs_signal.emit(bvp_fft)
            self.rr_signal.emit(rr)