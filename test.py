from pydub import AudioSegment
import time
import moviepy.editor as mp
import torchvision
import librosa
import GPUtil


# file_path = r'/data/lingfeng/full_data/phase1/trainset/afe82d134e2cdca2469325143a3806da.mp4'

# a = time.time()
# track = AudioSegment.from_file(file_path, "mp4")
# file_handle = track.set_frame_rate(16000).export('./test.wav', format="wav")
# y, sr = librosa.load('./test.wav', sr=16000)
# b = time.time()
# print(b-a)

# a = time.time()
# video, audio, info = torchvision.io.read_video(file_path, pts_unit="sec")
# b = time.time()
# print(b-a)

GPUtil.showUtilization(True)