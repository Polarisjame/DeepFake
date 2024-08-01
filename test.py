from pydub import AudioSegment
import time
import moviepy.editor as mp

file_path = r'/data/lingfeng/full_data/phase1/trainset/afe82d134e2cdca2469325143a3806da.mp4'
a = time.time()
track = AudioSegment.from_file(file_path, "mp4")
file_handle = track.set_frame_rate(16000).export('./test.wav', format="wav")
b = time.time()
print(file_handle, b-a)

# a = time.time()
# video = mp.VideoFileClip(file_path)
# video.audio.write_audiofile('./test.wav', verbose=False, logger=None, fps=16000)
# b = time.time()
# print(b-a)