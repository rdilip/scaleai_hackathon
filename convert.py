import os
from glob import glob

if __name__ == '__main__':
    for fname in glob("sample_files/*.mp3"):
        base = os.path.splitext(fname)[0]
        os.system(f"ffmpeg -i {base}.mp3 -ar 48000 -vn -c:a libvorbis {base}.ogg")