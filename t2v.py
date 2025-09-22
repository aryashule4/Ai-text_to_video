"""
t2v.py
Simple Text -> Video tool
Usage: python t2v.py --input "file.txt" --out output.mp4
"""
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
import textwrap

FONT_PATH = None  # gunakan default PIL jika None

def split_text_to_paragraphs(text, max_chars=300):
    # split by double newline first
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    out = []
    for p in paras:
        if len(p) <= max_chars:
            out.append(p)
        else:
            # wrap to chunks
            words = p.split()
            chunk = ''
            for w in words:
                if len(chunk) + len(w) + 1 > max_chars:
                    out.append(chunk.strip())
                    chunk = w
                else:
                    chunk += ' ' + w
            if chunk.strip():
                out.append(chunk.strip())
    return out


def render_slide(text, size=(1280,720), bg_color=(18,18,18), text_color=(255,255,255), font_size=44):
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    # load font
    try:
        font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    # wrap text
    margin = 80
    max_width = size[0] - 2*margin
    lines = []
    for line in text.split('\n'):
        lines += textwrap.wrap(line, width=40)
    y = (size[1] - len(lines)* (font_size+10)) // 2
    for l in lines:
        w,h = draw.textsize(l, font=font)
        x = (size[0]-w)//2
        draw.text((x,y), l, font=font, fill=text_color)
        y += font_size + 12
    return img


def tts_save(text, out_path, lang='id'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(out_path)


def ensure_audio_mono16(path):
    # moviepy prefers standard formats; ensure WAV mono 16-bit
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
    audio.export(path, format='wav')


def build_video(paragraphs, audio_path, out_path, slide_duration=None, fps=24, size=(1280,720)):
    # if slide_duration not provided, compute from audio length / slides
    audio = AudioFileClip(audio_path)
    total_audio = audio.duration
    n = len(paragraphs)
    if slide_duration is None:
        slide_duration = max(2, total_audio / n)
    clips = []
    for p in paragraphs:
        img = render_slide(p, size=size)
        tmp_img = 'tmp_slide.png'
        img.save(tmp_img)
        clip = ImageClip(tmp_img).set_duration(slide_duration)
        clips.append(clip)
    video = concatenate_videoclips(clips)
    video = video.set_audio(audio)
    video.write_videofile(out_path, fps=fps, codec='libx264', audio_codec='aac')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to input text file')
    parser.add_argument('--out', '-o', default='output.mp4', help='Output video file')
    parser.add_argument('--lang', default='id', help='TTS language code (default: id)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit('Input file not found')
    text = input_path.read_text(encoding='utf-8')
    paragraphs = split_text_to_paragraphs(text, max_chars=350)

    os.makedirs('build', exist_ok=True)
    audio_tmp = 'build/audio.wav'

    # create TTS
    print('Menyintesis suara...')
    tts_save('\n\n'.join(paragraphs), audio_tmp, lang=args.lang)
    ensure_audio_mono16(audio_tmp)

    print('Membangun video...')
    build_video(paragraphs, audio_tmp, args.out)
    print('Selesai ->', args.out)

if __name__ == '__main__':
    main()
