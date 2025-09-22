import torch
import cv2
import os
import numpy as np
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

def load_model(device):
    """
    Memuat model Stable Diffusion dari Hugging Face.
    """
    print("Memuat model Stable Diffusion...")
    model_id = "CompVis/stable-diffusion-v1-4"
    try:
        # Pipa diatur ke float16 untuk penggunaan memori yang efisien
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        return pipe.to(device)
    except OSError as e:
        print("\nKesalahan: Model tidak dapat dimuat. Pastikan Anda telah login ke Hugging Face.")
        print("Silakan jalankan 'huggingface-cli login' di terminal Anda.")
        raise e

def generate_frames(pipe, prompt, num_frames=10, device="cuda"):
    """
    Menghasilkan bingkai gambar dari prompt menggunakan Stable Diffusion.
    """
    print(f"Menghasilkan {num_frames} bingkai dari teks...")
    frames = []
    
    # Loop untuk membuat setiap bingkai
    for i in tqdm(range(num_frames)):
        # Menambahkan seed yang unik untuk setiap bingkai agar ada variasi
        generator = torch.Generator(device=device).manual_seed(i)
        
        image = pipe(
            prompt, 
            num_inference_steps=50, 
            guidance_scale=7.5, 
            generator=generator
        ).images[0]
        
        # Konversi gambar ke format yang dapat digunakan oleh OpenCV (BGR)
        image_np = np.array(image.convert("RGB"))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        frames.append(image_np)
    
    return frames

def create_video(frames, output_filename, fps=15):
    """
    Menggabungkan bingkai gambar menjadi satu file video.
    """
    if not frames:
        print("Tidak ada bingkai untuk dibuat video.")
        return

    print("Menggabungkan bingkai menjadi video...")
    height, width, _ = frames[0].shape
    
    # Inisialisasi VideoWriter
    # Codec 'mp4v' atau 'avc1' adalah yang paling umum
    video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame in tqdm(frames):
        video_writer.write(frame)
    
    video_writer.release()
    print(f"\nVideo berhasil dibuat! Tersimpan sebagai '{output_filename}'")

def main():
    """
    Fungsi utama untuk menjalankan program.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Perhatian: Anda menggunakan CPU. Proses akan sangat lambat.")

    try:
        pipe = load_model(device)
    except Exception:
        return

    text_prompt = input("Masukkan deskripsi untuk video Anda: ")
    if not text_prompt:
        print("Deskripsi tidak boleh kosong.")
        return
    
    num_frames = 15  # Jumlah bingkai yang dihasilkan
    fps = 5          # Bingkai per detik (Frames Per Second)

    frames = generate_frames(pipe, text_prompt, num_frames=num_frames, device=device)
    create_video(frames, "video_output.mp4", fps=fps)

if __name__ == "__main__":
    main()
      
