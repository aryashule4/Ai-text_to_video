jalankan Di Termux 




pkg update && pkg upgrade -y
pkg install python git ffmpeg -y



git clone https://github.com/aryashule4/Ai-text_to_video.git
cd Ai-text_to_video



pip install virtualenv
virtualenv venv
source venv/bin/activate


pip install -r requirements.txt



echo "Halo dunia! Ini video teks ke video dari Termux." > input.txt



python t2v.py --input input.txt --out hasil.mp4 --lang id


termux-setup-storage




cp hasil.mp4 /sdcard/Download/

