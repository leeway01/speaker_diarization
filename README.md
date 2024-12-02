# speaker_diarization
[https://huggingface.co/](https://huggingface.co/pyannote/speaker-diarization)

venv\Scripts\activate

[기본 환경설정]\n
pip install spleeter moviepy pyannote.audio
pip install torch torchvision torchaudio
pip install pyannote.audio

[moviepy.editor 오류시 moviepy 버전 다운그레이드 및 호환]\n
pip install moviepy==1.0.3
pip install protobuf==3.20.3
pip install numpy==1.23.5

[화자분리 정확성을 높히기 위해 모델과 맞는 pyannote, torch 버전 조절]\n
pip install pyannote.audio==0.0.1
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
