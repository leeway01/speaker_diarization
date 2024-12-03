# speaker_diarization
[https://huggingface.co/](https://huggingface.co/pyannote/speaker-diarization)<br/>

python version : 3.10.11<br/>

venv\Scripts\activate

[기본 환경설정]<br/>
pip install spleeter moviepy pyannote.audio<br/>
pip install torch torchvision torchaudio<br/>
pip install pyannote.audio<br/>
pip install pydub<br/>
pip install openai-whisper<br/>
pip install elevenlabs<br/>

[moviepy.editor 오류시 moviepy 버전 다운그레이드 및 호환]<br/>
pip install moviepy==1.0.3<br/>
pip install protobuf==3.20.3<br/>
pip install numpy==1.23.5<br/>

[FFMPEG 설치 필요] 주소 : https://www.gyan.dev/ffmpeg/builds/<br/>
ffmpeg-git-full.7z 다운 후 환경변수 Path에 추가<br/>

[관리자 권한 얻기] <br/>
1. Win+R 검색창
2. secpol.msc
3. 로컬정책 -> 보안옵션 -> 관리 승인 모드에서 모든 관리자 실행(속성) -> 로컬 보안설정(사용안함)
4. 재부팅

[화자분리 정확성을 높히기 위해 모델과 맞는 pyannote, torch 버전 조절]<br/>
pip install pyannote.audio==0.0.1<br/>
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html<br/>
