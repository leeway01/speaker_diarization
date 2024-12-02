import os

# 심볼릭 링크 대신 파일 복사 사용 설정
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SB_DISABLE_SYMLINKS"] = "1"

from moviepy.editor import AudioFileClip
from spleeter.separator import Separator
from pyannote.audio import Pipeline
import librosa
import soundfile as sf

# Step 1: 영상에서 오디오 추출
def extract_audio_from_video(video_path, audio_output_path):
    audio = AudioFileClip(video_path)
    audio.write_audiofile(audio_output_path)
    print(f"오디오 추출 완료: {audio_output_path}")

# Step 2: Spleeter로 배경음악과 목소리 분리
def separate_audio_with_spleeter(audio_path, output_dir):
    separator = Separator("spleeter:2stems")  # 2stems: 음성 + 배경음악
    separator.separate_to_file(audio_path, output_dir)
    print(f"음원 분리 완료: {output_dir}")

# Step 3: pyannote.audio로 화자 분리 및 화자별 음성 파일 생성
def diarize_speakers(audio_path, diarization_output_path, speaker_audio_dir, hf_token):
    # Hugging Face 토큰으로 Pipeline 초기화
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    
    diarization = pipeline(audio_path)
    
    # 오디오 로드
    audio, sr = librosa.load(audio_path, sr=None)

    # 화자별 발화 구간 저장 및 음성 파일 생성
    with open(diarization_output_path, "w") as diarization_file:
        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            diarization_file.write(f"Speaker {speaker}: {turn.start:.2f} --> {turn.end:.2f}\n")
            
            # 화자별 음성 저장
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            speaker_audio = audio[start_sample:end_sample]
            
            speaker_file_path = os.path.join(speaker_audio_dir, f"speaker_{speaker}_part_{i}.wav")
            sf.write(speaker_file_path, speaker_audio, sr)
            print(f"화자 {speaker} 음성 파일 저장 완료: {speaker_file_path}")
    
    print(f"화자 분리 및 음성 파일 생성 완료: {diarization_output_path}")

# Step 4: 통합 실행
def process_video(video_path, output_dir, hf_token):
    # 경로 설정
    audio_output_path = os.path.join(output_dir, "extracted_audio.wav")
    spleeter_output_dir = os.path.join(output_dir, "spleeter_output")
    diarization_output_path = os.path.join(output_dir, "speaker_diarization.txt")
    speaker_audio_dir = os.path.join(output_dir, "speakers_audio")
    
    if not os.path.exists(speaker_audio_dir):
        os.makedirs(speaker_audio_dir)
    
    # 작업 실행
    print("1. 영상에서 오디오 추출 중...")
    extract_audio_from_video(video_path, audio_output_path)
    
    print("2. 오디오에서 음원 분리 중...")
    separate_audio_with_spleeter(audio_output_path, spleeter_output_dir)
    
    print("3. 화자 분리 및 음성 파일 생성 작업 중...")
    vocals_path = os.path.join(spleeter_output_dir, "extracted_audio/vocals.wav")
    diarize_speakers(vocals_path, diarization_output_path, speaker_audio_dir, hf_token)

    print(f"처리 완료! 결과물은 {output_dir}에서 확인하세요.")

# 실행
if __name__ == "__main__":
    video_file = "videoplayback2.mp4"  # 입력 영상 파일 경로
    output_directory = "output"      # 출력 디렉터리

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Hugging Face 토큰 입력
    hf_token = input("Hugging Face 토큰을 입력하세요: ").strip()
    
    process_video(video_file, output_directory, hf_token)
