import os
from moviepy.editor import AudioFileClip
from spleeter.separator import Separator
from pyannote.audio import Pipeline
import librosa
import soundfile as sf
from whisper import load_model
import requests
from elevenlabs import ElevenLabs

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

# Step 3: pyannote.audio로 화자 분리
def diarize_speakers(audio_path, diarization_output_path, hf_token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    diarization = pipeline(audio_path)

    speaker_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})

    with open(diarization_output_path, "w") as diarization_file:
        for seg in speaker_segments:
            diarization_file.write(f"{seg['speaker']}: {seg['start']:.2f} --> {seg['end']:.2f}\n")

    print(f"화자 분리 완료: {diarization_output_path}")
    return speaker_segments

# Step 4: STT로 화자별 텍스트 변환
def transcribe_speakers(audio_path, speaker_segments, stt_model):
    model = load_model(stt_model)
    speaker_texts = {seg["speaker"]: [] for seg in speaker_segments}

    for seg in speaker_segments:
        segment_audio, _ = librosa.load(audio_path, sr=16000, offset=seg["start"], duration=seg["end"] - seg["start"])
        sf.write("temp_segment.wav", segment_audio, 16000)

        result = model.transcribe("temp_segment.wav")
        speaker_texts[seg["speaker"]].append({
            "start": seg["start"],
            "end": seg["end"],
            "text": result["text"]
        })

    os.remove("temp_segment.wav")

    return speaker_texts

# Step 5: 텍스트 파일 저장
def save_transcriptions(transcriptions, output_path):
    with open(output_path, "w") as f:
        for speaker, texts in transcriptions.items():
            f.write(f"{speaker}:\n")
            for entry in texts:
                f.write(f"  [{entry['start']:.2f} - {entry['end']:.2f}] {entry['text']}\n")
            f.write("\n")
    print(f"화자별 텍스트 저장 완료: {output_path}")

# Step 6: TTS로 텍스트를 음성으로 변환
def generate_tts_from_transcriptions(transcriptions, output_dir, tts_api_key):
    # ElevenLabs 클라이언트 초기화
    client = ElevenLabs(api_key=tts_api_key)

    for speaker, texts in transcriptions.items():
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        for i, entry in enumerate(texts):
            voice_id = "4XgajbaeFaof5lQX7hEo"  # ElevenLabs의 Voice ID
            text = entry["text"]

            # TTS 요청
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                text=text,
            )

            # 음성 파일 저장
            audio_file_path = os.path.join(speaker_dir, f"{speaker}_{i}.mp3")
            with open(audio_file_path, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)
            print(f"{speaker}의 음성 파일 저장 완료: {audio_file_path}")

# Main: 통합 실행
def process_video(video_path, output_dir, hf_token, tts_api_key, stt_model="large"):
    audio_output_path = os.path.join(output_dir, "extracted_audio.wav")
    spleeter_output_dir = os.path.join(output_dir, "spleeter_output")
    diarization_output_path = os.path.join(output_dir, "speaker_diarization.txt")
    transcriptions_output_path = os.path.join(output_dir, "speaker_transcriptions.txt")

    if not os.path.exists(spleeter_output_dir):
        os.makedirs(spleeter_output_dir)

    print("1. 영상에서 오디오 추출 중...")
    extract_audio_from_video(video_path, audio_output_path)

    print("2. 오디오에서 음원 분리 중...")
    separate_audio_with_spleeter(audio_output_path, spleeter_output_dir)

    print("3. 화자 분리 중...")
    vocals_path = os.path.join(spleeter_output_dir, "extracted_audio/vocals.wav")
    speaker_segments = diarize_speakers(vocals_path, diarization_output_path, hf_token)

    print("4. 화자별 텍스트 변환 중...")
    transcriptions = transcribe_speakers(vocals_path, speaker_segments, stt_model)

    print("5. 화자별 텍스트 파일 저장 중...")
    save_transcriptions(transcriptions, transcriptions_output_path)

    print("6. 화자별 음성을 음성 파일로 변환 중...")
    generate_tts_from_transcriptions(transcriptions, output_dir, tts_api_key)

    print(f"처리 완료! 결과물은 {output_dir}에서 확인하세요.")

# 실행
if __name__ == "__main__":
    video_file = "videoplayback_interview.mp4"  # 입력 영상 파일 경로
    output_directory = "output_interview_kimsuhyeon"  # 출력 디렉터리

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    hf_token = input("Hugging Face 토큰을 입력하세요: ").strip()
    tts_api_key = input("ElevenLabs API 키를 입력하세요: ").strip()
    process_video(video_file, output_directory, hf_token, tts_api_key)
