import os
from moviepy.editor import AudioFileClip, VideoFileClip, CompositeAudioClip, concatenate_videoclips
from spleeter.separator import Separator
from pyannote.audio import Pipeline
import librosa
import soundfile as sf
from whisper import load_model
import requests
from elevenlabs import ElevenLabs
import openai
from pydub import AudioSegment
from pydub.effects import speedup

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

# Step 4: STT로 화자별 텍스트 변환 및 한글 텍스트 저장
def transcribe_and_save_speakers(audio_path, speaker_segments, stt_model, output_path):
    model = load_model(stt_model)
    speaker_texts = {seg["speaker"]: [] for seg in speaker_segments}

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in speaker_segments:
            segment_audio, _ = librosa.load(audio_path, sr=16000, offset=seg["start"], duration=seg["end"] - seg["start"])
            sf.write("temp_segment.wav", segment_audio, 16000)

            result = model.transcribe("temp_segment.wav")
            speaker_texts[seg["speaker"]].append({
                "start": seg["start"],
                "end": seg["end"],
                "text": result["text"]
            })

            # 한글 텍스트 저장
            f.write(f"{seg['speaker']}:\n")
            f.write(f"  [{seg['start']:.2f} - {seg['end']:.2f}] {result['text']}\n\n")

        os.remove("temp_segment.wav")

    print(f"화자별 한글 텍스트 저장 완료: {output_path}")
    return speaker_texts

# Step 5: 텍스트 번역 (GPT-4 사용)
def translate_transcriptions_gpt(transcriptions, openai_api_key):
    openai.api_key = openai_api_key
    translated_texts = {}

    for speaker, texts in transcriptions.items():
        translated_texts[speaker] = []
        for entry in texts:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Translate text from Korean to English."},
                        {"role": "user", "content": entry["text"]},
                    ]
                )
                translated_text = response['choices'][0]['message']['content'].strip()

                translated_texts[speaker].append({
                    "start": entry["start"],
                    "end": entry["end"],
                    "original_text": entry["text"],
                    "translated_text": translated_text
                })
                print(f"번역 완료: {entry['text']} -> {translated_text}")

            except Exception as e:
                print(f"번역 실패: {e}")
                translated_texts[speaker].append({
                    "start": entry["start"],
                    "end": entry["end"],
                    "original_text": entry["text"],
                    "translated_text": f"번역 실패: {entry['text']}"
                })

    return translated_texts

# Step 6: 번역된 텍스트 저장
def save_translations(translations, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for speaker, texts in translations.items():
            f.write(f"{speaker}:\n")
            for entry in texts:
                f.write(f"  [{entry['start']:.2f} - {entry['end']:.2f}]\n")
                f.write(f"    한글: {entry['original_text']}\n")
                f.write(f"    영어: {entry['translated_text']}\n")
            f.write("\n")
    print(f"번역된 텍스트 저장 완료: {output_path}")

# 화자별 음성 선택
def select_voice_id_for_speaker(speaker):
    """
    화자별 voice_id를 사용자 입력으로 선택합니다.
    """
    voice_options = {
        1: "4XgajbaeFaof5lQX7hEo",  # 빵형
        2: "X1Jmp1QQbOVhd7xXXQRL",  # 코딩애플
        3: "mtkRlP2CvaLpJmIUrDk5",  # 김성회
    }

    print(f"화자 {speaker}에 대해 사용할 voice_id를 선택하세요:")
    print("1: 빵형")
    print("2: 코딩애플")
    print("3: 김성회")

    while True:
        try:
            choice = int(input(f"화자 {speaker}의 voice_id 선택 (1~3): "))
            if choice in voice_options:
                return voice_options[choice]
            else:
                print("잘못된 입력입니다. 1~3 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")

# Step 7: TTS로 텍스트를 음성으로 변환 (품질 개선 및 타임 스트레칭 포함)
def generate_tts_from_translations(translations, tts_output_dir, tts_api_key):
    client = ElevenLabs(api_key=tts_api_key)
    speaker_voice_map = {}

    for speaker, texts in translations.items():
        # 화자별 voice_id 선택
        if speaker not in speaker_voice_map:
            speaker_voice_map[speaker] = select_voice_id_for_speaker(speaker)

        speaker_dir = os.path.join(tts_output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        for entry in texts:
            output_path = os.path.join(speaker_dir, f"{speaker}_{entry['start']:.2f}_{entry['end']:.2f}.mp3")
            
            # 이미 파일이 존재하면 건너뜀
            if os.path.exists(output_path):
                print(f"{speaker}의 음성 파일이 이미 존재합니다: {output_path}")
                continue

            voice_id = speaker_voice_map[speaker]
            text = entry["translated_text"]
            duration = entry["end"] - entry["start"]

            # 텍스트가 없거나 재생 시간이 너무 짧은 경우 스킵
            if not text.strip() or duration <= 0:
                print(f"Skipping TTS for {speaker}: Invalid text or duration.")
                continue

            try:
                # TTS 요청
                print(f"Generating TTS for {speaker} [{entry['start']:.2f}-{entry['end']:.2f}] with voice_id {voice_id}: {text}")
                audio_generator = client.text_to_speech.convert(
                    voice_id=voice_id,
                    model_id="eleven_multilingual_v2",
                    text=text,
                )

                # TTS 음성을 고해상도 출력으로 저장
                with open(output_path, "wb") as f:
                    for chunk in audio_generator:
                        f.write(chunk)

                print(f"{speaker}의 음성 파일 저장 완료: {output_path}")

            except Exception as e:
                print(f"Failed to generate TTS for {speaker}: {e}")


# Step 8: 음성과 배경음 합쳐 최종 영상 생성
def combine_audio_with_background(video_path, tts_output_dir, translations, spleeter_output_dir, output_video_path):
    # Step 2에서 분리된 배경음 로드
    background_audio_path = os.path.join(spleeter_output_dir, "extracted_audio/accompaniment.wav")
    background_audio = AudioSegment.from_file(background_audio_path)

    # TTS 음성을 배경음에 맞게 오버레이
    for speaker, texts in translations.items():
        for entry in texts:
            tts_audio_path = os.path.join(tts_output_dir, speaker, f"{speaker}_{entry['start']:.2f}_{entry['end']:.2f}.mp3")
            tts_audio = AudioSegment.from_file(tts_audio_path)

            start_time_ms = int(entry["start"] * 1000)  # 시작 시간 (ms)
            background_audio = background_audio.overlay(tts_audio, position=start_time_ms)

    # 합성된 오디오 저장
    combined_audio_path = os.path.join(os.path.dirname(output_video_path), "combined_audio.wav")
    background_audio.export(combined_audio_path, format="wav")

    # 최종 비디오 생성 (배경음 + TTS 음성 포함)
    video_clip = VideoFileClip(video_path)
    final_audio_clip = AudioFileClip(combined_audio_path)
    final_video_clip = video_clip.set_audio(final_audio_clip)
    final_video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    print(f"최종 영상 저장 완료: {output_video_path}")

# Main: 통합 실행
def process_video(video_path, output_dir, hf_token, tts_api_key, openai_api_key, stt_model="large"):
    audio_output_path = os.path.join(output_dir, "extracted_audio.wav")
    spleeter_output_dir = os.path.join(output_dir, "spleeter_output")
    diarization_output_path = os.path.join(output_dir, "speaker_diarization.txt")
    transcriptions_output_path = os.path.join(output_dir, "speaker_transcriptions.txt")
    translated_transcriptions_output_path = os.path.join(output_dir, "translated_speaker_transcriptions.txt")
    tts_output_dir = os.path.join(output_dir, "tts_audio")
    final_video_path = os.path.join(output_dir, "final_video.mp4")

    if not os.path.exists(spleeter_output_dir):
        os.makedirs(spleeter_output_dir)

    print("1. 영상에서 오디오 추출 중...")
    extract_audio_from_video(video_path, audio_output_path)

    print("2. 오디오에서 음원 분리 중...")
    separate_audio_with_spleeter(audio_output_path, spleeter_output_dir)

    print("3. 화자 분리 중...")
    vocals_path = os.path.join(spleeter_output_dir, "extracted_audio/vocals.wav")
    speaker_segments = diarize_speakers(vocals_path, diarization_output_path, hf_token)

    print("4. 화자별 한글 텍스트 변환 및 저장 중...")
    transcriptions = transcribe_and_save_speakers(vocals_path, speaker_segments, stt_model, transcriptions_output_path)

    print("5. 화자별 텍스트 번역 중...")
    translated_transcriptions = translate_transcriptions_gpt(transcriptions, openai_api_key)

    print("6. 번역된 텍스트 파일 저장 중...")
    save_translations(translated_transcriptions, translated_transcriptions_output_path)

    print("7. TTS를 이용해 번역된 텍스트를 음성으로 변환 중...")
    generate_tts_from_translations(translated_transcriptions, tts_output_dir, tts_api_key)

    print("8. 음성과 배경음을 합쳐 최종 영상을 생성 중...")
    combine_audio_with_background(video_path, tts_output_dir, translated_transcriptions, spleeter_output_dir, final_video_path)

    print(f"처리 완료! 결과물은 {output_dir}에서 확인하세요.")

# 실행
if __name__ == "__main__":
    video_file = "videoplayback.mp4"  # 입력 영상 파일 경로
    output_directory = "output"  # 출력 디렉터리

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 사용자로부터 토큰 입력받기
    hf_token = input("Hugging Face 토큰을 입력하세요: ").strip()
    tts_api_key = input("ElevenLabs API 키를 입력하세요: ").strip()
    openai_api_key = input("OpenAI API 키를 입력하세요: ").strip()

    # 함수 실행
    process_video(video_file, output_directory, hf_token, tts_api_key, openai_api_key)
