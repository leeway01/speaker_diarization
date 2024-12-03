import requests
from elevenlabs import ElevenLabs

# 내가 사용 가능한 voice_id가 뭐가있는지 확인하기 위한 코드
url = "https://api.elevenlabs.io/v1/voices"
headers = {
    "xi-api-key": "sk_aed2e28e0bbd90061c4264b2890eec1c37435b4f43125533"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print(f"오류 발생: {response.status_code} - {response.text}")

# ElevenLabs 클라이언트 초기화
client = ElevenLabs(
    api_key="sk_aed2e28e0bbd90061c4264b2890eec1c37435b4f43125533",
)

# TTS 요청
audio_generator = client.text_to_speech.convert(
    voice_id="4XgajbaeFaof5lQX7hEo",  # Voice ID ( 빵형 모델 아이디)
    model_id="eleven_multilingual_v2",  # 다국어 모델 (학습 모델이 여러개가 있음 그중 eleven_multilingual_v2를 많이 쓴다.)
    text="정상적으로 실행 되었다면 빵형 목소리로 mp3 가 생성이 된다", #(목소리로 바꿀 텍스트)
)

# 생성된 음성을 파일로 저장
output_file = "output.mp3"   #output.mp3 파일로 생성함
with open(output_file, "wb") as f:
    for chunk in audio_generator:  # 제너레이터 순회
        f.write(chunk)

print(f"음성 파일이 {output_file}로 저장되었습니다.")