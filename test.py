import openai

def test_openai_api(api_key):
    openai.api_key = api_key
    try:
        # ChatCompletion 요청
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 접근 가능한 모델로 수정
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )
        print("응답:", response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    api_key = input("OpenAI API 키를 입력하세요: ").strip()
    test_openai_api(api_key)
