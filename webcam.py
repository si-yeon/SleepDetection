import tensorflow as tf
import cv2


model = tf.keras.models.load_model('2018_12_17_22_58_35.h5')  # 모델 파일 경로를 지정하세요.
cap = cv2.VideoCapture(1)  # 0은 기본 웹캠을 나타냅니다. 다른 웹캠을 사용하려면 숫자를 변경하세요.

while True:
    ret, frame = cap.read()

    # 프레임 전처리 (크기 조정, 정규화 등)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(26, 34, 1)),  # 입력 형태를 수정
    ])

    # 모델에 입력을 전달하여 출력 얻기
    model_input = frame  # 전처리된 프레임을 모델 입력으로 사용
    model_output = model.predict(tf.expand_dims(model_input, axis=0))

    # 출력을 화면에 표시하거나 다른 후처리 작업 수행
    # 예: 결과를 화면에 표시하거나 필요한 동작 수행

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


