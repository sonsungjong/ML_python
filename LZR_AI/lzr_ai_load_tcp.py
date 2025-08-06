# 실행하여 AI 모델을 로드한다
# TCP 클라이언트로
# TCP 서버에 연결한 후
# 서버로부터 메시지를 받으면
# AI 예측을 한 후
# 결과를 전송한다

# pip install scikit-learn
# pip install xgboost
# pip install lightgbm
# pip install catboost

# 랜덤포레스트 버전

import socket
import joblib
import pandas as pd
import numpy as np
import time
import threading

# 1. 모델 로드 (1회만)
model = joblib.load('lzr_ai_model.joblib')

def predict_distance_array(model, arr):
    # 반드시 float32로 변환 (모델 학습 dtype과 맞추는게 원칙)
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr.reshape(1, -1)
    pred = model.predict(arr)[0]
    return int(pred)

class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.running = True

    def connect(self):
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print("서버에 연결되었습니다.")
                return
            except Exception as e:
                print(f"서버 연결 실패: {e}")
                print("5초 후 재연결 시도...")
                try:
                    if self.sock:
                        self.sock.close()
                except:
                    pass
                time.sleep(5)

    def receive_loop(self):
        try:
            while self.running:
                # 3. 1096개 unsigned short (2byte) 수신 (총 2192 bytes)
                recv_size = 1096 * 2
                buf = b''
                while len(buf) < recv_size:
                    packet = self.sock.recv(recv_size - len(buf))
                    if not packet:
                        raise ConnectionError("서버 연결 끊김")
                    buf += packet

                # 4. uint16 → float32 변환 (모델 학습 타입과 맞추기)
                arr = np.frombuffer(buf, dtype=np.uint16)
                arr = arr.astype(np.float32)    # 모델이 float로 학습된 경우 반드시 필요

                # 5. 예측
                result = predict_distance_array(model, arr)

                # 6. 결과 서버로 전송 (1바이트, 0 또는 1)
                self.sock.sendall(bytes([result]))
                print("예측 결과 전송:", result)
        except Exception as e:
            print(f"수신 중 에러: {e}")
            # 여기서만 finally 진입
        finally:
            self.sock.close()

    def run(self):
        while self.running:
            self.connect()
            self.receive_loop()
            print("5초 후 재연결 시도...")
            time.sleep(5)

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

# main
if __name__ == "__main__":
    SERVER_IP = '127.0.0.1'   # 192.168.xxx.xxx
    SERVER_PORT = 9004
    client = TCPClient(SERVER_IP, SERVER_PORT)          # 서버 IP와 포트에 맞게 수정
    try:
        client.run()
    except KeyboardInterrupt:
        print("클라이언트 종료합니다.")
        client.stop()

