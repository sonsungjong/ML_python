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
import time
import threading
import numpy as np
import pandas as pd
import queue
import joblib

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
        self.q = queue.Queue(maxsize=1)            # 항상 최신값만 저장 (그 외에는 버림)
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

    # 수신을 위한 쓰레드
    def receive_thread(self):
        try:
            while self.running:
                recv_size = 1096 * 2
                buf = b''
                while len(buf) < recv_size:
                    packet = self.sock.recv(recv_size - len(buf))
                    if not packet:
                        raise ConnectionError("서버 연결 끊김")
                    buf += packet
                # 큐가 꽉 차 있으면 오래된 데이터 버리기
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(buf)
        except Exception as e:
            print(f"수신 중 에러: {e}")
            self.running = False
        finally:
            if self.sock:
                self.sock.close()

    # 처리부, 큐에서 꺼내서 처리한다
    def process_thread(self):
        try:
            while self.running:
                buf = self.q.get()  # 데이터가 들어올 때까지 블로킹
                arr = np.frombuffer(buf, dtype=np.uint16)
                arr = arr.astype(np.float32)
                result = predict_distance_array(model, arr)
                try:
                    self.sock.sendall(bytes([result]))
                    print("예측 결과 전송:", result)
                except Exception as e:
                    print(f"송신 중 에러: {e}")
                    self.running = False
        except Exception as e:
            print(f"처리 중 에러: {e}")
        
    def run(self):
        while self.running:
            self.connect()
            self.running = True
            t1 = threading.Thread(target=self.receive_thread, daemon=True)
            t2 = threading.Thread(target=self.process_thread, daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if self.sock:
                self.sock.close()
            if self.running:
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

