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

import csv
import socket
import time
import threading
import numpy as np
import pandas as pd
import queue
import struct
import joblib
from dataclasses import dataclass


# 1. 모델 로드 (1회만)
model = joblib.load('lzr_ai_model.joblib')

def predict_distance_array(model, arr):
    # 반드시 float32로 변환 (모델 학습 dtype과 맞추는게 원칙)
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr.reshape(1, -1)
    pred = model.predict(arr)[0]
    return int(pred)


HEADER_SIZE = 8
AI_EXPECT_U16 = 1096
AI_EXPECT_BYTES = AI_EXPECT_U16 * 2

@dataclass
class Header:
    source: int        # uint8
    destination: int   # uint8
    msg_id: int        # uint16
    size: int          # int32 (signed)

def parse_header_le(buf: bytes) -> Header:
    if len(buf) != HEADER_SIZE:
        raise ValueError(f"헤더 길이 에러: {len(buf)}")
    mv = memoryview(buf)
    source = mv[0]
    destination = mv[1]
    msg_id = int.from_bytes(mv[2:4], 'little', signed=False)
    size   = int.from_bytes(mv[4:8], 'little', signed=True)
    return Header(source, destination, msg_id, size)

def pack_header_le(h: Header) -> bytes:
    return (
        bytes([h.source]) +
        bytes([h.destination]) +
        h.msg_id.to_bytes(2, 'little', signed=False) +
        h.size.to_bytes(4, 'little', signed=True)
    )

def read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b''
        buf.extend(chunk)
    return bytes(buf)

class TCPClient:
    def __init__(self, host, port):
        self.q = queue.Queue(maxsize=1)            # 항상 최신값만 저장 (그 외에는 버림)
        self.host = host
        self.port = port
        self.sock = None
        self.running = True
        self.csv_path = 'log.csv'

        # TCP 클라이언트 시작점
    def run(self):
        while self.running:
            self.connect()
            if not self.running:
                break
            t_recv  = threading.Thread(target=self.receive_thread, daemon=True)
            t_proc  = threading.Thread(target=self.process_thread, daemon=True)
            t_proc.start()
            t_recv.start()
            t_recv.join()
            try:
                self.q.put_nowait(None)
            except queue.Full:
                pass
            t_proc.join()
            self.close()
            if self.running:
                print("5초 후 재연결 시도...")
                time.sleep(5)

    # 서버 접속용 함수
    def connect(self):
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None

    # 수신을 위한 쓰레드 (헤더8 → 바디(size) → print → 큐에 body(bytes) 넣기)
    def receive_thread(self):
        try:
            while self.running and self.sock:
                header_bytes = read_exact(self.sock, HEADER_SIZE)
                if len(header_bytes) != HEADER_SIZE:
                    print("수신 중단: 헤더 수신 실패.")
                    break
                h = parse_header_le(header_bytes)
                print(f"[HEADER] source(1B)={h.source}  destination(1B)={h.destination}  "
                      f"id(2B)={h.msg_id}  size(4B)={h.size}")
                body = read_exact(self.sock, h.size)
                if len(body) != h.size:
                    print(f"수신 중단: 바디 부족({len(body)}/{h.size}).")
                    break
                
                if h.destination == 3 or h.destination == 5:
                    # print('수신자 해당함')
                    if h.msg_id == 50011:
                        # print('ID 50011 받음')
                        # ushort(=uint16) 1096개로 변환 (리틀엔디안 명시)
                        # arr_u16 = np.frombuffer(body, dtype=np.dtype('<u2'), count=AI_EXPECT_U16)
                        # memcpy 개념 그대로: 바디 바이트를 u16 배열로 '해석'
                        # print(f"[U16  ] count={arr_u16.size} first5={arr_u16[:5].tolist()} "f"last5={arr_u16[-5:].tolist() if arr_u16.size>=5 else arr_u16.tolist()}")

                        if self.q.full():
                            try:
                                self.q.get_nowait()
                            except queue.Empty:
                                pass
                        self.q.put(body)
                else:
                    print('수신 대상아님')
                    continue

        except Exception as e:
            print(f"수신 중 에러: {e}")
            # self.running = False
        

    # 처리부, 큐 바디→u16(1096)→예측→응답 헤더/바디 직접 구성해서 송신
    def process_thread(self):
        try:
            while self.running and self.sock:
                buf = self.q.get()  # 데이터가 들어올 때까지 블로킹
                if buf is None:
                    break  # 재연결 루프로 복귀

                arr_u16 = np.frombuffer(buf, dtype=np.dtype('<u2'), count=AI_EXPECT_U16)
                # 2) float32로 변환
                arr_f32 = arr_u16.astype(np.float32)
                result = predict_distance_array(model, arr_f32)
                if result == 0:
                    print('NO HUMAN')
                else:
                    print('HUMAN')

                # 응답 body 1바이트
                body = bytes([result & 0xFF])
                # 응답 헤더 수동 구성
                print('바디사이즈:',len(body))
                header = pack_header_le(Header(source=3, destination=1, msg_id=41002, size=len(body)))
                packet = header + body
                try:
                    self.sock.sendall(packet)
                    print("예측 결과 전송:", result)
                except Exception as e:
                    print(f"송신 중 에러: {e}")
                    break
        except Exception as e:
            print(f"처리 중 에러: {e}")

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

