import socket
import time
import threading

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
                self.sock.close()
                time.sleep(5)

    def receive_loop(self):
        try:
            while self.running:
                data = self.sock.recv(1024)
                if not data:
                    print("서버 연결이 끊어졌습니다.")
                    break
                # decode 에러가 나도 소켓을 닫지 않고 계속 동작
                try:
                    print("서버로부터 메시지:", data.decode('utf-8'))
                except UnicodeDecodeError:
                    print("서버로부터 메시지(바이너리):", data.hex())
                msg = input("보낼 메시지 입력: ")
                self.sock.sendall(msg.encode())
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

# 사용 예시
if __name__ == "__main__":
    client = TCPClient("127.0.0.1", 9004)  # 서버 IP와 포트에 맞게 수정
    try:
        client.run()
    except KeyboardInterrupt:
        print("클라이언트 종료합니다.")
        client.stop()