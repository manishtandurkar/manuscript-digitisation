import socket

def check_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(('127.0.0.1', port))
        print(f"Port {port} is OPEN (a server is running on it)")
        s.close()
        return True
    except socket.error:
        print(f"Port {port} is CLOSED (no server running)")
        return False

check_port(8000)
check_port(5173)
