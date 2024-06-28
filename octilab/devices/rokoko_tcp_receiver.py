import socket

# Define the IP address and port to listen on
TCP_IP = "0.0.0.0"  # Listen on all available network interfaces
TCP_PORT = 14043     # Make sure this matches the port used in Rokoko Studio Live

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)

print(f"Listening for TCP connections on+ {TCP_IP}:{TCP_PORT}")

while True:
    conn, addr = sock.accept()
    print(f"Connection from: {addr}")
    while True:
        data = conn.recv(1024)  # Buffer size is 1024 bytes
        if not data:
            break
        print(f"Received message: {data}")
    conn.close()
