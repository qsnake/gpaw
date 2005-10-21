def send(sckt, msg):
    length = len(msg)
    msg = ('%10d' % length) + msg
    length += 10
    totalsent = 0
    while totalsent < length:
        sent = sckt.send(msg[totalsent:])
        if sent == 0:
            raise RuntimeError, 'Socket connection broken!'
        totalsent += sent
    
def recv(sckt, length=None):
    if length is None:
        length = int(recv(sckt, 10))
    received = 0
    chunks = []
    while received < length:
        chunk = sckt.recv(length - received)
        if chunk == '':
            raise RuntimeError, 'Socket connection broken!'
        received += len(chunk)
        chunks.append(chunk)
    return ''.join(chunks)
