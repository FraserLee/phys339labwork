import serial
import serial.tools.list_ports as lp
import time

mode = 'transparent'
# mode = 'polarised'

ports = lp.comports()
arduino_port = None
for p in ports:
    print(p.device, p.description)
    if 'arduino' in p.description.lower():
        arduino_port = p.device
        print('\t⬆️  identified as arduino')

if arduino_port is None:
    print('No arduino found, exiting')
    exit()




file_path = f'data/{mode}_{time.strftime("%Y_%m_%d-%H_%M_%S")}.txt'
file = open(file_path, 'w')
def log(data):
    file.write(data.strip() + '\n')
    file.flush()
    print('logged [', data.strip(), ']')

buffer = ''
system_ready = False
command_issued = False

s = serial.Serial()
try:
    s.port = arduino_port
    s.baudrate = 115200
    s.open()

    s.write(bytes('100\n', 'utf-8'))

    # read back data
    while True:
        while s.in_waiting > 0:
            piece = s.read()
            try:
                buffer += piece.decode('utf-8')
            except:
                print('read malformed data:', piece)

        if system_ready and not command_issued:
            s.write(bytes(mode + '\n', 'utf-8'))

        while '\n' in buffer:
            idx = buffer.index('\n')
            packet = ''.join(buffer[:idx+1]).strip()
            buffer = buffer[idx+1:]

            if packet.startswith('PD Analog read: '):
                log(packet[len('PD Analog read: '):])
            else:
                print('received (but didn\'t log):', packet)
                if 'Input a number of steps' in packet:
                    system_ready = True

except KeyboardInterrupt:
    file.close()
    s.close()
    print('Exiting')

