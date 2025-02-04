import serial
import serial.tools.list_ports as lp
import time

MODE = 'counter'
# MODE = 'interval'

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




file_path = f'data_{MODE}_{time.strftime("%Y_%m_%d-%H_%M_%S")}.txt'
file = open(file_path, 'w')
def log(data):
    file.write(data + '\n')
    file.flush()
    print('logged: ', data)



buffer = []
s = serial.Serial(arduino_port, 19200)

try:
    # send the mode to the arduino
    s.write(MODE.encode('utf-8'))

    # read back data
    while True:
        while s.in_waiting > 0:
            buffer.append(s.read().decode('utf-8'))
        while '\n' in buffer:
            idx = buffer.index('\n')

            packet = ''.join(buffer[:idx+1])
            buffer = buffer[idx+1:]
            log(packet)

except KeyboardInterrupt:
    file.close()
    s.close()
    print('Exiting')

