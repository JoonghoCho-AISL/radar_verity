import paho.mqtt.client as mqtt
from urllib.parse import urlparse

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)

def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))

def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))

def subscribing_once(ip, port, topic):
    
    received_msg = None  # To store the received message
    
    def on_message(client, userdata, msg):
        nonlocal received_msg  # Declare received_msg as nonlocal
        msg = msg.payload.decode("utf-8")
        msg = eval(msg)
        print(msg)
        try:
            # Try accessing the nested structure
            content = msg['pc']['m2m:sgn']['nev']['rep']['m2m:cin']['con']
            
            # If the above line doesn't raise an exception, then the structure exists
            received_msg = content  # Store the received message
            print("Stopping sub the loop and disconnecting.")
            client.loop_stop()  # Stop the loop
            client.disconnect()  # Disconnect the client
        except (TypeError, KeyError):
            received_msg = msg
            # If there's a TypeError or KeyError, then the structure doesn't exist
            pass

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(ip, port)
    client.subscribe('/oneM2M/req/+/' + topic + '/#', 1)
    client.loop_forever()  # Blocking function
    
    return received_msg  # Return the received message

def sub_iot_platform_cin_con(nu, port): #nu값이 들어감
    print(nu)
    ip = urlparse(nu).hostname
    path_segments = urlparse(nu).path.split('/')
    topic = next(segment for segment in path_segments if segment)
    print(topic)
    port = int(port)

    received_msg = subscribing_once(ip, port, topic)

    parsed_msg = received_msg


    return parsed_msg

if __name__ == "__main__":

    def sub_iot_platform_cin_con(nu, port): #nu값이 들어감
        print(nu)
        ip = urlparse(nu).hostname
        path_segments = urlparse(nu).path.split('/')
        topic = next(segment for segment in path_segments if segment)
        print(topic)
        port = int(port)

        received_msg = subscribing_once(ip, port, topic)

        parsed_msg = received_msg


        return parsed_msg

    ip = '203.250.148.120'
    NU = 'mqtt://203.250.148.120/sjhTarget?ct=json'
    PORT = 20516

    received_message = sub_iot_platform_cin_con(NU, PORT)
    # print("Received Message:", received_message)
