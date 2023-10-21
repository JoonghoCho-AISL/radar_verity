# Copyright (c) Acconeer AB, 2022
# All rights reserved

import acconeer.exptool as et
from acconeer.exptool import a121
import os


parser = a121.ExampleArgumentParser()
parser.add_argument("--output-file", required=True)
parser.add_argument("--num-frames", required=True, type=int)
args = parser.parse_args()
et.utils.config_logging(args)

# Here we create a H5Recorder.
# The H5Recorder is an object that saves frames directly to a H5-file.
name = args.output_file
folder = 'counting_demo_h5'
filename = os.path.join(folder, name)

h5_recorder = a121.H5Recorder(filename)

client = a121.Client(ip_address = '127.0.0.1')
sensor_config = a121.SensorConfig(
            subsweeps=[
                a121.SubsweepConfig(
                    start_point = 250,
                    step_length = 1,
                    num_points = 170,
                    profile = a121.Profile.PROFILE_1,
                    hwaas = 8,
                ),
            ],
            sweeps_per_frame = 10,
            frame_rate = 10,
        )
# client = a121.Client(**a121.get_client_args(args))
client.connect()

# Session setup, just like the other examples.
# config = a121.SessionConfig()
client.setup_session(sensor_config)

# Here we send the H5Recorder to the Client.
# After this call, the client is responsible for the H5Recorder.
# The Client will automatically close the file when
# "stop_session" is called.
client.start_session(recorder=h5_recorder)

for i in range(args.num_frames):
    # Client will send its Results to the H5Recorder.
    client.get_next()
    print(f"Result {i + 1}/{args.num_frames} was sampled")

client.stop_session()
client.disconnect()
