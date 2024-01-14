# Setup Azure Kinect

You need a laptop with USB (3.1 or type-C) port to collect data. 

## Compile k4a tools from source

We provide a bash script `scripts/azure_kinect.sh` to install everything. The following method is tested on Ubuntu 
18.04 and Ubuntu 20.04.

- `azure_kinect_activate` will just set the environment variables for usage etc.
- `azure_kinect_update` download and prepare the source code.
- `azure_kinect_build` everything will be built.

so to set everything up for the first time:

```bash
# dependency
sudo apt install libsoundio1 libsoundio-dev libudev-dev libusb-1.0-0-dev
```

```bash
source scripts/azure_kinect.sh
azure_kinect_activate
azure_kinect_update
azure_kinect_build
```

to setup environment variables before usage:

```bash
source scripts/azure_kinect.sh
azure_kinect_activate
```