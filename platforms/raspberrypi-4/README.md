# Platform: RaspberryPi 4 Bookworm

| Item    | Version             |
| ------- | ------------------- |
| Hardare | RaspberryPi 4b      |
| Memory  | 8 GBytes            |
| OS      | Bookworm Lite 64bit |

## Setup

### Operating System

Install your Operating system as described in detail [here](https://www.raspberrypi.com/documentation/computers/getting-started.html#install-using-imager)

I use the 'RaspberryPi OS Bookworm Lite 64bit' version to keep the system as simple as possible.

Once installed and you have network access and can remotely log in with ssh, update the OS:

    $ sudo apt update
    $ sudp apt dist-upgrade
    $ sudo reboot

### Docker

This repository uses docker to build the environments and run the tools. I've taken this approach as it can
be fully automated and isolated from the host environment - it should work reliably no matter what you have 
installed on your host.

Follow the instructions [here](https://docs.docker.com/engine/install/debian/) to install docker.

I used the instructions from the section "Install using the apt repository".

Once installed, add your user to the docker group so you can interact with the system without being root or
running sudo all the time:

    $ sudo usermod -aG docker pi

Log out and back in and you'll be ready to go. Test by running the hello-world example:

    $ docker run hello-world

