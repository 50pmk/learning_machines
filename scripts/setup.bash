#!/usr/bin/env bash
# replace localhost with the port you see on the smartphone
# export ROS_MASTER_URI="http://10.15.2.184:11311"
export ROS_MASTER_URI="http://10.15.3.239:11311"
# export ROS_MASTER_URI="http://localhost:11311"
# lab "http://10.15.2.184.62:11311"
# thuis "http://10.15.2.184:11311"

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# linux:
#    hostname -I | awk '{print $1}'
# macOS:
#    ipconfig getifaddr en1
export COPPELIA_SIM_IP="192.168.1.156"
# lab "10.15.3.108"
# thuis "192.168.1.156"