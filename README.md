## Finding cell frequencies with computer vision

### Overview
My biologist friend would like to be able to take an image like what's below, with an array water droplets containing varying numbers of cells, and automatically determine the frequency distribution of cells per droplet.

![array of water droplets containing cells](images/test_array_lo_res.png)

My initial plan is to divide this into two tasks: detecting the water droplets and detecting the cells. Once both have been detected, I can associate each cell with its containing droplet and get the desired frequencies. Since the droplets are mostly circles, it seems detecting them should be straightforward, and OpenCV has a built-in tool for circle detection.