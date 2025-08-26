# faster_whisper

## Requirements
* Python 3.9 or greater
* It is recommended to use Anaconda or UV to manage and install in the environment.

## GPU
* If your laptop has a GPU, GPU execution requires the following NVIDIA libraries to be installed:

* [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
* [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)

**Note**: The latest versions of `ctranslate2` only support CUDA 12 and cuDNN 9. For CUDA 11 and cuDNN 8, the current workaround is downgrading to the `3.24.0` version of `ctranslate2`. For CUDA 12 and cuDNN 8, downgrade to the `4.4.0` version of `ctranslate2` (This can be done with `pip install --force-reinstall ctranslate2==4.4.0` or specifying the version in a `requirements.txt`).

There are multiple ways to install the NVIDIA libraries mentioned above. The recommended way is described in the official NVIDIA documentation, but we also suggest other installation methods below. 

* For Anaconda, please use the conda command to install the environment settings of these Python files.


## Reference
* Those versions of the model refer to Faster-whisper https://github.com/SYSTRAN/faster-whisper

## Version
### faster-whisper_v1 
* It is the original version of the code 

### faster-whisper_v2
* It can detect and transcribe your speech in real-time within 5 seconds.

### faster-whisper_v3
* It can detect and transcribe your speech in real-time when you stop speaking.

### faster-whisper_v4
* For the control of the Inspire Lab Robot from faster-whisper_v3, which means after stopping speaking, it will send a UDP command if you have spoken the keywords.
  
### faster-whisper_v4.1
* Same with faster-whisper_v4, but the duration time is 5 seconds, which is similar to the  version of faster-whisper_v2

### faster-whisper_v5
* This version is for the C++ file of the robot to call this Python script.

