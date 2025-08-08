import pyaudio

def list_microphones():
    p = pyaudio.PyAudio()
    print("\nAvailable microphones:")
    for i in range(p.get_device_count()):
        try:
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels', 0) > 0:
                print(f"{i}: {device_info['name']}")
        except Exception as e:
            print(f"Error getting device {i}: {e}")
    p.terminate()

if __name__ == '__main__':
    list_microphones()
