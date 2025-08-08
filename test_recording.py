import pyaudio
import wave
import time

def test_recording(device_index=0, duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    print(f"Testing recording with device index {device_index}")
    print("Recording for 5 seconds...")
    
    try:
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=device_index,
                       frames_per_buffer=CHUNK)
        
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
        print("Finished recording")
        
        stream.stop_stream()
        stream.close()
        
        # Save the recorded data as a WAV file
        wf = wave.open("test_recording.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print("Recording saved as test_recording.wav")
        
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        p.terminate()

if __name__ == '__main__':
    # Use the MacBook Pro Microphone (index 0)
    test_recording(device_index=0) 