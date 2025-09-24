import subprocess
import os
from pydub import AudioSegment
import numpy as np
import struct

input_file = "sample_data/sample.mp4"
extracted_audio = "extracted_audio.wav"
encoded_audio = "encoded_audio.wav"
encoded_video = "encoded_video.mov"

input_encode_file = "encoded_video.mov"
encoded_extracted_audio = "encoded_extracted_audio.wav"

input_encoded_file = ""

encoded_message = "This is a secret message to embed in the video file."
extracted_message = None

# Extract audio from video file
input_extract_command = [
  "ffmpeg",
  "-i", input_file,
  "-vn",  # Exclude video stream
  "-acodec", "pcm_s32le",  # Specify audio codec and format
  extracted_audio
]

try:
  result = subprocess.run(input_extract_command, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    print(f"Error extracting audio: ffmpeg returned exit code {result.returncode}")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
  else:
    print(f"Audio extracted successfully to: {extracted_audio}")

except FileNotFoundError:
  print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
except Exception as e:
  print(f"An unexpected error occurred: {e}")

# Encode Audio
try:
  audio = AudioSegment.from_wav(extracted_audio)
  print("Audio file loaded successfully.")
except Exception as e:
  print(f"Error loading audio file: {e}")
  audio = None

if audio:
  binary_message = ''.join(format(ord(char), '08b') for char in encoded_message)
  delimiter = '1111111111111110'
  binary_message += delimiter

  #print(f"Original text message: '{text_message}'")
  #print(f"Binary message (with delimiter): {binary_message}")

  sample_width = audio.sample_width
  if sample_width == 2: # 16-bit audio
    dtype = np.int16
  elif sample_width == 4: # 32-bit audio (Default)
    dtype = np.int32
  else:
    raise ValueError("Unsupported sample width")
  audio_data = np.array(audio.get_array_of_samples()).astype(dtype)

  # Embed the binary message into the LSB of audio samples
  # Ensure there are enough samples to embed the message
  if len(audio_data) < len(binary_message):
    print("Error: Audio file is too short to embed the message.")
    embedded_audio = None
  else:
    # Iterate through the binary message and modify LSB of audio samples
    embedded_audio_data = audio_data.copy()
    for i in range(len(binary_message)):
      # Get the current sample
      sample = embedded_audio_data[i]

      # Get the i-th bit of the binary message (0 or 1)
      bit = int(binary_message[i])

      # Clear the LSB of the sample
      sample = (sample & ~1)

      # Set the LSB of the sample to the current bit
      sample = (sample | bit)

      # Update the sample in the embedded audio data
      embedded_audio_data[i] = sample

    # Convert the modified numpy array back to AudioSegment
    embedded_audio = AudioSegment(
      embedded_audio_data.tobytes(),
      frame_rate=audio.frame_rate,
      sample_width=sample_width,
      channels=audio.channels
    )
    print("Text message embedded into audio.")


  if embedded_audio:
    try:
      embedded_audio.export(encoded_audio, format="wav")
      print(f"Embedded audio saved successfully to: {encoded_audio}")
    except Exception as e:
      print(f"Error saving embedded audio file: {e}")

# Export Embedded Video (MOV)
encode_video_command = [
  "ffmpeg",
  "-i", input_file,       # Input video file
  "-i", encoded_audio,    # Input audio file
  "-i", encoded_audio,    # Input audio file
  "-c:v", "copy",         # Copy video stream without re-encoding
  "-c:a:0", "copy",       
  "-c:a:1", "aac",        # Encode audio stream using AAC codec (for playability)
  "-map", "0:v:0",        # Map the first video stream from the first input (video file)
  "-map", "1:a:0",        # Map the first audio stream from the second input (audio file)
  "-map", "2:a:0",        # Add encoded audio to second audio stream
  "-shortest",            # Finish encoding when the shortest input stream ends
  "-f", "mov",
  "-y",                   # Overwrite output file without asking
  encoded_video       # Output video file
]

try:
  result = subprocess.run(encode_video_command, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    print(f"Error combining video and audio: ffmpeg returned exit code {result.returncode}")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
  else:
    print(f"Video and audio combined successfully and saved to: {encoded_video}")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
except FileNotFoundError:
  print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
except Exception as e:
  print(f"An unexpected error occurred: {e}")

# Extract audio from video file (Decoding)
input_encoded_extract_command = [
  "ffmpeg",
  "-i", input_encode_file,
  "-vn",  # Exclude video stream
  "-map", "0:a:0", # Select only uncompressed stream
  "-c:a", "copy",
  "-y",
  encoded_extracted_audio
]

try:
  result = subprocess.run(input_encoded_extract_command, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    print(f"Error extracting audio: ffmpeg returned exit code {result.returncode}")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
  else:
    print(f"Audio extracted successfully to: {encoded_extracted_audio}")

except FileNotFoundError:
  print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
except Exception as e:
  print(f"An unexpected error occurred: {e}")

# Extract Encoded Message
try:
  audio = AudioSegment.from_wav(encoded_extracted_audio)
  print("Audio file loaded successfully.")
except Exception as e:
  print(f"Error loading audio file: {e}")
  audio = None

if audio:
  sample_width = audio.sample_width
  if sample_width == 2: # 16-bit audio
    dtype = np.int16
  elif sample_width == 4: # 32-bit audio (Default)
    dtype = np.int32
  else:
    raise ValueError("Unsupported sample width")

  if dtype:
    audio_data = np.array(audio.get_array_of_samples()).astype(dtype)

    binary_message_bits = ""
    for sample in audio_data:
        # Extract the LSB (least significant bit)
        lsb = sample & 1
        binary_message_bits += str(lsb)

    # print(f"Extracted binary bits (may contain message and delimiter): {binary_message_bits[:200]}...")

    delimiter = '1111111111111110'
    delimiter_index = binary_message_bits.find(delimiter)

    if delimiter_index != -1:
      # 6. Extract the binary message before the delimiter
      binary_message = binary_message_bits[:delimiter_index]
      print(f"Extracted binary message (before delimiter): {binary_message}")

      extracted_message = ""
      # Ensure the binary message length is a multiple of 8 for byte conversion
      if len(binary_message) % 8 == 0:
        for i in range(0, len(binary_message), 8):
          byte = binary_message[i:i+8]
          try:
              extracted_message += chr(int(byte, 2))
          except ValueError:
              # Handle cases where a byte might not be a valid character (unlikely if embedding was correct)
              print(f"Warning: Could not convert binary byte '{byte}' to character.")
              extracted_message = None # Indicate failure if conversion fails
              break # Stop processing if we encounter an invalid byte
      else:
          print("Error: Extracted binary message length is not a multiple of 8.")
          extracted_message = None

      print(f"\nExtracted Text Message: '{extracted_message}'")
    else:
      print("Delimiter not found in the extracted binary bits.")
      print("Extracted Text Message: None (Delimiter not found)")

  else:
      print("Could not process audio due to unsupported sample width.")
else:
  print("Could not proceed with text extraction as audio loading failed.")
