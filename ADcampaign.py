# main.py
# Simple AI-Generated Nike Serena Williams Ad Campaign

# Install these before running:
# pip install -r requirements.txt

# Imports
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from moviepy.editor import ImageSequenceClip, AudioFileClip
from gtts import gTTS
import os

# 1️⃣ Generate ad script using GPT
print("Generating ad script...")

generator = pipeline('text-generation', model='gpt2')

prompt = """
Write a short and inspiring script for an AI-generated Nike ad featuring Serena Williams.
Theme: resilience, determination, empowerment, legacy.
"""

script_output = generator(prompt, max_length=150, num_return_sequences=1)
script_text = script_output[0]['generated_text']
print("\nGenerated Script:\n")
print(script_text)

# 2️⃣ Generate AI images for key phrases
print("\nGenerating AI images...")

# Load Stable Diffusion
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
)
sd_pipeline.to("cuda")  # Use GPU

# Example key phrases (simple version)
key_phrases = [
    "Serena Williams on a tennis court at night",
    "Close-up of Serena Williams showing determination",
    "Nike logo with empowering background",
]

# Make outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

image_files = []

for idx, phrase in enumerate(key_phrases):
    image = sd_pipeline(phrase).images[0]
    filename = f"outputs/frame_{idx}.png"
    image.save(filename)
    image_files.append(filename)
    print(f"Saved image: {filename}")

# 3️⃣ Generate voiceover using Google TTS
print("\nGenerating voiceover...")

tts = gTTS(script_text)
voiceover_path = "outputs/voiceover.mp3"
tts.save(voiceover_path)
print(f"Saved voiceover: {voiceover_path}")

# 4️⃣ Combine images and voiceover into video
print("\nCreating final video...")

clip = ImageSequenceClip(image_files, fps=1)  # 1 image per second
audio_clip = AudioFileClip(voiceover_path)

final_clip = clip.set_audio(audio_clip)
final_clip.write_videofile("outputs/serena_nike_ad.mp4", fps=24)

print("\nDone! Your ad video is ready: outputs/serena_nike_ad.mp4")

# Optional cleanup (if you want)
# for f in image_files:
#     os.remove(f)
# os.remove(voiceover_path)

