import os

# Requirements install karna
print("Installing dependencies... Please wait.")
os.system("pip install -r requirements.txt")

# Software Start karna
print("Starting VibeVoice Premium...")
# Aapke folder ka rasta (path) yahan set hai
os.system("python vibevoice_premium/app.py")
