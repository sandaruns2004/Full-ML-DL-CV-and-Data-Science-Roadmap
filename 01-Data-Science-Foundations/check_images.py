import glob
from PIL import Image

def check_images():
    for file in glob.glob('c:/Users/ADMIN/Desktop/Full-ML-DL-CV-and-Data-Science-Roadmap/01-Data-Science-Foundations/assets/*.png'):
        try:
            with Image.open(file) as img:
                img.verify()
            print(f"{file} is valid.")
        except Exception as e:
            print(f"{file} is INVALID: {e}")

if __name__ == '__main__':
    check_images()
