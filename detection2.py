from retinaface import RetinaFace
import matplotlib.pyplot as plt

faces = RetinaFace.extract_faces(img_path="img.png", align=True)
for face in faces:
    plt.imshow(face)
    plt.show()
