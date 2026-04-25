from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def show_img(img):
    img_array = np.array(img)
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")
    plt.show()

def center_by_mass(img):
    arr = np.array(img.convert('L'))  # ← forcer en 2D
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return img
    dx = int(14 - xs.mean())
    dy = int(14 - ys.mean())
    img = ImageOps.expand(img, border=20, fill=0)
    img = img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy))
    w, h = img.size
    return img.crop(((w-28)//2, (h-28)//2, (w+28)//2, (h+28)//2))

def mnist_normalize(img):
    arr = np.array(img.convert('L'))
    
    # 1. Crop sur le contenu puis resize dans 20x20
    bbox = img.getbbox()
    img = img.crop(bbox)
    img.thumbnail((20, 20), Image.BILINEAR)  # conserve les proportions dans 20x20
    
    # 2. Placer dans 28x28 vide
    canvas = Image.new('L', (28, 28), 0)
    offset = ((28 - img.width) // 2, (28 - img.height) // 2)
    canvas.paste(img, offset)
    
    # 3. Centrage par centre de masse
    canvas = center_by_mass(canvas)
    return canvas

img_path = "7.jpg"  # Chemin vers votre image
img = Image.open(img_path)

#Etape 1 : Convertir en niveaux de gris
img_gray = img.convert('L')  # Convertir en niveaux de gris (8-bit pixels, noir et blanc)

#Etape 2 : Contraste à 100%
img_contrasted = img.point(lambda p: 255 if p > 127 else 0)  # Appliquer un seuil pour augmenter le contraste (seuil à 127 pour séparer les pixels clairs et foncés)

#Etape 3 : Inverser les couleurs
img_inverted = ImageOps.invert(img_contrasted)  # Inverser les couleurs (noir devient blanc et vice versa)

bbox = img_inverted.getbbox()  # Obtenir les coordonnées de la boîte englobante
img_cropped = img_inverted.crop(bbox)  # Recadrer l'image en utilisant la boîte englobante

pad = int(max(img_cropped.size)*0.15)  # Trouver la taille maximale pour le padding
img_cropped = ImageOps.expand(img_cropped, border=pad, fill=0)  # Ajouter du padding noir pour rendre l'image carrée

#Etape 4 : Redimensionner à 28x28 pixels (Bilinéaire)
img = img_cropped.resize((28, 28), Image.BILINEAR)
img = center_by_mass(img)  # Centrer l'image en utilisant la méthode de la masse

#Etape 5 : Luminosité à 100% et Contraste à 80%
img = ImageEnhance.Brightness(img).enhance(1.9)  # Luminosité à 100%
img = ImageEnhance.Contrast(img).enhance(2.2)  # Contraste à 80%


img.save("7_transformed.png")  # Enregistrer l'image transformée

mnist_normalize(img).save("7_mnist_normalized.png")  # Enregistrer l'image normalisée pour MNIST

fig, axes = plt.subplots(1, 5, figsize=(10, 5))
axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("Image en niveaux de gris")
axes[0].axis("off")

axes[1].imshow(img_contrasted, cmap="gray")
axes[1].set_title("Image avec contraste à 100%")
axes[1].axis("off")

axes[2].imshow(img_inverted, cmap="gray")
axes[2].set_title("Image avec couleurs inversées")
axes[2].axis("off")

axes[3].imshow(img, cmap="gray")
axes[3].set_title("Image redimensionnée à 28x28 pixels")
axes[3].axis("off")

axes[4].imshow(mnist_normalize(img), cmap="gray")
axes[4].set_title("Image normalisée pour MNIST")
axes[4].axis("off")

plt.tight_layout()
plt.show()

"""
 1. Noir et blanc
 2. Contraste -> 100%
 3. Inverser les couleurs
 4. Redimensionner à 28x28 pixels (Bilinéaire)
 5. Luminosité -> 100% | Contraste -> 80%
"""