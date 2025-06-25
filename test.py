import pyvips

# Vérifie la version
print("✅ pyvips version:", pyvips.__version__)

# Crée une petite image noire 100x100 avec un point blanc
image = pyvips.Image.black(100, 100).draw_circle(255, 50, 50, 10, fill=True)

# Sauvegarde en PNG pour voir le résultat (facultatif)
image.write_to_file("test_pyvips_output.png")
print("✅ Image test créée : test_pyvips_output.png")
