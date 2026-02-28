import imageio

input_file = "static/demo_prototype.webp"
output_file = "static/demo_prototype.mp4"

print("Lecture du fichier webp...")
# Lire les images WebP (vidéo)
reader = imageio.get_reader(input_file)

# Définir le fps : on met 5 ce qui est typique des petites recs
fps = reader.get_meta_data().get('fps', 10)

print(f"Conversion en MP4 ({fps} fps)... Patientez...")
# Écrire en MP4
writer = imageio.get_writer(output_file, fps=fps, codec='libx264')

for i, frame in enumerate(reader):
    writer.append_data(frame)

writer.close()
reader.close()
print("Terminé ! Le fichier static/demo_prototype.mp4 a été créé.")
