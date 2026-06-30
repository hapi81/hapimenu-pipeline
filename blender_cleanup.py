"""
blender_cleanup.py — rulat de Blender headless
Folosire: blender --background --python blender_cleanup.py -- input.obj output.glb
"""
import bpy
import sys
import os

def cleanup_and_export(input_path, output_path):
    print(f"[Blender] Input: {input_path}, exists: {os.path.exists(input_path)}")
    print(f"[Blender] Output target: {output_path}")

    # Curăță scena
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Importă .obj
    try:
        bpy.ops.wm.obj_import(filepath=input_path)
        print("[Blender] obj_import OK")
    except Exception as e:
        print(f"[Blender] obj_import FAILED: {e}")
        raise

    if len(bpy.context.selected_objects) == 0:
        print("[Blender] EROARE: niciun obiect selectat după import!")
        raise Exception("Niciun obiect importat din .obj")

    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    print(f"[Blender] Obiect activ: {obj.name}, verts: {len(obj.data.vertices) if hasattr(obj.data, 'vertices') else 'N/A'}")

    # Centrează la origine
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

    # Normalizează dimensiunea la max 1m
    max_dim = max(obj.dimensions)
    print(f"[Blender] max_dim: {max_dim}")
    if max_dim > 0:
        scale = 1.0 / max_dim
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True)

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Exportă .glb
    print(f"[Blender] Export către: {output_path}")
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_colors=True
    )

    print(f"[Blender] Export complet. Fișier există: {os.path.exists(output_path)}")
    if os.path.exists(output_path):
        print(f"[Blender] Dimensiune fișier: {os.path.getsize(output_path)} bytes")

if __name__ == "__main__":
    argv = sys.argv
    print(f"[Blender] sys.argv complet: {argv}")

    if "--" not in argv:
        print("[Blender] EROARE: nu există '--' în argumente!")
        sys.exit(1)

    argv = argv[argv.index("--") + 1:]
    print(f"[Blender] Argumente după --: {argv}")

    if len(argv) < 2:
        print(f"[Blender] EROARE: necesare 2 argumente (input, output), primite {len(argv)}")
        sys.exit(1)

    input_path = argv[0]
    output_path = argv[1]

    cleanup_and_export(input_path, output_path)
