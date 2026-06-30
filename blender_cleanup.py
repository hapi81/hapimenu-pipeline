"""
blender_cleanup.py — rulat de Blender headless
Folosire: blender --background --python blender_cleanup.py -- input.obj output.glb
"""
import bpy
import sys
import os

def cleanup_and_export(input_path, output_path):
    print(f"[Blender] Input: {input_path}, exists: {os.path.exists(input_path)}", flush=True)
    print(f"[Blender] Output target: {output_path}", flush=True)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    try:
        bpy.ops.wm.obj_import(filepath=input_path)
        print("[Blender] obj_import OK", flush=True)
    except Exception as e:
        print(f"[Blender] obj_import FAILED: {e}", flush=True)
        raise

    if len(bpy.context.selected_objects) == 0:
        print("[Blender] EROARE: niciun obiect selectat după import!", flush=True)
        raise Exception("Niciun obiect importat din .obj")

    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    print(f"[Blender] Obiect activ: {obj.name}", flush=True)

    # SKIP origin_set și transform_apply — pot crăpa pe mesh-uri mari
    # Doar mutăm direct fără transformări complexe
    print("[Blender] Skip origin/scale transforms pentru stabilitate", flush=True)

    print(f"[Blender] Export către: {output_path}", flush=True)
    try:
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            use_selection=False,
        )
        print("[Blender] export_scene.gltf call returned", flush=True)
    except Exception as e:
        print(f"[Blender] export FAILED: {e}", flush=True)
        raise

    print(f"[Blender] Export complet. Fișier există: {os.path.exists(output_path)}", flush=True)
    if os.path.exists(output_path):
        print(f"[Blender] Dimensiune fișier: {os.path.getsize(output_path)} bytes", flush=True)
    else:
        print("[Blender] EROARE CRITICĂ: fișierul nu există după export!", flush=True)

if __name__ == "__main__":
    argv = sys.argv
    print(f"[Blender] sys.argv complet: {argv}", flush=True)

    if "--" not in argv:
        print("[Blender] EROARE: nu există '--' în argumente!", flush=True)
        sys.exit(1)

    argv = argv[argv.index("--") + 1:]
    print(f"[Blender] Argumente după --: {argv}", flush=True)

    if len(argv) < 2:
        print(f"[Blender] EROARE: necesare 2 argumente, primite {len(argv)}", flush=True)
        sys.exit(1)

    input_path = argv[0]
    output_path = argv[1]

    cleanup_and_export(input_path, output_path)
    print("[Blender] Script complet finalizat.", flush=True)
