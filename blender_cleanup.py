"""
blender_cleanup.py — rulat de Blender headless
Folosire: blender --background --python blender_cleanup.py -- input.obj output.glb
"""
import bpy
import sys

def cleanup_and_export(input_path, output_path):
    # Curăță scena
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Importă .obj
    bpy.ops.wm.obj_import(filepath=input_path)
    
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # Centrează la origine
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    
    # Normalizează dimensiunea la max 1m
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        scale = 1.0 / max_dim
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True)
    
    # Smooth shading
    bpy.ops.object.shade_smooth()
    
    # Exportă .glb
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_colors=True
    )
    
    print(f"Export .glb finalizat: {output_path}")

if __name__ == "__main__":
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    
    input_path = argv[0]
    output_path = argv[1]
    
    cleanup_and_export(input_path, output_path)
