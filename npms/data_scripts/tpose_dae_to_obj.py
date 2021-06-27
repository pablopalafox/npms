# # simple script to batch convert collada to obj.
# # run as:
# # blender --background --python data_scripts/tpose_dae_to_obj.py
# # or:
# # ./run_data_preprocess.sh

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import bpy
import mathutils
import math
import numpy as np

from data_scripts import config_data as cfg

def rotate_around_center(mat_rot, center):
    return (mathutils.Matrix.Translation( center) * 
            mat_rot * 
            mathutils.Matrix.Translation(-center))
    

if __name__ == "__main__":

    OVERWRITE = False

    print()

    identities = cfg.identities + cfg.identities_augmented
    # identities = ["mannequin"]

    # for i, character in enumerate(cfg.identities_consistent):
    for i, character in enumerate(identities):

        is_dae = True
        dae_filename = f"{cfg.root_in}/data/mixamo/{character}/raw_dae/a_t_pose_000001.dae"

        if not os.path.isfile(dae_filename):

            # Maybe it's an .fbx
            dae_filename = f"{cfg.root_in}/data/mixamo/{character}/raw_dae/a_t_pose_000001.fbx"
            
            if not os.path.isfile(dae_filename):
                
                print()
                print()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Could not find", character)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print()
                print()
                continue
            
            is_dae = False

        # Create output folder 
        obj_dir = f"{cfg.root_in}/data/mixamo/{character}/obj/a_t_pose"
        obj_filename = f"{obj_dir}/a_t_pose_000001.obj"

        if not os.path.isdir(obj_dir):
            os.makedirs(obj_dir)
            
        # Skip if it already exists
        if not OVERWRITE and os.path.isfile(obj_filename):
            print()
            print()
            print("##############################################################################")
            print("##############################################################################")
            print("Skipping", character)
            print("##############################################################################")
            print("##############################################################################")
            print()
            print()
            continue

        print()
        print()
        print("##############################################################################")
        print("##############################################################################")
        print(character)
        print("##############################################################################")
        print("##############################################################################")
        print()
        print()

        # Initialize blender (delete everything that is currently in the scene)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Import our sequence
        if is_dae:
            bpy.ops.wm.collada_import(filepath=dae_filename)
        else:
            bpy.ops.import_scene.fbx(filepath=dae_filename)

        ########################################################################
        C = bpy.context

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Select the armature
        C.scene.objects['Armature'].select_set(True)
        # Also activate it
        arm = C.window.scene.objects["Armature"]
        C.view_layer.objects.active = arm

        # Pose mode
        bpy.ops.object.mode_set(mode='POSE')

        for pbone in arm.pose.bones:
            if "RightUpLeg" in pbone.name:
                angle = 0.436332
            elif "LeftUpLeg" in pbone.name:
                angle = -0.436332
            else: 
                continue

            print(pbone.name, angle)

            # pbone = arm.pose.bones["mixamorig_LeftUpLeg"]
            pbone.bone.select = True
            bpy.ops.transform.rotate(value=angle, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
            pbone.bone.select = False
            
        # Apply the rotation
        for o in C.scene.objects:
            o.select_set(True)
        bpy.ops.object.convert(target='MESH')

        # ########################################################################

        # Export scene
        bpy.ops.export_scene.obj(filepath=obj_filename, use_animation=False, use_materials=False)