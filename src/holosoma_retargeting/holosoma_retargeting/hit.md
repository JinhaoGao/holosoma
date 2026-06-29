7.1 Climbing
# 单条序列重定向
python examples/robot_retarget.py \
  --data_path demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_0 \
  --data_format mocap \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --save_dir demo_results/g1/climbing/mocap_climb \
  --retargeter.debug \
  --retargeter.visualize \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5

# 批量序列重定向
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/climb \
  --task-type climbing \
  --data_format mocap \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results_parallel/g1/climbing/mocap_climb \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5 \
  --max-workers 4
  
# 单条序列可视化
python viser_player.py \
  --robot_urdf models/g1/g1_29dof_spherehand.urdf \
  --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.74.urdf \
  --qpos_npz demo_results/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz \
  --show_mapped_skeletons \
  --show_interaction_mesh \
  --interaction_mesh_mode both \
  --interaction_mesh_edges cross \
  --interaction_mesh_line_width 1.5 \
  --robot_mesh_opacity 0.8 \
  --object_mesh_opacity 1

7.2 Robot-Only
7.2.1 OMOMO
# 单条序列重定向
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new \
  --task-type robot_only \
  --task-name sub3_largebox_003 \
  --data_format smplh \
  --save_dir demo_results/g1/robot_only/omomo \
  --retargeter.debug \
  --retargeter.visualize \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5

# 批量序列重定向  
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type robot_only \
  --data_format smplh \
  --save_dir demo_results_parallel/g1/robot_only/omomo \
  --task-config.object-name ground \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5 \
  --max-workers 8
  
# 单条序列可视化
python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
  --show_mapped_skeletons \
  --show_interaction_mesh \
  --interaction_mesh_mode both \
  --interaction_mesh_edges cross \
  --interaction_mesh_line_width 1.5 \
  --robot_mesh_opacity 0.8

7.2.2 LAFAN
# 单条序列重定向
python examples/robot_retarget.py \
  --data_path demo_data/lafan \
  --task-type robot_only \
  --task-name dance2_subject1 \
  --data_format lafan \
  --task-config.ground-range -10 10 \
  --save_dir demo_results/g1/robot_only/lafan \
  --retargeter.foot-sticking-tolerance 0.02 \
  --retargeter.debug \
  --retargeter.visualize \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5
 
# 批量序列重定向  
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/lafan \
  --task-type robot_only \
  --data_format lafan \
  --save_dir demo_results_parallel/g1/robot_only/lafan \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5 \
  --max-workers 8
  
# 单条序列可视化
python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz \
  --show_mapped_skeletons \
  --show_interaction_mesh \
  --interaction_mesh_mode both \
  --interaction_mesh_edges cross \
  --interaction_mesh_line_width 1.5 \
  --robot_mesh_opacity 0.8

7.2.3 AMASS
# 单条序列重定向
python examples/robot_retarget.py \
  --data_path demo_data/amass_smplx_processed \
  --task-type robot_only \
  --task-name ACCAD_Female1Running_c3d_C20_-__run_to_jump_to_walk_stageii \
  --data_format smplx \
  --task-config.ground-range -10 10 \
  --save_dir demo_results/g1/robot_only/amass_smplx \
  --retargeter.debug \
  --retargeter.visualize \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5

# 批量序列重定向  
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/amass_smplx_processed \
  --task-type robot_only \
  --data_format smplx \
  --save_dir demo_results_parallel/g1/robot_only/amass_smplx \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5 \
  --max-workers 8
  
# 单条序列可视化
python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz demo_results/g1/robot_only/amass_smplx/ACCAD_Female1Running_c3d_C20_-__run_to_jump_to_walk_stageii.npz \
  --show_mapped_skeletons \
  --show_interaction_mesh \
  --interaction_mesh_mode both \
  --interaction_mesh_edges cross \
  --interaction_mesh_line_width 1.5 \
  --robot_mesh_opacity 0.8
  
7.3 Object-Interaction
# 单条序列重定向
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub3_largebox_003 \
  --data_format smplh \
  --save_dir demo_results/g1/object_interaction/omomo \
  --retargeter.debug \
  --retargeter.visualize \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5
  
# 批量序列重定向  
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data_format smplh \
  --save_dir demo_results_parallel/g1/object_interaction/omomo \
  --task-config.object-name largebox \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross \
  --retargeter.interaction-mesh-line-width 1.5 \
  --max-workers 8
  
# 单条序列可视化
python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --object_urdf models/largebox/largebox.urdf \
  --qpos_npz demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --show_mapped_skeletons \
  --show_interaction_mesh \
  --interaction_mesh_mode both \
  --interaction_mesh_edges cross \
  --interaction_mesh_line_width 1.5 \
  --robot_mesh_opacity 0.8 \
  --object_mesh_opacity 1
